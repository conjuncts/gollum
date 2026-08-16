"""
Lightweight tests for BatchWorklist -- no real DB and no real provider
client. Storage is an in-memory fake implementing the BatchStorage ABC,
and the "provider" is a hand-rolled fake BatchWorker, so these run fast
and fully offline.
"""

import asyncio
from typing import Dict, List, Optional, Tuple

import pytest

from gollum.batch.batch_handler import BatchHandler
from gollum.batch.batch_trigger import SizeTrigger
from gollum.batch.job import BatchJob
from gollum.batch.result import BatchResult
from gollum.batch.storage.batch_storage import BatchStorage
from gollum.permacache.cache_method import CacheMethod
from gollum.types import GollumRequest, GollumResponse
from gollum.worklist.batch_worklist import BatchWorklist


class InMemoryBatchStorage(BatchStorage):
    """Minimal BatchStorage fake: dicts instead of DuckDB tables."""

    def __init__(self):
        self._batches: Dict[str, Tuple[BatchJob, bool]] = {}
        self._keys: Dict[str, str] = {}  # cache_key -> batch_id

    async def record_batch(self, batch: BatchJob, cache_keys: List[str]):
        self._batches[batch.batch_id] = (batch, False)
        for key in cache_keys:
            self._keys[key] = batch.batch_id

    async def retrieve_batch(self, cache_key: str, likely_partition: str) -> Optional[BatchJob]:
        batch_id = self._keys.get(cache_key)
        if batch_id is None:
            return None
        return self._batches[batch_id][0]

    async def complete_batch(self, batch: BatchJob):
        job, _ = self._batches[batch.batch_id]
        self._batches[batch.batch_id] = (job, True)

    async def free_completed(self):
        done_ids = [bid for bid, (_, completed) in self._batches.items() if completed]
        for bid in done_ids:
            del self._batches[bid]
        self._keys = {k: v for k, v in self._keys.items() if v not in done_ids}

    async def get_all_batches(self) -> List[BatchJob]:
        return [job for job, completed in self._batches.values() if not completed]


class DictCacheMethod(CacheMethod):
    """Cache key is just whatever the caller stuffed into request.metadata."""

    def generate_cache_key(self, request: GollumRequest) -> str:
        return request.metadata["id"]


class FakeBatchWorker:
    """Fake provider: 'sends' a batch by remembering its cache_keys, and
    reports it complete (with a canned response per key) on the first
    check_batch() call -- no polling delay, no network."""

    def __init__(self):
        self.sent: Dict[str, List[str]] = {}
        self._counter = 0

    async def send_batch(self, worklist_entries) -> BatchJob:
        self._counter += 1
        batch_id = f"job-{self._counter}"
        self.sent[batch_id] = [
            DictCacheMethod().generate_cache_key(e.request) for e in worklist_entries
        ]
        return BatchJob(batch_id, "fake-provider")

    async def check_batch(self, batch_job: BatchJob) -> BatchResult:
        cache_keys = self.sent[batch_job.batch_id]
        results = [
            GollumResponse({"model": "m", "choices": []}, extras={}, metadata={})
            for _ in cache_keys
        ]
        return BatchResult(status="completed", cache_keys=cache_keys, results=results)


class InMemoryPermacache:
    def __init__(self):
        self._store: Dict[str, GollumResponse] = {}

    async def retrieve(self, cache_key: str, likely_partition: str):
        return self._store.get(cache_key)

    async def store(self, value, cache_key: str, likely_partition: str):
        self._store[cache_key] = value


def make_request(entry_id: str) -> GollumRequest:
    return GollumRequest(
        chat_completion={"model": "gpt-test", "messages": []},
        extras={},
        metadata={"id": entry_id},
        provider_name="openai",
    )


@pytest.mark.asyncio
async def test_batch_worklist_flushes_and_resolves_entries():
    storage = InMemoryBatchStorage()
    permacache = InMemoryPermacache()
    worker = FakeBatchWorker()

    handler = BatchHandler(
        storage,
        worker,
        permacache=permacache,
        cache_method=DictCacheMethod(),
        polling_frequency=0.05,
    )
    worklist = BatchWorklist(handler, trigger=SizeTrigger(2))
    await worklist.start()

    entries = [await worklist.enroll(make_request(f"k{i}")) for i in range(2)]

    # SizeTrigger(2) should have auto-flushed once the second entry enrolled.
    results = await asyncio.wait_for(asyncio.gather(*entries), timeout=2)
    assert len(results) == 2
    assert all(isinstance(r, GollumResponse) for r in results)

    # Results are durably cached now.
    assert await permacache.retrieve("k0", "") is not None
    assert await permacache.retrieve("k1", "") is not None

    await worklist.shutdown()


@pytest.mark.asyncio
async def test_batch_worklist_survives_restart_via_reconnect():
    """
    Simulates a process restart: entry A is submitted and its batch_id is
    durably recorded, but the process "dies" before the batch finishes (no
    JobState survives -- a fresh BatchHandler/BatchWorklist is built on top
    of the same storage). A brand-new worklist must still be able to
    reconnect() to the still-recorded batch and resolve once it completes,
    without resubmitting.
    """
    storage = InMemoryBatchStorage()
    permacache = InMemoryPermacache()
    worker = FakeBatchWorker()
    cache_method = DictCacheMethod()

    # "Session 1": submit, but never poll to completion (simulates a crash).
    handler_1 = BatchHandler(
        storage, worker, permacache=permacache, cache_method=cache_method,
        polling_frequency=100.0,  # effectively never fires during the test
    )
    worklist_1 = BatchWorklist(handler_1, trigger=SizeTrigger(1))
    await worklist_1.start()

    entry = await worklist_1.enroll(make_request("kA"))
    assert entry.status != "done"

    await handler_1.stop_polling()  # simulate the old process going away

    # "Session 2": brand-new handler/worklist over the SAME storage.
    handler_2 = BatchHandler(
        storage, worker, permacache=permacache, cache_method=cache_method,
        polling_frequency=0.05,
    )
    worklist_2 = BatchWorklist(handler_2, trigger=SizeTrigger(1))
    await worklist_2.start()  # frees nothing yet (batch isn't marked complete)

    # A fresh entry for the same cache_key reconnects to the still-recorded
    # batch instead of being resubmitted.
    reconnected = await worklist_2.enroll(make_request("kA"))
    assert len(worker.sent) == 1  # still only ever sent once

    result = await asyncio.wait_for(reconnected, timeout=2)
    assert isinstance(result, GollumResponse)

    await worklist_2.shutdown()
