import asyncio
import itertools

from gollum.worklist.base import WorklistEntry
from gollum.worklist.worklist import Worklist


class ConcurrentWorklist(Worklist):
    """
    Concurrent worklist that round-robins the worker try-order across entries.

    Rotation state (`_rr_counter`) is advanced synchronously with no `await`
    in between, so -- like `kickstart_work`'s enroll-draining loop -- it's
    safe under asyncio's single-threaded cooperative scheduling without a
    lock; nothing can interleave between reading and advancing the counter.
    """

    def __init__(self, max_concurrency: int | None = None):
        super().__init__()
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        # Guards cache reads/writes so a lookup can't race a concurrent record()
        # for the same entry. Cheap to hold since cache_worker calls are the
        # only shared-state touchpoints across concurrent entries.
        self._cache_lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task] = set()
        # Monotonic counter used to pick each entry's starting worker index.
        self._rr_counter = itertools.count()

    async def kickstart_work(self, entry: WorklistEntry):
        """
        Drain any newly-enrolled entries into tasks and return immediately --
        do NOT await the tasks here. Awaiting would make enroll() block until
        that entry finishes, which serializes concurrently-enrolled entries
        behind each other and defeats the point of this subclass.

        Note this loop has no `await` in its body, so it's atomic with
        respect to other coroutines even though multiple `enroll()` calls
        each invoke `kickstart_work()`: nothing can interleave between the
        `while` check and the `pop(0)`.
        """
        if not self.workers:
            raise ValueError("No workers available to process entries.")

        task = asyncio.create_task(self._process_entry(entry))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _process_entry(self, entry: WorklistEntry):
        if self._semaphore is not None:
            async with self._semaphore:
                await self._process_entry_inner(entry)
        else:
            await self._process_entry_inner(entry)

    def _next_worker_order(self) -> list:
        """
        Return `self.workers` rotated so consecutive calls start from a
        different worker (wrapping around), e.g. for workers [A, B, C]:
        call 1 -> [A, B, C], call 2 -> [B, C, A], call 3 -> [C, A, B], ...
        """
        n = len(self.workers)
        start = next(self._rr_counter) % n
        return self.workers[start:] + self.workers[:start]

    async def _process_entry_inner(self, entry: WorklistEntry):
        # cache hit
        if self.cache_worker is not None:
            async with self._cache_lock:
                if await self.cache_worker.process(entry):
                    return

        # live processing, starting from the round-robin-selected worker
        for worker in self._next_worker_order():
            if await worker.process(entry):
                break
        else:
            raise ValueError("No worker could process this entry:", entry)

        # record value
        if self.cache_worker is not None:
            async with self._cache_lock:
                await self.cache_worker.record(entry)

    async def join(self):
        """Wait for all currently in-flight entries to finish processing."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks)
