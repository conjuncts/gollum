import asyncio
from typing import TYPE_CHECKING, Dict, List, Literal


from gollum.batch.job import BatchJob
from gollum.batch.splitting_method import SplittingMethod
from gollum.batch.storage.batch_storage import BatchStorage
from gollum.permacache.base import Permacache
from gollum.permacache.cache_method import CacheMethod
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import BatchWorker

if TYPE_CHECKING:
    pass


class BatchHandler:
    """
    batch handler is a weird intermediary between worker and worklist. it should see globally.
    it must have access to permacache (to )
    """

    def __init__(
        self,
        batch_storage: BatchStorage,
        batch_worker: BatchWorker,
        splitting_method=None,
        permacache: Permacache = None,
        cache_method: CacheMethod = None,
        strategy: Literal["interrupt", "poll"] = "poll",
        polling_frequency: float = 60.0, # poll every 60 seconds
    ):
        self.batch_storage = batch_storage
        self.batch_worker: BatchWorker = batch_worker
        self.splitting_method = (
            splitting_method
            if splitting_method is not None
            else SplittingMethod(1000)
        )
        self.strategy: Literal["interrupt", "poll"] = strategy
        self.permacache = permacache
        self.cache_method = cache_method
        self.reconnection_board: Dict[str, WorklistEntry] = {}

        self._actively_polling = False
        self._polling_task: asyncio.Task | None = None
        self.polling_frequency = polling_frequency

        self._tracked_jobs = []

    async def _restart_polling(self):
        """
        Restarts polling loop
        """
        if self._polling_task is not None:
            self._polling_task.cancel()

        self._polling_task = asyncio.create_task(self._poll_loop())
        self._actively_polling = True

    async def _poll_loop(self):
        while self._actively_polling:
            await self._poll_point_check()
            await asyncio.sleep(self.polling_frequency)



    async def _poll_point_check(self):
        """
        Poll point check of all batches; also handles receiving results.
        """
        # Point check of all batches
        jobs = await self.batch_storage.get_all_batches()
        for job in jobs:
            result = await self.batch_worker.check_batch(job)
            if result.status == "pending":
                # TODO
                pass
            elif result.status == "completed":
                for cache_key, result in zip(result.cache_keys, result.results):
                    await self.permacache.store(result, cache_key, "")

                    # check the reconnection board to see if any worklist entries are waiting for this result
                    if cache_key in self.reconnection_board:
                        entry = self.reconnection_board[cache_key]
                        entry.finish(result)
                        del self.reconnection_board[cache_key]

                await self.batch_storage.free_batch(job)

    async def submit_entries(self, entries: list[WorklistEntry]) -> None:

        if not entries:
            return

        mini_batches = self.splitting_method.split(entries)

        # just use worker 0
        jobs: List[BatchJob] = []
        for mini_batch in mini_batches:
            job = await self.batch_worker.send_batch(mini_batch)
            jobs.append(job)
            cache_keys = [self.cache_method.generate_cache_key(item.request) for item in mini_batch]
            self.batch_storage.record_batch(job, cache_keys)

        if self.strategy == "interrupt":
            # these entries are now in a batch job, 
            # we can only interrupt if the user demands an immediate answer
            for entry in entries:
                entry._must_interrupt = True

    async def reconnect(self, entry: WorklistEntry):
        """
        Reconnects a worklist entry to its batch job if it was previously disconnected.
        Reconnecting means that once the batch job completes, it will fill the future of this worklist entry with the result.
        """
        # if permacache has it, that means that it already came it, we're done!
        # TODO: I am a bit paranoid about interleaving 
        result = await self.permacache.retrieve(self.cache_method.generate_cache_key(entry.request), "")
        if result is not None:
            entry.finish(result)
            return

        # otherwise, put it on the reconnection board
        cache_key = self.cache_method.generate_cache_key(entry.request)
        job = await self.batch_storage.retrieve_batch(cache_key, "")
        if job is not None:
            # Reconnect the entry to the batch job
            self.reconnection_board[cache_key] = entry
