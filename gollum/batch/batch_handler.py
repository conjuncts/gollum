import asyncio
from typing import TYPE_CHECKING, Dict, List, Literal, Optional


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
        permacache: Optional[Permacache] = None,
        cache_method: Optional[CacheMethod] = None,
        strategy: Literal["interrupt", "poll"] = "poll",
        polling_frequency: float = 60.0,  # poll every 60 seconds
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

        # cache_key -> list of entries waiting on that key. A list (not a
        # single entry) because more than one WorklistEntry can legitimately
        # be waiting on the same cache_key (duplicate/identical requests).
        self.reconnection_board: Dict[str, List[WorklistEntry]] = {}

        # Guards the "check permacache, else register as a waiter on
        # reconnection_board" critical section, and is held by the poll loop
        # while it drains completed results into permacache / the board.
        # Without this lock there's a TOCTOU race: reconnect() can check
        # permacache (miss), then _poll_point_check() can store the result
        # and sweep the board (entry not registered yet), then reconnect()
        # registers the entry *after* the sweep already happened -> the
        # entry never gets finished.
        self._board_lock = asyncio.Lock()

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
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self._actively_polling = True
        self._polling_task = asyncio.create_task(self._poll_loop())

    async def stop_polling(self):
        """
        Stops the polling loop cleanly.
        """
        self._actively_polling = False
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None

    async def _poll_loop(self):
        while self._actively_polling:
            await self._poll_point_check()
            await asyncio.sleep(self.polling_frequency)

    async def _poll_point_check(self):
        """
        Poll point check of all batches; also handles receiving results.
        """
        jobs = await self.batch_storage.get_all_batches()
        for job in jobs:
            check_result = await self.batch_worker.check_batch(job)

            if check_result.status == "pending":
                continue

            elif check_result.status == "completed":
                async with self._board_lock:
                    # NOTE: when a batch comes in, it could take a long time... then the lock is held up
                    for cache_key, item_result in zip(check_result.cache_keys, check_result.results):
                        await self.permacache.store(item_result, cache_key, "")

                        waiters = self.reconnection_board.pop(cache_key, None)
                        if waiters:
                            for entry in waiters:
                                entry.finish(item_result)

                    await self.batch_storage.free_batch(job)

            else:
                # e.g. "failed" or any other terminal/unknown status.
                # TODO: decide how failures should propagate to waiting
                # WorklistEntry objects (entry.fail(...)? raise?) -- currently
                # these entries would otherwise hang forever with no signal.
                async with self._board_lock:
                    for cache_key in getattr(check_result, "cache_keys", []):
                        waiters = self.reconnection_board.pop(cache_key, None)
                        if waiters:
                            for entry in waiters:
                                # Placeholder: replace with your real failure API.
                                raise RuntimeError(
                                    "placeholder: batch failed, need to mark entry as failed"
                                )

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
            await self.batch_storage.record_batch(job, cache_keys)

        if self.strategy == "interrupt":
            # these entries are now in a batch job,
            # if the user demands an immediate answer, our only recourse is to interrupt
            for entry in entries:
                entry._must_interrupt = True

    async def reconnect(self, entry: WorklistEntry) -> bool:
        """
        Reconnects a worklist entry to its batch job if it was previously disconnected.
        Reconnecting means that once the batch job completes, it will fill the future of this worklist entry with the result.

        This typically happens once the WorklistEntry is created as it comes in (ie. in kickstart_work) to make sure
        that .


        If we can guarantee a lifecycle such 
        1. WorklistEntry created 
        2. reconnect() is called to connect it to a stored batch job - ie. from a previous session
            Suppose reconnection is successful: that is, a previous batch job exists.
            Since that stored batch job is being polled, eventually it will complete
        3. if cannot reconnect, then this entry must be put in a batch! ==> will be sent into a batch
            and hence (eventually) that batch will be polled and completed.

        Then every entry will eventually complete.

        :return: True if the entry was successfully reconnected to a batch job, False otherwise.
        """
        cache_key = self.cache_method.generate_cache_key(entry.request)

        async with self._board_lock:
            # this is in the hot loop (once per request)...
            # if a batch comes in, then the lock means that the hot loop is completely held up...
            # this can cause a significant performance bottleneck.

            # if permacache has it, that means it already came in, we're done!
            result = await self.permacache.retrieve(cache_key, "")
            if result is not None:
                entry.finish(result)
                return True

            # otherwise, put it on the reconnection board -- but only if we
            # can actually find the batch job it belongs to. If we can't,
            # registering it as a waiter would just make it hang forever
            # with no completion event to ever wake it up.
            job = await self.batch_storage.retrieve_batch(cache_key, "")
            if job is None:
                # This entry is NOT associated with a batch job
                # Therefore, returning false will signal to the caller that this entry should be put into a new batch job.
                return False
            elif self.strategy == "interrupt":
                # the batch job is in progress. If the user wants an immediate answer, we can only interrupt.
                entry._must_interrupt = True
            else:
                # then strategy is poll, so eventually it will be polled and completed.
                # so once complete, we need to make sure that this entry is finished.
                # Hence, we register it on the reconnection board - entry listens for completion.
                self.reconnection_board.setdefault(cache_key, []).append(entry)
            return True
