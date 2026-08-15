import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from gollum.batch.job import BatchJob
from gollum.batch.result import BatchResult
from gollum.batch.splitting_method import SplittingMethod
from gollum.batch.storage.batch_storage import BatchStorage
from gollum.permacache.base import Permacache
from gollum.permacache.cache_method import CacheMethod
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import BatchWorker

JobStatus = Literal["pending", "finalizing", "complete"]


@dataclass
class JobState:
    """
    Per-BatchJob bookkeeping. Exactly one instance per `BatchJob.batch_id`,
    created lazily the first time either reconnect() or the poll loop sees
    that batch_id, and never removed (see note in BatchHandler docstring).

    `lock` is the *only* synchronization primitive needed, and it's only
    held for two short, non-blocking-I/O moments in `batch_arrival`: the
    pending->finalizing claim, and the finalizing->complete drain. The
    (slow) permacache-store loop and complete_batch() call run in between,
    with the lock released, so reconnect() calls for this job are never
    stalled behind them -- they can register into `waiters` right up until
    the drain snapshot is taken, since "pending" and "finalizing" are
    treated identically by reconnect(). The only thing the lock protects
    is: (a) at most one caller ever wins the pending->finalizing claim, and
    (b) no registration can land after `waiters` has been snapshotted and
    cleared.
    """

    status: JobStatus = "pending"
    # cache_key -> entries waiting on that specific cache_key within this job.
    # Open for registration during BOTH "pending" and "finalizing" -- see
    # reconnect(), which treats them identically. Drained to empty exactly
    # once, atomically with the finalizing -> complete transition.
    waiters: Dict[str, List[WorklistEntry]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class BatchHandler:
    """
    BatchHandler owns three jobs -- submit, arrival, reconnect -- built
    around an explicit per-BatchJob lifecycle:

        (no state) -> pending -> finalizing -> complete

    "no state" is not stored anywhere; it's simply the absence of a record
    in `batch_storage` (and therefore no JobState in memory either). It
    covers both a cache_key that has genuinely never been submitted, and
    one whose batch completed and was freed in a *previous* process
    lifetime (free_completed() only runs at the start of the next
    session -- see `start()`).

    "pending" vs "finalizing" does NOT gate reconnect()'s behavior -- both
    accept new waiter registrations, identically. "finalizing" exists
    solely as a claim flag so batch_arrival() can't be entered twice
    concurrently for the same job (relevant once there's more than one
    path that can discover completion -- e.g. the poll loop and an
    interrupt-triggered BatchJob.wait_for_completion() racing to report
    the same finished batch). The critical section this protects is
    intentionally small: just the claim itself, and later the drain. The
    (potentially slow) permacache-store loop and complete_batch() call run
    *outside* any lock, so reconnect() calls -- for this job's own
    cache_keys or any other job's -- are never stalled waiting on them.

    Design notes / things worth knowing before extending this:

    * JobState objects are never evicted. That's a deliberate choice, not
      an oversight: cardinality here is bounded by the number of *batches*
      (hundreds/thousands), not entries, so the memory cost is small. The
      alternative -- deleting a JobState once its batch completes -- is
      actively dangerous: a reconnect() that raced past
      `retrieve_batch()` just before the batch's storage row is freed
      could find no JobState, create a fresh "pending" one, and register
      itself as a waiter on a job that will never fire arrival again.
      That's a permanent hang, not just a leak. If this ever needs to be
      bounded (e.g. very long-running processes with huge batch counts),
      replace-with-a-permanent-"complete"-tombstone is safe; outright
      deletion is not.

    * `retrieve_batch` is assumed valid for a cache_key from submission
      all the way through "complete" (Option 2 from our discussion,
      matching the BatchStorage ABC with complete_batch/free_completed).
      `free_completed()` is expected to run once, at process start,
      *before* polling begins -- call `start()` rather than
      `_restart_polling()` directly to get that for free.

    * A batch that fails (BatchResult.status == "error") and a batch that
      partially fails (some cache_keys missing from `.results`) are
      handled identically to a normal completion in `batch_arrival`: every
      cache_key that *did* get a result is resolved and stored; every
      waiter whose cache_key did *not* get a result gets `fail()` and
      subsequent entries are passed through
      as if it were never requested.
    """

    def __init__(
        self,
        batch_storage: BatchStorage,
        batch_worker: BatchWorker,
        splitting_method: Optional[SplittingMethod] = None,
        permacache: Optional[Permacache] = None,
        cache_method: Optional[CacheMethod] = None,
        strategy: Literal["interrupt", "poll"] = "poll",
        polling_frequency: float = 60.0,  # poll every 60 seconds
    ):
        self.batch_storage = batch_storage
        self.batch_worker: BatchWorker = batch_worker
        self.splitting_method = (
            splitting_method if splitting_method is not None else SplittingMethod(1000)
        )
        self.strategy: Literal["interrupt", "poll"] = strategy
        self.permacache = permacache
        self.cache_method = cache_method
        self.polling_frequency = polling_frequency

        self._job_states: Dict[str, JobState] = {}
        # Guards *creation* of a JobState only. Never held across an await,
        # so it can never become a cross-job bottleneck the way the old
        # single global lock was.
        self._job_states_lock = asyncio.Lock()

        self._actively_polling = False
        self._polling_task: Optional[asyncio.Task] = None

    # ---------------------------------------------------------------- #
    # lifecycle plumbing
    # ---------------------------------------------------------------- #

    async def start(self):
        """
        Call this once at process start, instead of calling
        _restart_polling() directly. Frees any batch_storage rows left
        over (marked complete) from a previous session before polling
        begins, per the Option-2 BatchStorage contract.
        """
        await self.batch_storage.free_completed()
        await self._restart_polling()

    async def _restart_polling(self):
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self._actively_polling = True
        self._polling_task = asyncio.create_task(self._poll_loop())

    async def stop_polling(self):
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

    async def _get_or_create_job_state(self, batch_id: str) -> JobState:
        async with self._job_states_lock:
            job_state = self._job_states.get(batch_id)
            if job_state is None:
                job_state = JobState()
                self._job_states[batch_id] = job_state
            return job_state

    # ---------------------------------------------------------------- #
    # SUBMIT: (no state) -> pending
    # ---------------------------------------------------------------- #

    async def submit_entries(self, entries: List[WorklistEntry]) -> None:
        if not entries:
            return

        mini_batches = self.splitting_method.split(entries)

        for mini_batch in mini_batches:
            job = await self.batch_worker.send_batch(mini_batch)
            cache_keys = [self.cache_method.generate_cache_key(item.request) for item in mini_batch]
            await self.batch_storage.record_batch(job, cache_keys)

            # Register these entries as this job's waiters *now*, rather
            # than requiring the caller to call reconnect() a second time
            # after submitting. This isn't just a convenience: it's the
            # only thing that makes retried (previously-failed) entries
            # work at all, since batch_arrival calls submit_entries()
            # directly on their behalf, with nobody else in a position to
            # call reconnect() for them afterwards.
            job_state = await self._get_or_create_job_state(job.batch_id)
            async with job_state.lock:
                if job_state.status == "pending":
                    for item, cache_key in zip(mini_batch, cache_keys):
                        if self.strategy == "interrupt":
                            item._must_interrupt = True
                        else:
                            job_state.waiters.setdefault(cache_key, []).append(item)
                    continue

            # Extremely unlikely (job_state existed and already moved past
            # "pending" before we got here -- e.g. a hyper-fast worker plus
            # a poll tick winning a race), but don't strand entries if it
            # happens: fall back to the same resolution path reconnect()
            # uses.
            for item, cache_key in zip(mini_batch, cache_keys):
                await self._resolve_from_permacache(item, cache_key)

    # ---------------------------------------------------------------- #
    # RECONNECT
    # ---------------------------------------------------------------- #

    async def reconnect(self, entry: WorklistEntry) -> bool:
        """
        Reconnects a WorklistEntry to its BatchJob, per the lifecycle:

          no state              -> permacache check (miss means "go submit
                                    a new batch")
          pending / finalizing  -> attach as a listener (both states are
                                    open for registration -- see JobState)
          complete              -> permacache check (guaranteed populated
                                    by now, whether the batch succeeded or
                                    failed -- a failure just means the
                                    check misses, which is the correct
                                    "please resubmit" signal)

        :return: True if the entry is guaranteed to complete without the
            caller submitting it into a new batch (either because it's
            now attached as a listener, or because it was resolved
            immediately from permacache). False means the caller should
            call submit_entries([entry]).
        """
        cache_key = self.cache_method.generate_cache_key(entry.request)
        job = await self.batch_storage.retrieve_batch(cache_key, "")

        if job is None:
            return await self._resolve_from_permacache(entry, cache_key)

        job_state = await self._get_or_create_job_state(job.batch_id)

        async with job_state.lock:
            if job_state.status != "complete":
                # pending or finalizing: both still accept registration.
                if self.strategy == "interrupt":
                    entry._must_interrupt = True
                else:
                    job_state.waiters.setdefault(cache_key, []).append(entry)
                return True

        return await self._resolve_from_permacache(entry, cache_key)

    async def _resolve_from_permacache(self, entry: WorklistEntry, cache_key: str) -> bool:
        if self.permacache is not None:
            cached_result = await self.permacache.retrieve(cache_key, "")
            if cached_result is not None:
                entry.finish(cached_result)
                return True
        return False

    # ---------------------------------------------------------------- #
    # ARRIVAL: pending -> finalizing -> complete
    # ---------------------------------------------------------------- #

    async def batch_arrival(self, job: BatchJob, batch: BatchResult) -> None:
        """
        Handles both outright success and failure/partial-failure the same
        way: whatever cache_keys came back with a result get stored and
        their waiters notified; whatever didn't get swept back into a
        fresh submit_entries() call. `batch.status == "error"` just means
        every waiter ends up in the second bucket.

        Two short lock-guarded moments, with the slow work done in between
        while the lock is released -- see JobState docstring for why this
        is safe:
          1. claim the job (pending -> finalizing), so at most one caller
             ever proceeds past this point for a given batch_id.
          2. drain (finalizing -> complete): snapshot-and-clear `waiters`
             atomically with closing off further registration.
        """
        job_state = await self._get_or_create_job_state(job.batch_id)

        async with job_state.lock:
            if job_state.status != "pending":
                # Already claimed (finalizing) or already done (complete)
                # -- e.g. the poll loop and an interrupt-triggered
                # wait_for_completion() both discovered this batch. Only
                # the first caller through here does any work.
                return
            job_state.status = "finalizing"

        # --- lock released: the slow part. reconnect() calls for this
        # job's cache_keys (or any other job's) are never stalled behind
        # this -- they can still register into job_state.waiters right up
        # until the drain below takes its snapshot. ---
        results_by_key = dict(zip(batch.cache_keys, batch.results))
        for cache_key, item_result in results_by_key.items():
            await self.permacache.store(item_result, cache_key, "")
        await self.batch_storage.complete_batch(job)

        async with job_state.lock:
            waiters_snapshot, job_state.waiters = job_state.waiters, {}
            job_state.status = "complete"

        resolved: List[tuple] = []
        for cache_key, entries in waiters_snapshot.items():
            if cache_key in results_by_key:
                resolved.extend((e, results_by_key[cache_key]) for e in entries)

        for entry, result in resolved:
            entry.finish(result)

        # call .fail() for error_results
        error_results_by_key = dict(zip(batch.cache_keys_errors, batch.errors))
        for cache_key, entries in waiters_snapshot.items():
            if cache_key in error_results_by_key:
                for entry in entries:
                    entry.fail(Exception(f"Batch job failed for cache_key {cache_key}: {error_results_by_key[cache_key]}"))

        # Do NOT resubmit

    # ---------------------------------------------------------------- #
    # polling
    # ---------------------------------------------------------------- #

    async def _poll_point_check(self):
        jobs = await self.batch_storage.get_all_batches()
        for job in jobs:
            job_state = self._job_states.get(job.batch_id)
            if job_state is not None and job_state.status in ("finalizing", "complete"):
                # "complete": already resolved in-memory this session --
                # get_all_batches() will keep returning it until the next
                # process start's free_completed() call, so skip it rather
                # than re-hitting the worker API every tick.
                # "finalizing": another path
                # has already claimed this job and
                # is processing it right now; batch_arrival()'s own claim
                # check would no-op anyway, but there's no reason to also
                # pay for a check_batch() call to get there.
                continue

            check_result = await self.batch_worker.check_batch(job)

            if check_result.status == "pending":
                continue

            # "completed" and "error" both flow through batch_arrival,
            # which resolves whatever succeeded and retries whatever
            # didn't -- see its docstring.
            await self.batch_arrival(job, check_result)