from __future__ import annotations

import abc
import asyncio
import atexit
from typing import List, Optional, Tuple

from gollum.batch.job import BatchJob
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker
from gollum.worklist.worklist import Worklist


class BatchTrigger(abc.ABC):
    """
    Strategy object that decides *when* accumulated entries get flushed
    into a batch and handed to workers.

    A BatchWorklist has no opinion on timing at all -- it just:
      1. calls `on_enroll(entry)` synchronously every time something new
         is enrolled, and
      2. exposes `await worklist.flush()` as the one always-available
         manual escape hatch.

    Everything else -- whether that means "accumulate everything and
    flush once at interpreter exit" or "flush every N entries" or "poll
    every couple seconds regardless of volume" or some OR-combination of
    the above -- is entirely up to the trigger implementation. To add a
    new policy, subclass this and override `on_enroll` and/or
    `start`/`stop`; whenever the policy decides it's time, call
    `self.trigger_flush()` (schedules an awaited, tracked flush) or
    `await self.flush()` if you're already in an async context.
    """

    def __init__(self):
        self._worklist: Optional["BatchWorklist"] = None

    def attach(self, worklist: "BatchWorklist") -> None:
        """Called once when the trigger is bound to a worklist."""
        self._worklist = worklist
        self.start()

    def start(self) -> None:
        """
        Called once, synchronously, on attach (i.e. possibly before any
        event loop is running). Override to register atexit hooks, etc.
        If your trigger needs a *running* event loop (e.g. to spawn a
        polling task), don't do it here -- do it lazily from the first
        `on_enroll` call instead, since that's guaranteed to happen from
        inside a running loop. Default: no-op.
        """
        pass

    def stop(self) -> None:
        """
        Called on worklist shutdown. Override to cancel background tasks,
        unregister hooks, etc. Default: no-op.
        """
        pass

    def on_enroll(self, entry: WorklistEntry) -> None:
        """
        Called synchronously every time a new entry is enrolled. This
        runs in a sync context (inside `Worklist.enroll`), so if you want
        to cause a flush here, schedule it -- don't try to await directly.
        Use `self.trigger_flush()` for that. Default: no-op (purely
        manual triggering).
        """
        pass

    async def flush(self) -> None:
        """Convenience passthrough to the attached worklist's flush()."""
        if self._worklist is not None:
            await self._worklist.flush()

    def trigger_flush(self) -> None:
        """
        Schedule a flush as a tracked background task. Prefer this (over
        raw `asyncio.create_task(self.flush())`) from sync hooks like
        `on_enroll`, since it lets the worklist's `join()`/`shutdown()`
        know to wait for it.
        """
        if self._worklist is not None:
            self._worklist._schedule_flush()


class ManualTrigger(BatchTrigger):
    """
    Never auto-flushes. Entries accumulate indefinitely until someone
    explicitly calls `await worklist.flush()` (or `flush_all()`). This is
    the default trigger -- the safest possible one, since it never does
    anything surprising in the background.
    """
    pass


class SizeTrigger(BatchTrigger):
    """
    Flushes as soon as `batch_size` entries are queued up. Good baseline
    for "send in chunks of N" behavior.
    """

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def on_enroll(self, entry: WorklistEntry) -> None:
        if self._worklist is not None and len(self._worklist.entries) >= self.batch_size:
            self.trigger_flush()


class IntervalTrigger(BatchTrigger):
    """
    Polls on a fixed interval and flushes whatever has accumulated since
    the last poll (flush() is a cheap no-op if nothing is queued). This
    is the "rapid-fire mini-batch" strategy: pick a small interval and
    workers stay continuously fed without any per-entry logic at all.

    The polling loop is started lazily on the first `on_enroll` call
    rather than in `start()`, because `start()` may run before any event
    loop exists (e.g. if the worklist is constructed at module import
    time), whereas `on_enroll` is only ever called from inside
    `Worklist.enroll`, which is already async.
    """

    def __init__(self, interval_seconds: float):
        super().__init__()
        self.interval_seconds = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    def on_enroll(self, entry: WorklistEntry) -> None:
        if self._task is None and not self._stopped:
            self._task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self.interval_seconds)
            await self.flush()

    def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()


class ExitTrigger(BatchTrigger):
    """
    Sends exactly one (possibly huge) batch at process exit, and nothing
    before that -- entries just pile up in `worklist.entries` for the
    life of the program.

    Caveat: atexit handlers run synchronously, and by that point the
    event loop entries were enrolled under may be in a weird state or
    already closed. We attempt `asyncio.run` on a fresh loop as a
    best-effort final flush; if that's not viable in your deployment
    (e.g. a framework that owns the loop lifecycle), don't rely on this
    trigger -- instead call `await worklist.shutdown()` explicitly from
    your application's own graceful-shutdown path.
    """

    def __init__(self):
        super().__init__()
        self._registered = False

    def start(self) -> None:
        if not self._registered:
            atexit.register(self._on_exit)
            self._registered = True

    def _on_exit(self) -> None:
        if self._worklist is None or not self._worklist.entries:
            return
        try:
            asyncio.run(self._worklist.flush_all())
        except RuntimeError:
            # A loop is already running / already closed in a way we
            # can't safely drive from a sync atexit hook. Nothing more
            # we can do here -- see the shutdown() recommendation above.
            pass

    def stop(self) -> None:
        # atexit doesn't support clean unregistration of a bound method
        # closure reliably across versions; the handler is a cheap no-op
        # once entries are drained, so we just leave it registered.
        pass


class CompositeTrigger(BatchTrigger):
    """
    Combine multiple triggers -- whichever fires first wins. e.g.

        CompositeTrigger([SizeTrigger(50), IntervalTrigger(5.0)])

    gives "flush every 50 entries OR every 5 seconds, whichever comes
    first", covering the rapid-fire mini-batch + polling case alongside
    a volume-based cap.
    """

    def __init__(self, triggers: list[BatchTrigger]):
        super().__init__()
        self.triggers = triggers

    def attach(self, worklist: "BatchWorklist") -> None:
        self._worklist = worklist
        for t in self.triggers:
            t.attach(worklist)

    def on_enroll(self, entry: WorklistEntry) -> None:
        for t in self.triggers:
            t.on_enroll(entry)

    def stop(self) -> None:
        for t in self.triggers:
            t.stop()


class BatchWorklist(Worklist):
    """
    Worklist that lets entries accumulate and submits them to workers in
    batches (via `Worker.process_batch`) rather than processing each
    entry as soon as it's enrolled.

    *When* a batch actually gets sent is deliberately not this class's
    concern -- that's entirely delegated to a `BatchTrigger` (see above).
    This worklist just needs to:
      - buffer entries (inherited `self.entries`),
      - notify the trigger on every enrollment,
      - provide `flush()` as a single, always-safe "send what's queued
        right now" primitive that triggers (or callers) can invoke
        whenever they decide it's time.

    Examples of what a trigger can do with that:
      - `BatchWorklist(ExitTrigger())` -- accumulate for the whole
        program lifetime, send one giant batch on exit.
      - `BatchWorklist(IntervalTrigger(2.0))` -- rapid-fire mini-batches
        via polling every 2 seconds, regardless of volume.
      - `BatchWorklist(SizeTrigger(100))` -- flush every 100 entries.
      - `BatchWorklist(CompositeTrigger([SizeTrigger(100), IntervalTrigger(5.0)]))`
        -- whichever condition fires first.
      - `BatchWorklist()` (default `ManualTrigger`) -- fully manual;
        caller decides when to `await worklist.flush()`.

    `max_batch_size` optionally caps how many entries a single `flush()`
    call submits at once (useful if you want the exit-trigger's "huge
    batch" to still be chunked into worker-friendly sizes) -- use
    `flush_all()` to drain everything across multiple such chunks.
    """

    def __init__(self, trigger: BatchTrigger | None = None, max_batch_size: int | None = None):
        super().__init__()
        self.trigger = trigger or ManualTrigger()
        self.max_batch_size = max_batch_size
        self._flush_lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task] = set()
        self.trigger.attach(self)
        self.entries: list[WorklistEntry] = []


    async def kickstart_work(self, entry: WorklistEntry):
        """
        Unlike Eager/Concurrent, this does NOT process entries itself.
        It just tells the trigger something new arrived and lets the
        trigger's policy decide what, if anything, happens next.
        """
        # eagerly use cache worker
        if self.cache_worker is not None:
            if await self.cache_worker.process(entry):
                return

        self.entries.append(entry)
        if self.entries:
            self.trigger.on_enroll(self.entries[-1])

    async def flush(self) -> list[WorklistEntry]:
        """
        Submit whatever is currently queued (up to `max_batch_size`, if
        set) as a single batch. Safe to call concurrently/re-entrantly --
        e.g. a size trigger and a poll trigger firing at nearly the same
        moment won't double-send, since only whoever wins the lock
        actually gets entries to drain.

        Returns the entries that were flushed (empty list if the queue
        was empty).
        """
        async with self._flush_lock:
            if not self.entries:
                return []
            if self.max_batch_size is not None:
                batch = self.entries[: self.max_batch_size]
                self.entries = self.entries[self.max_batch_size :]
            else:
                batch = self.entries
                self.entries = []

        if not batch:
            return []

        await self._submit_batch(batch)
        return batch

    async def flush_all(self) -> None:
        """Keep flushing until the queue is fully drained -- relevant
        when `max_batch_size` caps each individual `flush()` call."""
        while self.entries:
            await self.flush()

    async def _submit_batch(self, batch: list[WorklistEntry]) -> None:
        if not self.workers:
            raise ValueError("No workers available to process entries.")

        remaining = batch
        if not remaining:
            return

        # Live processing: split the batch across whichever workers can
        # handle each entry (mirrors EagerWorklist's "first capable
        # worker wins" fallback, generalized so a single flush can be
        # serviced by more than one worker if the batch is heterogeneous).
        jobs: List[Tuple[Worker, BatchJob]] = []
        for worker in self.workers:
            if not remaining:
                break
            capable = [e for e in remaining if await worker.can_process(e)]
            if not capable:
                continue
            job = await worker.send_batch(capable)
            jobs.append((worker, job))
            capable_set = set(id(e) for e in capable)
            remaining = [e for e in remaining if id(e) not in capable_set]

        # await all jobs, swallowing exceptions
        await asyncio.gather(*(worker.await_batch(job) for worker, job in jobs), return_exceptions=True)

        if remaining:
            raise ValueError("No worker could process these entries:", remaining)

        if self.cache_worker is not None:
            for entry in batch:
                if entry not in remaining:
                    await self.cache_worker.record(entry)

    def _schedule_flush(self) -> asyncio.Task:
        """Schedule flush() as a tracked background task -- used by
        triggers so join()/shutdown() know what to wait on."""
        task = asyncio.create_task(self.flush())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def join(self) -> None:
        """Wait for all currently in-flight (trigger-scheduled) flushes
        to finish. Does not itself trigger a flush of anything still
        sitting unflushed in the queue -- see `shutdown()` for that."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks)

    async def shutdown(self) -> None:
        """Stop the trigger (cancel timers/polling) and drain whatever
        is left in the queue. Call this on graceful shutdown paths."""
        self.trigger.stop()
        await self.join()
        await self.flush_all()
