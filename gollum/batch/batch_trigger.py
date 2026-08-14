import asyncio
import atexit
from typing import Optional

from gollum.worklist.base import WorklistEntry
from gollum.worklist.batch_worklist import BatchWorklist


import abc


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
            await self._worklist.flush()

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