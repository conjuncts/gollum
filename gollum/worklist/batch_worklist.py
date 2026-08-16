import asyncio
import atexit
import logging
from typing import TYPE_CHECKING, List, Optional

from gollum.worklist.base import WorklistEntry
from gollum.worklist.worklist import Worklist

if TYPE_CHECKING:
    from gollum.batch.batch_handler import BatchHandler
    from gollum.batch.batch_trigger import BatchTrigger

logger = logging.getLogger(__name__)


class BatchWorklist(Worklist):
    """
    Worklist backed by a BatchHandler instead of live Workers: entries that
    can't be resolved immediately (no existing batch owns their cache_key,
    and permacache misses) accumulate in `self.entries` until a BatchTrigger
    decides it's time to flush them into `handler.submit_entries()`.

    Hardening notes -- this is the piece that has to survive process
    restarts and abrupt exits, since a batch may still be "in flight" at
    the provider long after this process is gone:

      * `start()` MUST be called (and awaited) before enrolling anything.
        It delegates to `handler.start()`, which frees any batch_storage
        rows left over from a previous session and (re)starts the poll
        loop -- see BatchHandler.start()'s docstring for why that ordering
        matters. Constructing a BatchWorklist does not do this implicitly,
        since it requires a running event loop.
      * Every enrolled entry either resolves immediately (cache hit /
        already-owned batch) or lands in `self.entries`, which is exactly
        the durability boundary: nothing here is "lost" on interrupt, it's
        either already recorded in batch_storage (owned by some batch) or
        still sitting in this in-memory list, uncommitted to any batch,
        which is safe to lose (the caller's `await entry` simply never
        resolves and the process is going down anyway).
      * An atexit hook makes a best-effort synchronous flush + poll-stop on
        interpreter exit, mirroring ExitTrigger's caveats: atexit handlers
        run outside any guaranteed-live event loop, so this is a *best
        effort*, not a guarantee. Prefer calling `await worklist.shutdown()`
        explicitly from your application's own graceful-shutdown path.
    """

    def __init__(self, handler: "BatchHandler", trigger: Optional["BatchTrigger"] = None):
        super().__init__()
        self.handler = handler

        if trigger is None:
            from gollum.batch.batch_trigger import ManualTrigger
            trigger = ManualTrigger()
        self.trigger = trigger

        self._pending: List[WorklistEntry] = []
        # Guards the "snapshot-and-clear self._pending" step in flush() so
        # two concurrent flushes (e.g. a SizeTrigger firing and a manual
        # flush() racing) can't both grab (and double-submit) the same
        # entries.
        self._flush_lock = asyncio.Lock()
        self._flush_tasks: set[asyncio.Task] = set()

        self._started = False
        self._shutdown = False

        self.trigger.attach(self)
        atexit.register(self._atexit_shutdown)

    @property
    def entries(self) -> List[WorklistEntry]:
        """Entries enrolled but not yet handed to handler.submit_entries()."""
        return self._pending

    # ---------------------------------------------------------------- #
    # lifecycle
    # ---------------------------------------------------------------- #

    async def start(self):
        """
        Must be called once, from a running event loop, before enrolling
        anything. Idempotent.
        """
        if not self._started:
            await self.handler.start()
            self._started = True

    async def kickstart_work(self, entry: WorklistEntry):
        # Reconnect first: maybe this cache_key is already owned by an
        # in-flight (or just-completed) batch, or already in permacache.
        if await self.handler.reconnect(entry):
            return

        self._pending.append(entry)
        # Sync hook -- may schedule a background flush via trigger_flush().
        self.trigger.on_enroll(entry)

    def _schedule_flush(self) -> None:
        """Called by BatchTrigger.trigger_flush(). Runs flush() as a
        tracked background task so join()/shutdown() know to wait for it."""
        task = asyncio.create_task(self.flush())
        self._flush_tasks.add(task)
        task.add_done_callback(self._flush_tasks.discard)

    async def flush(self) -> None:
        """
        Hands everything currently pending to handler.submit_entries().
        Cheap no-op if nothing is queued -- safe to call speculatively
        (e.g. from a polling trigger).
        """
        async with self._flush_lock:
            if not self._pending:
                return
            batch, self._pending = self._pending, []
        await self.handler.submit_entries(batch)

    async def flush_all(self) -> None:
        """Alias for flush() -- kept for BatchTrigger's ExitTrigger, which
        doesn't distinguish "flush what's queued" from "flush everything"
        for this worklist (there's only ever one queue)."""
        await self.flush()

    async def join(self) -> None:
        """Wait for all currently in-flight (scheduled-but-not-awaited)
        flushes to finish. Does NOT wait for submitted batches to
        complete -- that's what handler polling / reconnect() is for."""
        if self._flush_tasks:
            await asyncio.gather(*self._flush_tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        """
        Graceful shutdown: stop the trigger from scheduling any more
        flushes, flush whatever's left, wait for in-flight flushes, then
        stop the handler's poll loop. Idempotent -- safe to call more than
        once (e.g. once explicitly and once from the atexit fallback).
        """
        if self._shutdown:
            return
        self._shutdown = True

        self.trigger.stop()
        await self.flush()
        await self.join()
        await self.handler.stop_polling()

    def _atexit_shutdown(self) -> None:
        """
        Best-effort fallback for processes that exit without calling
        `await shutdown()` explicitly. Mirrors ExitTrigger's caveats: this
        runs synchronously, possibly with no event loop alive, so it can
        only help when a fresh loop can safely be spun up. If that's not
        viable in your deployment, don't rely on this -- call
        `await worklist.shutdown()` from your own shutdown path instead.
        """
        if self._shutdown:
            return
        try:
            asyncio.run(self.shutdown())
        except RuntimeError:
            # A loop is already running / already closed in a way we can't
            # safely drive from a sync atexit hook. Whatever's still in
            # self._pending is lost; anything already submitted is safe --
            # it's durably recorded in batch_storage and will be picked up
            # by the next session's poll loop after start() frees it.
            logger.warning(
                "BatchWorklist: could not flush %d pending entries at exit "
                "(no usable event loop); call `await worklist.shutdown()` "
                "explicitly for a clean exit.",
                len(self._pending),
            )
