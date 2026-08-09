import asyncio

from gollum.worklist.base import WorklistEntry
from gollum.worklist.worklist import Worklist


class ConcurrentWorklist(Worklist):
    """
    Worklist that processes entries concurrently using asyncio tasks on the
    *same* event loop as the caller.

    Why not a separate event loop: asyncio concurrency comes from cooperative
    yielding at `await` points, not from OS threads. Scheduling each entry as
    its own Task on the current loop already lets N entries be in flight at
    once. A second loop is only useful if the work is CPU-bound or uses
    blocking I/O -- and in that case the fix belongs inside `Worker.process`
    (e.g. `await loop.run_in_executor(None, blocking_fn)`), not in the
    worklist. Running two asyncio loops in one process adds real complexity
    (run_coroutine_threadsafe, cross-loop futures) for no benefit here.
    """

    def __init__(self, max_concurrency: int | None = None):
        super().__init__()
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        # Guards cache reads/writes so a lookup can't race a concurrent record()
        # for the same entry. Cheap to hold since cache_worker calls are the
        # only shared-state touchpoints across concurrent entries.
        self._cache_lock = asyncio.Lock()
        self._pending_tasks: set[asyncio.Task] = set()

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

    async def _process_entry_inner(self, entry: WorklistEntry):
        # cache hit
        if self.cache_worker is not None:
            async with self._cache_lock:
                if await self.cache_worker.process(entry):
                    return

        # live processing
        for worker in self.workers:
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
