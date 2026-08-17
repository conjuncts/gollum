import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Coroutine, Optional, TypeVar, Union

from gollum.worklist.worklist import Worklist

if TYPE_CHECKING:
    from gollum.batch.batch_trigger import BatchTrigger
    from gollum.provider.provider_registry import ProviderRegistry
    from gollum.worklist.worker import BatchWorker


T = TypeVar("T")

class GollumClient:
    def __init__(self, worklist):
        self.worklist: Worklist = worklist
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()

    @classmethod
    def create(
        cls,
        cache_location: Optional[Union[Path, str]] = None,
        *,
        max_concurrency: Optional[int] = None,
        provider_registry: Optional["ProviderRegistry"] = None,
    ) -> "GollumClient":
        """
        Out-of-the-box client: ConcurrentWorklist + AsyncPolymorphicWorker
        over the default provider registry, with an optional DuckDB-backed
        permacache.

        This is the one client-construction path everything should build on
        -- GollumRouter's default, the module-level singleton, and anything
        else that just wants a working client without hand-wiring
        worklist/worker/cache plumbing. Reach past it (build a
        Worklist/Worker combo directly, as tests do to swap in
        MockWorker/EagerWorklist) only when you need behavior it doesn't
        cover.
        """
        from gollum.folder.file_manager import FileManager
        from gollum.permacache.cache_method import CacheMethod
        from gollum.permacache.duckdb_permacache import DuckDBPermacache
        from gollum.provider.provider_registry import get_default_registry
        from gollum.worklist.concurrent_worklist import ConcurrentWorklist
        from gollum.worklist.workers.permacache_worker import PermacacheWorker
        from gollum.worklist.workers.polymorphic_worker import AsyncPolymorphicWorker

        worklist = ConcurrentWorklist(max_concurrency=max_concurrency)

        if cache_location is not None:
            permacache = DuckDBPermacache(FileManager(cache_location), flush_threshold=10)
            worklist.enroll_cache_worker(PermacacheWorker(permacache, CacheMethod()))

        worker = AsyncPolymorphicWorker(provider_registry=provider_registry or get_default_registry())
        worklist.enroll_worker(worker)
        return cls(worklist)

    @classmethod
    async def create_batch(
        cls,
        cache_location: Union[Path, str] = ".gollum",
        *,
        batch_worker: Optional["BatchWorker"] = None,
        trigger: Optional["BatchTrigger"] = None,
        polling_frequency: float = 60.0,
        confirm_before_submit: bool = True,
    ) -> "GollumClient":
        """
        Out-of-the-box batch client: OpenAI's Batch API behind a
        DuckDB-backed BatchHandler/BatchStorage (so in-flight batches
        survive a process restart), auto-flushing whatever's queued every
        100 entries or every 30 seconds, whichever comes first.

        Must be awaited from the same event loop that will later drive
        calls through this client -- it starts BatchHandler's poll loop on
        the calling loop (see BatchWorklist.start()), which is why this is
        a coroutine and `create()` above isn't. There's no sync flavor for
        the same reason `GollumRouter(...)`'s plain constructor can't set
        this up for you.

        `batch_worker` defaults to `BatchOpenAIWorker(AsyncOpenAI())`,
        which needs `OPENAI_API_KEY` set and talks to the real Batch API --
        submitting a batch costs money and can take hours to resolve, so
        `confirm_before_submit=True` by default prompts on the console
        before each submission. Pass a different `batch_worker` for another
        provider, and `confirm_before_submit=False` once you trust the
        flow (e.g. in an automated pipeline).
        """
        from gollum.batch.batch_handler import BatchHandler
        from gollum.batch.batch_trigger import CompositeTrigger, IntervalTrigger, SizeTrigger
        from gollum.batch.storage.duckdb_batch_storage import DuckDBBatchStorage
        from gollum.folder.file_manager import FileManager
        from gollum.permacache.cache_method import CacheMethod
        from gollum.permacache.duckdb_permacache import DuckDBPermacache
        from gollum.provider.openai_batch import BatchOpenAIWorker
        from gollum.worklist.batch_worklist import BatchWorklist

        fm = FileManager(cache_location)

        if batch_worker is None:
            from openai import AsyncOpenAI
            batch_worker = BatchOpenAIWorker(AsyncOpenAI())

        if trigger is None:
            trigger = CompositeTrigger([SizeTrigger(100), IntervalTrigger(30.0)])

        handler = BatchHandler(
            batch_storage=DuckDBBatchStorage(fm),
            batch_worker=batch_worker,
            permacache=DuckDBPermacache(fm),
            cache_method=CacheMethod(),
            polling_frequency=polling_frequency,
            confirm_before_submit=confirm_before_submit,
        )

        worklist = BatchWorklist(handler, trigger=trigger)
        await worklist.start()
        return cls(worklist)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily spin up the background thread + loop, once."""
        with self._loop_lock:
            if self._loop is None or not self._loop.is_running():
                ready = threading.Event()

                def _run_loop():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self._loop = loop
                    ready.set()
                    loop.run_forever()

                self._loop_thread = threading.Thread(
                    target=_run_loop, name="gollum-event-loop", daemon=True
                )
                self._loop_thread.start()
                ready.wait()
            return self._loop

    def run_coroutine_sync(self, coro: Coroutine[None, None, T]) -> T:
        """
        Run `coro` on this client's persistent background loop, blocking the
        calling thread until it's done.

        Compared to asyncio.run(): reuses one loop across calls (no per-call
        setup/teardown), and works even if the calling thread already has its
        own running loop, since execution happens on a separate thread.

        A few things worth flagging:

        - Coroutine objects aren't bound to a loop until scheduled, so it's fine to build 
        coro = acompletion(...) on the calling thread and only hand it 
        to the background loop via run_coroutine_threadsafe.
        - Reentrancy risk: if completion() is somehow called from inside the background loop's 
        own thread (e.g., a callback running on it calls back into completion()), future.result() 
        would deadlock waiting on itself. Unlikely given your architecture, but if it's a real 
        risk you can guard with threading.get_ident() != self._loop_thread.ident 
        and raise instead of blocking.
        - Shutdown: since the thread is a daemon it won't block process exit, 
        but if you want deterministic cleanup (e.g. draining in-flight requests, 
        closing shared_session), register atexit.register(get_singleton_client().close) 
        wherever the singleton is created, or expose it for explicit teardown in tests.
        - _ensure_loop()'s check-then-create isn't perfectly atomic against self._loop.is_running()
        racing with close(), but under the lock it's safe enough for a singleton client
        used the way get_singleton_client() implies.
        """
        
        loop = self._ensure_loop()
        fut: asyncio.Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()

    def close(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None
