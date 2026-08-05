import asyncio
import threading
from typing import Coroutine, Optional, TypeVar

from gollum.worklist.worklist import Worklist


T = TypeVar("T")

class GollumClient:
    def __init__(self, worklist):
        self.worklist: Worklist = worklist
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()

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
