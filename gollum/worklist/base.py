import asyncio
from typing import TYPE_CHECKING, Literal

from gollum.types import GollumInterrupt, GollumRequest, GollumResponse

if TYPE_CHECKING:
    from gollum.worklist.worklist import Worklist


class WorklistEntry:
    """
    An entry waiting to be done.
    """
    def __init__(self, request: GollumRequest, worklist: "Worklist"):
        self.request = request
        # self.payload = None
        self.worklist: "Worklist" = worklist
        self.status: Literal["starting", "in_progress", "done"] = "starting"
        # self.permacache_key = None
        # self.permacache_likely_partition = None
        # self._lock = threading.Lock()
        # self.done = threading.Event()
        self._future: asyncio.Future[GollumResponse] = worklist.get_event_loop().create_future()
        self._must_interrupt = False

    # def start(self):
    #     self.status = "in_progress"

    def finish(self, payload: GollumResponse):
        self.status = "done"

        # worker pool must be designed so that each entry is assigned to exactly 1 worker
        self._future.get_loop().call_soon_threadsafe(
            self._future.set_result, payload
        )

    async def wait(self) -> GollumResponse:
        if self._must_interrupt:
            raise GollumInterrupt()
        return await self._future

    def __await__(self):
        return self.wait().__await__()