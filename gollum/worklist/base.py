import asyncio
from typing import Literal

from gollum.types import GollumRequest, GollumResponse


class WorklistEntry:
    """
    An entry waiting to be done.
    """
    def __init__(self, request: GollumRequest, worklist: "Worklist"):
        self.request = request
        # self.payload = None
        self.worklist: Worklist = worklist
        self.status: Literal["starting", "in_progress", "done"] = "starting"
        # self.permacache_key = None
        # self.permacache_likely_partition = None
        # self._lock = threading.Lock()
        # self.done = threading.Event()
        self._future: asyncio.Future[GollumResponse] = worklist.get_event_loop().create_future()

    # def start(self):
    #     self.status = "in_progress"

    def finish(self, payload: GollumResponse):
        self.status = "done"

        # worker pool must be designed so that each entry is assigned to exactly 1 worker
        self._future.get_loop().call_soon_threadsafe(
            self._future.set_result, payload
        )

    async def wait(self) -> GollumResponse:
        return await self._future

class Worklist:
    """
    Producer-consumer pattern
    """

    def __init__(self):
        self.entries = []

    def enroll(self, request: GollumRequest) -> WorklistEntry:
        """
        Enroll a request to be processed later
        :param request:
        :return:
        """
        entry = WorklistEntry(request, self)
        self.entries.append(entry)
        return entry

    # A llm provider should pop an entry, process it, attach the payload, and then mark it as done.
    # The worklist can then be set to null.
    # On the other hand, a WorklistEntry should have a hook that allows it to be awaited on demand

    def get_event_loop(self):
        return asyncio.get_running_loop()