from abc import abstractmethod
from typing import TYPE_CHECKING

from gollum.types import GollumRequest
from gollum.worklist.base import WorklistEntry


import asyncio

from gollum.worklist.worker import Worker

if TYPE_CHECKING:
    from gollum.worklist.workers.permacache_worker import PermacacheWorker


class Worklist:
    """
    Producer-consumer pattern
    """

    def __init__(self):
        self.workers: list[Worker] = []
        self.cache_worker = None

    async def enroll(self, request: GollumRequest) -> WorklistEntry:
        """
        Enroll a request to be processed later
        :param request:
        :return:
        """
        entry = WorklistEntry(request, self)
        await self.kickstart_work(entry)
        return entry

    # A llm provider should pop an entry, process it, attach the payload, and then mark it as done.
    # The worklist can then be set to null.
    # On the other hand, a WorklistEntry should have a hook that allows it to be awaited on demand

    def get_event_loop(self):
        return asyncio.get_running_loop()

    def enroll_worker(self, worker: Worker):
        self.workers.append(worker)

    def enroll_cache_worker(self, worker: "PermacacheWorker"):
        """
        Enroll a permacache worker to handle caching operations.
        """
        self.workers.append(worker)
        self.cache_worker = worker


    @abstractmethod
    async def kickstart_work(self, entry: WorklistEntry):
        """
        Start processing the entries
        """
        pass


class EagerWorklist(Worklist):
    """
    Simple, no-concurrency worklist where immediately processes entries.
    """

    async def kickstart_work(self, entry: WorklistEntry):
        # simply use the first worker
        if not self.workers:
            raise ValueError("No workers available to process entries.")

        # cache hit
        if self.cache_worker is not None:
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
            await self.cache_worker.record(entry)
