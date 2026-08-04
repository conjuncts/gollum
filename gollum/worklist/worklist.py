
from gollum.types import GollumRequest
from gollum.worklist.base import WorklistEntry


import asyncio

from gollum.worklist.worker import Worker


class Worklist:
    """
    Producer-consumer pattern
    """

    def __init__(self):
        self.entries = []
        self.workers: list[Worker] = []

    def enroll(self, request: GollumRequest) -> WorklistEntry:
        """
        Enroll a request to be processed later
        :param request:
        :return:
        """
        entry = WorklistEntry(request, self)
        self.entries.append(entry)
        self.kickstart_work()
        return entry

    # A llm provider should pop an entry, process it, attach the payload, and then mark it as done.
    # The worklist can then be set to null.
    # On the other hand, a WorklistEntry should have a hook that allows it to be awaited on demand

    def get_event_loop(self):
        return asyncio.get_running_loop()

    def enroll_worker(self, worker: Worker):
        self.workers.append(worker)

    def kickstart_work(self):
        # Start processing the entries
        pass


class EagerWorklist(Worklist):
    """
    Simple, no-concurrency worklist where immediately processes entries.
    """

    def kickstart_work(self):
        # simply use the first worker
        if not self.workers:
            raise ValueError("No workers available to process entries.")
        worker = self.workers[0]
        while self.entries:
            entry = self.entries.pop(0)
            worker.process(entry)
                