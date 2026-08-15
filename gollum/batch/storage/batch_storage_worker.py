from gollum.batch.batch_handler import BatchHandler
from gollum.worklist.base import WorklistEntry
from gollum.worklist.workers.permacache_worker import PermacacheWorker


class BatchPermacacheWorker(PermacacheWorker):
    """
    BatchWorker serves as BOTH a permacache worker and batch handler.
    """
    def __init__(self, handler: BatchHandler, permacache_worker: PermacacheWorker):
        self.handler = handler
        self.permacache_worker = permacache_worker

    async def process(self, entry: WorklistEntry) -> bool:
        """
        Processes a worklist entry by sending it to the batch handler.
        :return: True = handled, False = requires downstream processing.
        """

        # 1. reconnect
        if await self.handler.reconnect(entry):
            return True

        # 2. check permacache
        if await self.permacache_worker.process(entry):
            return True
        
        # 3. unsuccessful
        return False