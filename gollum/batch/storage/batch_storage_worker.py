from gollum.batch.storage.batch_storage import BatchStorage
from gollum.worklist.worker import Worker
from gollum.worklist.workers.permacache_worker import PermacacheWorker


class BatchStorageWorker(Worker):
    """
    Tracks batch jobs. Is highly coupled to permacache worker.
    """
    def __init__(self, batch_storage: BatchStorage, permacache_worker: PermacacheWorker):
        self.batch_storage = batch_storage
        self.permacache_worker = permacache_worker
