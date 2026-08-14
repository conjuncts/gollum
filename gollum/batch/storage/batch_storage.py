from abc import ABC, abstractmethod
from typing import Optional

from gollum.batch.job import BatchJob


class BatchStorage(ABC):
    """
    A cache for storing batch results
    """

    @abstractmethod
    async def record_batch(self, batch: BatchJob, cache_keys: list[str]):
        """
        Records that these cache_keys are associated with this batch job.
        Signals gollum to avoid reprocessing these cache_keys.
        """
        pass

    @abstractmethod
    async def retrieve_batch(self, cache_key: str, likely_partition: str) -> Optional[BatchJob]:
        """
        Checks if a batch job has been recorded for this cache_key, and returns it if so.
        """
        pass

    @abstractmethod
    async def free_batch(self, batch: BatchJob):
        """
        Frees the cache entries associated with this batch job.
        Should be called when the batch completes (individual cache_keys then possibly transfer to the permacache)
        """
        pass

    @abstractmethod
    async def get_all_batches(self) -> list[BatchJob]:
        """
        Returns all batch jobs currently in the cache.
        """
        pass
    