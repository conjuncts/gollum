
from gollum.batch.job import BatchJob
from gollum.batch.result import BatchResult
from gollum.worklist.base import WorklistEntry


class Worker:
    """
    Consumes entries from the worklist and uses LLMs to answer it.
    """

    async def process(self, worklist_entry: WorklistEntry) -> bool:
        """
        Asynchronously accepts a WorklistEntry and deposits the true value.
        Returns: True if the entry was processed successfully, False otherwise.
        """
        pass

class BatchWorker:
    """
    Works in batches.
    """

    async def send_batch(self, worklist_entries: list[WorklistEntry]) -> BatchJob:
        """
        Asynchronously accepts a list of WorklistEntry and deposits the true values.
        :param worklist_entries: A list of WorklistEntry objects to be processed.
        :return: A BatchJob object or None if batch processing is not needed.
        """
        pass

    async def check_batch(self, batch_job: BatchJob) -> BatchResult:
        """
        Checks the batch job in its current state: possibly pending.
        :param batch_job: The BatchJob object to fetch.
        :return: A BatchResult object containing the batch job's results.
        """
        pass

    async def await_batch(self, batch_job: BatchJob) -> BatchResult:
        """
        Awaits for the batch job to fully complete and returns the results.
        :param batch_job: The BatchJob object to await.
        :return: A BatchResult object containing the batch job's results.
        """
        pass
