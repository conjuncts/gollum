
from gollum.batch.job import BatchJob
from gollum.batch.result import BatchResult
from gollum.worklist.base import WorklistEntry
from gollum.batch.job import ImmediateBatchJob


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

    async def can_process(self, worklist_entry: WorklistEntry) -> bool:
        """
        Asynchronously checks if the provider can process the given WorklistEntry.
        """
        pass

    async def send_batch(self, worklist_entries: list[WorklistEntry]) -> BatchJob:
        """
        Asynchronously accepts a list of WorklistEntry and deposits the true values.
        :param worklist_entries: A list of WorklistEntry objects to be processed.
        :return: A BatchJob object or None if batch processing is not needed.
        """
        for entry in worklist_entries:
            await self.process(entry)
        return ImmediateBatchJob()


    async def check_batch(self, batch_job: BatchJob) -> BatchResult:
        """
        Checks the batch job in its current state: possibly pending.
        :param batch_job: The BatchJob object to fetch.
        :return: A BatchResult object containing the batch job's results.
        """
        # Placeholder for actual implementation
        return BatchResult("pending", [], [])

    async def await_batch(self, batch_job: BatchJob) -> BatchResult:
        """
        Awaits for the batch job to fully complete and returns the results.
        :param batch_job: The BatchJob object to await.
        :return: A BatchResult object containing the batch job's results.
        """
        # Placeholder for actual implementation
        return BatchResult("completed", [], [])