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


    async def process_batch(self, worklist_entries: list[WorklistEntry]) -> None:
        """
        Asynchronously accepts a list of WorklistEntry and deposits the true values.
        """
        for entry in worklist_entries:
            await self.process(entry)

    async def can_process(self, worklist_entry: WorklistEntry) -> bool:
        """
        Asynchronously checks if the provider can process the given WorklistEntry.
        """
        pass
