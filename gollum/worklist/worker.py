from gollum.worklist.base import WorklistEntry


class Worker:
    """
    Consumes entries from the worklist and uses LLMs to answer it.
    """

    def process(self, worklist_entry: WorklistEntry):
        """
        Accepts a WorklistEntry and deposits the true value.
        """
        pass

    def process_batch(self, worklist_entries: list[WorklistEntry]):
        """
        Accepts a list of WorklistEntry and deposits the true values.
        """
        for entry in worklist_entries:
            self.process(entry)