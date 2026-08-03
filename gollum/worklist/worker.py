from gollum.worklist.base import WorklistEntry


class Worker:
    """
    Consumes entries from the worklist and uses LLMs to answer it.
    """

    def process(self, worklist_entry: WorklistEntry):
        """
        Consumes a WorklistEntry and deposits the true value.
        """
        pass
