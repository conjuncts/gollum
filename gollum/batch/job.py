class BatchJob:
    def __init__(self, batch_id: str, provider_name: str):
        self.batch_id = batch_id
        """uuid which uniquely identifies the batch"""
        self.provider_name = provider_name
        """name of the provider which processed the batch"""

    def __repr__(self):
        return f"BatchJob(batch_id={self.batch_id}, provider_name={self.provider_name})"

    async def wait_for_completion(self):
        """
        Waits for the batch job to complete. This is a placeholder for actual implementation.
        """
        pass


class ImmediateBatchJob(BatchJob):
    """
    Represents a batch job that is completed immediately.
    """

    def __init__(self):
        super().__init__(batch_id="immediate", provider_name=None)

    async def wait_for_completion(self):
        """
        Since this is an immediate batch job, it is already complete.
        """
        pass