from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

class BatchJob:
    def __init__(self, batch_id: str, provider_name: str):
        self.batch_id = batch_id
        """uuid which uniquely identifies the batch"""
        self.provider_name = provider_name
        """name of the provider which processed the batch"""

    def __repr__(self):
        return f"BatchJob(batch_id={self.batch_id}, provider_name={self.provider_name})"
