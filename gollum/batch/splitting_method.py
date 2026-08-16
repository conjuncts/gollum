from typing import Dict, List

from gollum.worklist.base import WorklistEntry


class SplittingMethod:
    """
    A method for splitting a list of worklist entries into batches.
    """
    def __init__(self, max_batch_size: int = 1000):
        self.max_batch_size = max_batch_size

    def split(self, entries: List[WorklistEntry]) -> List[List[WorklistEntry]]:
        """
        Split a list of requests into batches according to the strategy.
        By default: splits by model name and up to max_batch_size requests per batch.
        """
        batches_per_model_name: Dict[str, List[WorklistEntry]] = {}
        for entry in entries:
            request = entry.request
            model_name = request.chat_completion["model"]
            if model_name not in batches_per_model_name:
                batches_per_model_name[model_name] = []
            batches_per_model_name[model_name].append(entry)

        batches: List[List[WorklistEntry]] = []
        for model_name, entries_for_model in batches_per_model_name.items():
            for i in range(0, len(entries_for_model), self.max_batch_size):
                batch = entries_for_model[i:i + self.max_batch_size]
                batches.append(batch)
        return batches
