from typing import List, Union

from gollum.convert.output.primitive import primitive_to_completion
from gollum.types import GollumResponse
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker


class MockWorker(Worker):
    """
    Worker but it just parrots a constant value
    """

    def __init__(self, parroted_value: Union[str, List[str]]):
        if isinstance(parroted_value, str):
            self.parroted_value = [parroted_value]
        else:
            self.parroted_value = parroted_value
        self._current_index = 0

    def get_value(self) -> str:
        value = self.parroted_value[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.parroted_value)
        return value

    async def process(self, worklist_entry: WorklistEntry) -> bool:
        worklist_entry.finish(GollumResponse(primitive_to_completion(self.get_value()), {}, {}))
        return True
