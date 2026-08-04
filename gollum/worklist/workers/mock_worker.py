from gollum.convert.output.primitive import primitive_to_completion
from gollum.types import GollumResponse
from gollum.worklist.worker import Worker


class MockWorker(Worker):
    """
    Worker but it just parrots a constant value
    """

    def __init__(self, parroted_value: str):
        self.parroted_value = parroted_value

    def process(self, worklist_entry):
        worklist_entry.finish(GollumResponse(primitive_to_completion(self.parroted_value), {}, {}))
