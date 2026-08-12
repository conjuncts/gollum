from typing import List
from typing_extensions import Literal

from gollum.types import GollumResponse


class BatchResult:
    def __init__(self, status: Literal["pending", "completed", "error"], results: List[GollumResponse], errors: List[dict]):
        self.status = status
        self.results = results
        self.errors = errors

    def __bool__(self):
        return self.status == "completed" and len(self.results) > 0
