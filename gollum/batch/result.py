from typing import List
from typing_extensions import Literal

from gollum.types import GollumResponse


class BatchResult:
    def __init__(
        self,
        status: Literal["pending", "completed", "error"],
        cache_keys: list[str],
        results: List[GollumResponse],
        cache_keys_errors: list[str] = [],
        errors: List[dict] = [],
    ):
        self.status: Literal["pending", "completed", "error"] = status
        self.cache_keys = cache_keys
        self.results = results
        self.cache_keys_errors = cache_keys_errors
        self.errors = errors

    def __bool__(self):
        return self.status == "completed" and len(self.results) > 0
