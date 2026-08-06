from typing import Optional


from gollum.folder.file_manager import FileManager
from gollum.types import GollumRequest


class Permacache:
    """
    A permacache is a cache that stores LLM responses to disk.
    """
    def __init__(self, fm: FileManager):
        self.fm = fm

    def store(self, request: GollumRequest, cache_key: str, likely_partition: str):
        pass

    def retrieve(self, cache_key: str, likely_partition: str) -> Optional[GollumRequest]:
        pass
