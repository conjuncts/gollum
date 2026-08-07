from typing import Optional


from gollum.folder.file_manager import FileManager
from gollum.types import GollumResponse


class Permacache:
    """
    A permacache is a cache that stores LLM responses to disk.
    """
    def __init__(self, fm: FileManager):
        self.fm = fm

    async def store(self, response: GollumResponse, cache_key: str, likely_partition: str):
        pass

    async def retrieve(self, cache_key: str, likely_partition: str) -> Optional[GollumResponse]:
        pass
