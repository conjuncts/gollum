from abc import ABC, abstractmethod
from typing import Optional


from gollum.folder.file_manager import FileManager
from gollum.types import GollumResponse


class Permacache(ABC):
    """
    A permacache is a cache that stores LLM responses to disk.
    """
    def __init__(self, fm: FileManager):
        self.fm = fm

    @abstractmethod
    async def store(self, response: GollumResponse, cache_key: str, likely_partition: str):
        pass

    @abstractmethod
    async def retrieve(self, cache_key: str, likely_partition: str) -> Optional[GollumResponse]:
        pass
