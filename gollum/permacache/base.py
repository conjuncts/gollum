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

    @abstractmethod
    async def store_custom_id(self, custom_id: str, cache_key: str, likely_partition: str):
        """
        Link cache_key <=> custom_id, for use in batch processing.
        TODO: consider directly making custom_id the cache_key, as that would mean this
        association would not have to go through disk, at expense of slightly larger
        requests.
        """
        pass
