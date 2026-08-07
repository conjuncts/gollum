from gollum.permacache.base import Permacache
from gollum.permacache.cache_method import CacheMethod
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker


class PermacacheWorker(Worker):
    def __init__(self, permacache: Permacache, cache_method: CacheMethod):
        self.permacache = permacache
        self.cache_method = cache_method

    async def process(self, worklist_entry: WorklistEntry) -> bool:
        cache_key = self.cache_method.generate_cache_key(worklist_entry.request)
        cached_response = await self.permacache.retrieve(cache_key, likely_partition="")
        if cached_response is not None:
            worklist_entry.finish(cached_response)
            return True
        else:
            return False

    async def record(self, worklist_entry: WorklistEntry):
        cache_key = self.cache_method.generate_cache_key(worklist_entry.request)
        await self.permacache.store(await worklist_entry._future, cache_key, likely_partition="")
