
import tempfile

import pytest

from gollum.client.base import GollumClient
from gollum.client.litellm import GollumRouter
from gollum.folder.file_manager import FileManager
from gollum.permacache.cache_method import CacheMethod
from gollum.permacache.duckdb_permacache import DuckDBPermacache
from gollum.provider.provider_registry import ProviderRegistry
from gollum.worklist.concurrent_worklist import ConcurrentWorklist
from gollum.worklist.workers.mock_worker import MockWorker
from gollum.worklist.workers.permacache_worker import PermacacheWorker
from gollum.worklist.workers.polymorphic_worker import AsyncPolymorphicWorker
from gollum.worklist.worklist import EagerWorklist



@pytest.fixture
def gollum_client() -> GollumClient:
    worklist = EagerWorklist()

    with tempfile.TemporaryDirectory() as temp_dir:
        permacache = DuckDBPermacache(FileManager(temp_dir), flush_threshold=10)
        cache_method = CacheMethod()
        cacher = PermacacheWorker(permacache, cache_method)
        worklist.enroll_cache_worker(cacher)

        worker = MockWorker(parroted_value=["Hello 1", "Hello 2"])
        worklist.enroll_worker(worker)
        return GollumClient(worklist)



@pytest.mark.asyncio
async def test_cache_hit(gollum_client: GollumClient):
    # response = await acompletion(
    # router = GollumRouter()
    router = GollumRouter(cache_responses=True, client=gollum_client)
    response = await router.acompletion(
        model="openai/gpt-5.6-luna",
        # model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )


    assert response.choices[0].message.content == "Hello 1"
    # The second call should hit the cache
    response = await router.acompletion(
        model="openai/gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    assert response.choices[0].message.content == "Hello 1"
    
