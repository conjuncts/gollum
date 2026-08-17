
from pathlib import Path
import shutil

import pytest

from gollum.client.base import GollumClient
from gollum.client.litellm import GollumRouter
from gollum.folder.file_manager import FileManager
from gollum.permacache.cache_method import CacheMethod
from gollum.permacache.duckdb_permacache import DuckDBPermacache
from gollum.worklist.workers.mock_worker import MockWorker
from gollum.worklist.workers.permacache_worker import PermacacheWorker
from gollum.worklist.worklist import EagerWorklist

@pytest.fixture
def gollum_client(tmp_path) -> GollumClient:
    worklist = EagerWorklist()

    # copy the duckdb file over to where DuckDBPermacache expects it:
    # permacache/duckdb/v1/cache.duckdb
    dest_dir = Path(tmp_path) / "permacache/duckdb/v1"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/data/cache.duckdb", dest_dir / "cache.duckdb")

    permacache = DuckDBPermacache(FileManager(tmp_path), flush_threshold=10)
    # do not flush the cache at exit, because the temp dir will be gone
    permacache._flush_at_exit = lambda: None
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
            {"role": "user", "content": "Foo"}
        ],
    )


    assert response.choices[0].message.content == "Hello 1"
    # The second call should hit the cache
    response = await router.acompletion(
        model="openai/gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "Foo"}
        ],
    )

    assert response.choices[0].message.content == "Hello 1"  # not Hello 2

    # The response is stored in the cache
    response = await router.acompletion(
        model="openai/gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    assert response.choices[0].message.content == "The capital of France is **Paris**."
