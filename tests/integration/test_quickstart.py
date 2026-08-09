
import pytest

from gollum.client.base import GollumClient
from gollum.client.litellm import GollumRouter
from gollum.worklist.concurrent_worklist import ConcurrentWorklist
from gollum.worklist.workers.mock_worker import MockWorker

@pytest.fixture
def gollum_client() -> GollumClient:
    # worklist = EagerWorklist()
    worklist = ConcurrentWorklist()

    # worker = MockWorker(parroted_value="Hello, World!")

    # from openai import AsyncOpenAI
    # worker = AsyncOpenAIWorker(client=AsyncOpenAI())
    # if storage:
    #     # permacache = PolarsPermacache(FileManager(".gollum"), flush_threshold=10)
    #     permacache = DuckDBPermacache(FileManager(".gollum"), flush_threshold=10)
    #     cache_method = CacheMethod()
    #     cacher = PermacacheWorker(permacache, cache_method)
    #     worklist.enroll_cache_worker(cacher)

    # worker = AsyncPolymorphicWorker(provider_registry=get_default_registry())
    worker = MockWorker(parroted_value="Hello, World!")
    worklist.enroll_worker(worker)
    return GollumClient(worklist)

@pytest.mark.asyncio
async def test_amain(gollum_client: GollumClient):
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

    assert response.choices[0].message.content == "Hello, World!"


def test_main(gollum_client):
    # response = completion(
    router = GollumRouter(client=gollum_client)
    response = router.completion(
        model="openai/gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    assert response.choices[0].message.content == "Hello, World!"
