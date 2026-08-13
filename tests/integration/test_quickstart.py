
import pytest

from gollum.client.base import GollumClient
from gollum.client.litellm import GollumRouter
from gollum.provider.provider_registry import ProviderRegistry
from gollum.worklist.concurrent_worklist import ConcurrentWorklist
from gollum.worklist.workers.mock_worker import MockWorker
from gollum.worklist.workers.polymorphic_worker import AsyncPolymorphicWorker



@pytest.fixture
def mock_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register_provider("openai", lambda: MockWorker("Hello from OpenAI!"))
    registry.register_provider("google", lambda: MockWorker("Hello from Google!"))
    
    # Add mock providers to the registry as needed for testing
    return registry

@pytest.fixture
def gollum_client(mock_provider_registry) -> GollumClient:
    worklist = ConcurrentWorklist()


    worker = AsyncPolymorphicWorker(provider_registry=mock_provider_registry)
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

    assert response.choices[0].message.content == "Hello from OpenAI!"


def test_main(gollum_client):
    # response = completion(
    router = GollumRouter(client=gollum_client)
    response = router.completion(
        model="openai/gpt-5.6-luna",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    assert response.choices[0].message.content == "Hello from OpenAI!"
