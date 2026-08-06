"""
A global client

"""


from gollum.client.base import GollumClient



from gollum.provider.provider_registry import get_default_registry
from gollum.worklist.workers.polymorphic_worker import AsyncPolymorphicWorker
from gollum.worklist.worklist import EagerWorklist


_singleton = None
def get_singleton_client() -> GollumClient:
    global _singleton
    if _singleton is None:
        worklist = EagerWorklist()
        # worklist.enroll_worker(MockWorker(parroted_value="Hello, World!"))

        # from openai import AsyncOpenAI
        # worklist.enroll_worker(AsyncOpenAIWorker(client=AsyncOpenAI()))
        worker = AsyncPolymorphicWorker(provider_registry=get_default_registry())
        worklist.enroll_worker(worker)
        _singleton = GollumClient(worklist)
    return _singleton

