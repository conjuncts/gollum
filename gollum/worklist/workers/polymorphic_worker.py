from gollum.provider.provider_registry import ProviderRegistry
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker


class AsyncPolymorphicWorker(Worker):
    def __init__(self, provider_registry: ProviderRegistry):
        self.registry = provider_registry

    async def process(self, worklist_entry: WorklistEntry) -> None:
        for provider in self.providers:
            if provider.supports(worklist_entry):
                await provider.process(worklist_entry)
                return
        raise ValueError("No provider supports this worklist entry")

    def supports(self, worklist_entry: WorklistEntry) -> bool:
        return any(provider.supports(worklist_entry) for provider in self.providers)
