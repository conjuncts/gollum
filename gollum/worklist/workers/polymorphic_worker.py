from gollum.provider.provider_registry import ProviderRegistry
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker


class AsyncPolymorphicWorker(Worker):
    def __init__(self, provider_registry: ProviderRegistry):
        self.registry = provider_registry
        self.providers = []

    async def process(self, worklist_entry: WorklistEntry) -> bool:
        provider = self.registry.get_provider(worklist_entry.request.provider_name)
        if provider is None:
            raise ValueError("No provider supports this worklist entry")
        return await provider.process(worklist_entry)

    def supports(self, worklist_entry: WorklistEntry) -> bool:
        return any(provider.supports(worklist_entry) for provider in self.providers)
