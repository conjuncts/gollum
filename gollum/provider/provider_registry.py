from typing import Callable, Dict

from gollum.worklist.worker import Provider

class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, Callable[[], Provider]] = {}

    def register_provider(self, provider_name: str, produces_provider: Callable[[], Provider]):
        self._providers[provider_name] = produces_provider

    def get_provider(self, provider_name: str) -> Provider:
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' is not registered.")
        return self._providers[provider_name]()


def _build_openai() -> Provider:
    from gollum.provider.openai import AsyncOpenAIWorker
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    return AsyncOpenAIWorker(client=client)

_default_registry = None
def get_default_registry():
    global _default_registry
    if _default_registry is None:
        # build default registry
        _default_registry = ProviderRegistry()
        _default_registry.register_provider("openai", _build_openai)
    return _default_registry

def register_provider(provider_name: str, produces_provider: Callable[[], Provider]):
    global _default_registry
    _default_registry[provider_name] = produces_provider