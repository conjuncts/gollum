from typing import Callable, Dict

from gollum.worklist.worker import Worker

class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, Callable[[], Worker]] = {}

    def register_provider(self, provider_name: str, produces_provider: Callable[[], Worker]):
        self._providers[provider_name] = produces_provider

    def get_provider(self, provider_name: str) -> Worker:
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' is not registered.")
        return self._providers[provider_name]()


def _build_openai() -> Worker:
    from gollum.provider.openai import AsyncOpenAIWorker
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    return AsyncOpenAIWorker(client=client)

def _build_anthropic() -> Worker:
    from gollum.provider.anthropic import AsyncAnthropicWorker
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    return AsyncAnthropicWorker(client=client)

_default_registry = None
def get_default_registry():
    global _default_registry
    if _default_registry is None:
        # build default registry
        _default_registry = ProviderRegistry()
        _default_registry.register_provider("openai", _build_openai)
        _default_registry.register_provider("anthropic", _build_anthropic)
        # need to register both google and gemini
    return _default_registry

def register_provider(provider_name: str, produces_provider: Callable[[], Worker]):
    global _default_registry
    _default_registry[provider_name] = produces_provider