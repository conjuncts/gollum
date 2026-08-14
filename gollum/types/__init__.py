from typing import Optional

from gollum.types.chat_completions import ChatCompletionRequest, ChatCompletionResponse


class GollumRequest:
    def __init__(
        self,
        chat_completion: ChatCompletionRequest,
        extras: dict,
        metadata: dict,
        provider_name: str
    ):
        """

        :param completion: The OpenAI ChatCompletions request to be processed
        :param extras: Provider-specific data, often
        :param metadata: Metadata to attach alongside the request
        :param provider_name: The name of the provider (e.g. "openai", "anthropic", "gemini")
        """
        self.chat_completion = chat_completion
        self.extras = extras
        self.metadata = metadata
        self.provider_name = provider_name

class GollumResponse:
    def __init__(
        self,
        response: ChatCompletionResponse,
        extras: dict,
        metadata: dict,
        original: Optional[str] = None,
    ):
        self.chat_completion = response
        self.extras = extras
        self.metadata = metadata
        self.original = original
        """Whatever the provider returned as original response (BaseModel.model_dump_json()) - helps make sure that """

class GollumInterrupt(Exception):
    """
    Raised when a request is interrupted (e.g. by a timeout or cancellation)
    """
    pass
