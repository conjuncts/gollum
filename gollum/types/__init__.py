from gollum.types.chat_completions import ChatCompletionRequest, ChatCompletionResponse


class GollumRequest:
    def __init__(
        self,
        request: ChatCompletionRequest,
        extras: dict,
        metadata: dict,
        provider_type: str,
    ):
        """

        :param request: The OpenAI ChatCompletions request to be processed
        :param extras: Provider-specific data, often
        :param metadata: Metadata to attach alongside the request
        """
        self.request = request
        self.extras = extras
        self.metadata = metadata
        self.provider_type = provider_type

class GollumResponse:
    def __init__(
        self,
        response: ChatCompletionResponse,
        extras: dict,
        metadata: dict,
    ):
        self.response = response
        self.extras = extras
        self.metadata = metadata
