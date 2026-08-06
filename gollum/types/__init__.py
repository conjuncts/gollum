from gollum.types.chat_completions import ChatCompletionRequest, ChatCompletionResponse


class GollumRequest:
    def __init__(
        self,
        chat_completion: ChatCompletionRequest,
        extras: dict,
        metadata: dict,
        provider_name: str,
    ):
        """

        :param completion: The OpenAI ChatCompletions request to be processed
        :param extras: Provider-specific data, often
        :param metadata: Metadata to attach alongside the request
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
    ):
        self.response = response
        self.extras = extras
        self.metadata = metadata
