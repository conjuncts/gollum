

from typing import TYPE_CHECKING, List, Literal, NotRequired, Optional, TypeAlias, TypedDict

if TYPE_CHECKING:
    # basic messages
    from openai.types.chat import ChatCompletionMessageParam  # noqa: F401
    from openai.types.chat import ChatCompletionAssistantMessageParam

    # basic request params
    from openai.types.chat.completion_create_params import CompletionCreateParamsBase  # noqa: F401

    # basic response - note: only BaseModel is available
    from openai.types.chat.chat_completion import ChatCompletion  # noqa: F401
else:
    ChatCompletionMessageParam = dict
    ChatCompletionAssistantMessageParam = dict
    CompletionCreateParamsBase = dict
    ChatCompletion = dict


ChatCompletionMessage: TypeAlias = "ChatCompletionMessageParam"
ChatCompletionRequest: TypeAlias = "CompletionCreateParamsBase"
ChatCompletionResponse: TypeAlias = "ChatCompletion"


class AnthropicThinkingParam(TypedDict, total=False):
    type: Literal["enabled", "adaptive"]
    budget_tokens: int



class OpenAIWebSearchUserLocationApproximate(TypedDict):
    city: str
    country: str
    region: str
    timezone: str


class OpenAIWebSearchUserLocation(TypedDict):
    approximate: OpenAIWebSearchUserLocationApproximate
    type: Literal["approximate"]


class OpenAIWebSearchOptions(TypedDict, total=False):
    search_context_size: Optional[Literal["low", "medium", "high"]]
    user_location: Optional[OpenAIWebSearchUserLocation]



# ---------- Response ----------

class TopLogprob(TypedDict):
    token: str
    logprob: float
    bytes: Optional[List[int]]


class LogprobContent(TypedDict):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: List[TopLogprob]


class Logprobs(TypedDict):
    content: Optional[List[LogprobContent]]
    refusal: Optional[List[LogprobContent]]


class Choice(TypedDict):
    index: int
    message: "ChatCompletionAssistantMessageParam"
    logprobs: NotRequired[Optional[Logprobs]]
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ]


class CompletionTokensDetails(TypedDict):
    reasoning_tokens: NotRequired[int]
    accepted_prediction_tokens: NotRequired[int]
    rejected_prediction_tokens: NotRequired[int]


class PromptTokensDetails(TypedDict):
    cached_tokens: NotRequired[int]
    audio_tokens: NotRequired[int]


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: NotRequired[CompletionTokensDetails]
    prompt_tokens_details: NotRequired[PromptTokensDetails]


class ChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: NotRequired[Usage]
    system_fingerprint: NotRequired[str]
    service_tier: NotRequired[Optional[str]]
