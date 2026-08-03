from typing import Literal, Union, Optional, List, Dict, Any
from typing_extensions import TypedDict, NotRequired


# ---------- Messages ----------

class FunctionCall(TypedDict):
    name: str
    arguments: str  # JSON-encoded string


class ToolCallFunction(TypedDict):
    name: str
    arguments: str  # JSON-encoded string


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ImageContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Union[TextContentPart, ImageContentPart]


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str
    name: NotRequired[str]


class UserMessage(TypedDict):
    role: Literal["user"]
    content: Union[str, List[ContentPart]]
    name: NotRequired[str]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: NotRequired[Optional[str]]
    name: NotRequired[str]
    function_call: NotRequired[FunctionCall]      # deprecated, legacy
    tool_calls: NotRequired[List[ToolCall]]
    refusal: NotRequired[Optional[str]]


class ToolMessage(TypedDict):
    role: Literal["tool"]
    content: str
    tool_call_id: str


class FunctionMessage(TypedDict):  # deprecated, legacy
    role: Literal["function"]
    content: str
    name: str


Message = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    FunctionMessage,
]


# ---------- Tools / Functions ----------

class FunctionDefinition(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]  # JSON Schema
    strict: NotRequired[Optional[bool]]


class Tool(TypedDict):
    type: Literal["function"]
    function: FunctionDefinition


class ToolChoiceFunction(TypedDict):
    name: str


class NamedToolChoice(TypedDict):
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Union[Literal["none", "auto", "required"], NamedToolChoice]


# ---------- Response format ----------

class JSONSchemaSpec(TypedDict):
    name: str
    description: NotRequired[str]
    schema: NotRequired[Dict[str, Any]]
    strict: NotRequired[Optional[bool]]


class ResponseFormatText(TypedDict):
    type: Literal["text"]


class ResponseFormatJSONObject(TypedDict):
    type: Literal["json_object"]


class ResponseFormatJSONSchema(TypedDict):
    type: Literal["json_schema"]
    json_schema: JSONSchemaSpec


ResponseFormat = Union[
    ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema
]


# ---------- Streaming options ----------

class StreamOptions(TypedDict):
    include_usage: NotRequired[bool]


# ---------- Request ----------

class ChatCompletionRequest(TypedDict):
    model: str
    messages: List[Message]
    frequency_penalty: NotRequired[Optional[float]]
    logit_bias: NotRequired[Optional[Dict[str, int]]]
    logprobs: NotRequired[Optional[bool]]
    top_logprobs: NotRequired[Optional[int]]
    max_tokens: NotRequired[Optional[int]]  # deprecated in favor of max_completion_tokens
    max_completion_tokens: NotRequired[Optional[int]]
    n: NotRequired[Optional[int]]
    presence_penalty: NotRequired[Optional[float]]
    response_format: NotRequired[ResponseFormat]
    seed: NotRequired[Optional[int]]
    service_tier: NotRequired[Optional[Literal["auto", "default"]]]
    stop: NotRequired[Optional[Union[str, List[str]]]]
    stream: NotRequired[Optional[bool]]
    stream_options: NotRequired[Optional[StreamOptions]]
    temperature: NotRequired[Optional[float]]
    top_p: NotRequired[Optional[float]]
    tools: NotRequired[List[Tool]]
    tool_choice: NotRequired[ToolChoice]
    parallel_tool_calls: NotRequired[bool]
    user: NotRequired[str]

    # deprecated legacy fields
    functions: NotRequired[List[FunctionDefinition]]
    function_call: NotRequired[Union[Literal["none", "auto"], ToolChoiceFunction]]


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
    message: AssistantMessage
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


# ---------- Streaming chunk ----------

class DeltaToolCallFunction(TypedDict):
    name: NotRequired[str]
    arguments: NotRequired[str]


class DeltaToolCall(TypedDict):
    index: int
    id: NotRequired[str]
    type: NotRequired[Literal["function"]]
    function: NotRequired[DeltaToolCallFunction]


class Delta(TypedDict):
    role: NotRequired[Literal["system", "user", "assistant", "tool"]]
    content: NotRequired[Optional[str]]
    tool_calls: NotRequired[List[DeltaToolCall]]
    refusal: NotRequired[Optional[str]]


class StreamChoice(TypedDict):
    index: int
    delta: Delta
    logprobs: NotRequired[Optional[Logprobs]]
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ]


class ChatCompletionChunk(TypedDict):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[StreamChoice]
    usage: NotRequired[Optional[Usage]]
    system_fingerprint: NotRequired[str]
    service_tier: NotRequired[Optional[str]]


# Polars