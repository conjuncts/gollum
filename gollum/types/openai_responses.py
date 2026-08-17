"""
TypedDicts that mirror the OpenAI Responses API request/response shape.

Reference: https://platform.openai.com/docs/api-reference/responses

These are structural mirrors of the Responses *wire format*, kept in the same
spirit as `gollum.types.anthropic` (which mirrors the Anthropic format). They
intentionally don't import the openai SDK, so these types work even when the
SDK isn't installed.
"""

from typing import List, Literal, NotRequired, Optional, Required, TypeAlias, TypedDict, Union


# ---------- input content ----------

class ResponsesInputText(TypedDict, total=False):
    type: Literal["input_text"]
    text: str


class ResponsesInputImage(TypedDict, total=False):
    type: Literal["input_image"]
    image_url: Optional[str]
    detail: NotRequired[Literal["auto", "low", "high"]]


ResponsesInputContentPart: TypeAlias = Union[ResponsesInputText, ResponsesInputImage]


class ResponsesOutputText(TypedDict, total=False):
    type: Literal["output_text"]
    text: str
    annotations: NotRequired[list]


# ---------- input items ----------

class ResponsesInputMessage(TypedDict, total=False):
    type: Literal["message"]
    role: Literal["user", "assistant", "system", "developer"]
    content: Union[str, List[ResponsesInputContentPart]]


class ResponsesFunctionCallItem(TypedDict, total=False):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str
    id: NotRequired[str]


class ResponsesFunctionCallOutputItem(TypedDict, total=False):
    type: Literal["function_call_output"]
    call_id: str
    output: str


ResponsesInputItem: TypeAlias = Union[
    ResponsesInputMessage,
    ResponsesFunctionCallItem,
    ResponsesFunctionCallOutputItem,
]


# ---------- tools ----------

class ResponsesFunctionTool(TypedDict, total=False):
    type: Literal["function"]
    name: str
    description: Optional[str]
    parameters: dict
    strict: NotRequired[Optional[bool]]


class ResponsesToolChoiceFunction(TypedDict, total=False):
    type: Literal["function"]
    name: str


ResponsesToolChoice: TypeAlias = Union[
    Literal["auto", "none", "required"],
    ResponsesToolChoiceFunction,
]


# ---------- structured outputs ----------

class ResponsesTextFormatJsonSchema(TypedDict, total=False):
    type: Literal["json_schema"]
    name: str
    schema: dict
    strict: NotRequired[Optional[bool]]


class ResponsesTextFormatJsonObject(TypedDict, total=False):
    type: Literal["json_object"]


class ResponsesTextFormatText(TypedDict, total=False):
    type: Literal["text"]


ResponsesTextFormat: TypeAlias = Union[
    ResponsesTextFormatText,
    ResponsesTextFormatJsonObject,
    ResponsesTextFormatJsonSchema,
]


class ResponsesTextConfig(TypedDict, total=False):
    format: ResponsesTextFormat


# ---------- reasoning ----------

class ResponsesReasoningConfig(TypedDict, total=False):
    effort: Literal["minimal", "low", "medium", "high"]
    summary: Optional[Literal["auto", "concise", "detailed"]]


# ---------- request ----------

class ResponsesRequest(TypedDict, total=False):
    """OpenAI Responses API request body.

    See https://platform.openai.com/docs/api-reference/responses/create
    `model` and `input` are required by the API.
    """

    model: Required[str]
    input: Required[Union[str, List[ResponsesInputItem]]]
    instructions: NotRequired[Optional[str]]
    tools: List[ResponsesFunctionTool]
    tool_choice: ResponsesToolChoice
    parallel_tool_calls: bool
    temperature: float
    top_p: float
    max_output_tokens: int
    previous_response_id: Optional[str]
    store: bool
    stream: bool
    metadata: dict
    text: ResponsesTextConfig
    reasoning: ResponsesReasoningConfig
    truncation: Literal["auto", "disabled"]
    user: str
    service_tier: Literal["auto", "default", "flex", "priority"]


# ---------- response ----------

class ResponsesOutputMessage(TypedDict, total=False):
    type: Literal["message"]
    id: str
    role: Literal["assistant"]
    status: NotRequired[str]
    content: List[ResponsesOutputText]


class ResponsesOutputFunctionCall(TypedDict, total=False):
    type: Literal["function_call"]
    id: str
    call_id: str
    name: str
    arguments: str
    status: NotRequired[str]


class ResponsesReasoningSummaryText(TypedDict, total=False):
    type: Literal["summary_text"]
    text: str


class ResponsesOutputReasoning(TypedDict, total=False):
    type: Literal["reasoning"]
    id: str
    summary: List[ResponsesReasoningSummaryText]


ResponsesOutputItem: TypeAlias = Union[
    ResponsesOutputMessage,
    ResponsesOutputFunctionCall,
    ResponsesOutputReasoning,
    dict,
]


class ResponsesInputTokensDetails(TypedDict, total=False):
    cached_tokens: int


class ResponsesOutputTokensDetails(TypedDict, total=False):
    reasoning_tokens: int


class ResponsesUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: NotRequired[ResponsesInputTokensDetails]
    output_tokens_details: NotRequired[ResponsesOutputTokensDetails]


class ResponsesIncompleteDetails(TypedDict, total=False):
    reason: str


class ResponsesError(TypedDict, total=False):
    code: str
    message: str


class ResponsesResponse(TypedDict, total=False):
    id: str
    object: Literal["response"]
    created_at: float
    status: Literal["completed", "failed", "in_progress", "incomplete", "cancelled", "queued"]
    model: str
    output: List[ResponsesOutputItem]
    usage: NotRequired[ResponsesUsage]
    incomplete_details: NotRequired[Optional[ResponsesIncompleteDetails]]
    error: NotRequired[Optional[ResponsesError]]
    instructions: NotRequired[Optional[str]]
    metadata: NotRequired[dict]
    parallel_tool_calls: NotRequired[bool]
    previous_response_id: NotRequired[Optional[str]]
    temperature: NotRequired[Optional[float]]
    text: NotRequired[ResponsesTextConfig]
    tool_choice: NotRequired[ResponsesToolChoice]
    tools: NotRequired[List[ResponsesFunctionTool]]
    top_p: NotRequired[Optional[float]]
    truncation: NotRequired[Optional[str]]
    user: NotRequired[Optional[str]]
