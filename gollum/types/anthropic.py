"""
TypedDicts that mirror the Anthropic Messages API request shape.

Reference: https://docs.anthropic.com/en/api/messages

These are structural mirrors of the Anthropic *wire format*, kept in the same
spirit as `gollum.types.chat_completions` (which mirrors the OpenAI format).
They intentionally don't import the anthropic SDK, so these types work even
when the SDK isn't installed.
"""

from typing import List, Literal, Required, TypeAlias, TypedDict, Union


# ---------- cache control ----------

class AnthropicCacheControl(TypedDict):
    type: Literal["ephemeral"]


# ---------- content blocks ----------

class AnthropicTextBlock(TypedDict, total=False):
    type: Literal["text"]
    text: str
    cache_control: AnthropicCacheControl


class AnthropicImageSource(TypedDict, total=False):
    type: Literal["base64", "url"]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str  # base64-encoded bytes; required when type == "base64"
    url: str  # public URL; required when type == "url"


class AnthropicImageBlock(TypedDict, total=False):
    type: Literal["image"]
    source: AnthropicImageSource
    cache_control: AnthropicCacheControl


class AnthropicDocumentSource(TypedDict, total=False):
    type: Literal["base64", "url", "content"]
    media_type: str
    data: str  # base64-encoded bytes; required when type == "base64"
    url: str  # required when type == "url"
    content: str  # required when type == "content"


class AnthropicDocumentBlock(TypedDict, total=False):
    type: Literal["document"]
    source: AnthropicDocumentSource
    title: str
    context: str
    cache_control: AnthropicCacheControl


class AnthropicToolUseBlock(TypedDict, total=False):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict


class AnthropicToolResultBlock(TypedDict, total=False):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List["AnthropicContentBlockParam"]]
    is_error: bool
    cache_control: AnthropicCacheControl


class AnthropicThinkingBlock(TypedDict, total=False):
    type: Literal["thinking"]
    thinking: str
    signature: str


class AnthropicRedactedThinkingBlock(TypedDict, total=False):
    type: Literal["redacted_thinking"]
    data: str


AnthropicContentBlockParam: TypeAlias = Union[
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicDocumentBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicThinkingBlock,
    AnthropicRedactedThinkingBlock,
]


# ---------- messages ----------

class AnthropicUserMessage(TypedDict, total=False):
    role: Literal["user"]
    content: Union[str, List[AnthropicContentBlockParam]]


class AnthropicAssistantMessage(TypedDict, total=False):
    role: Literal["assistant"]
    content: Union[str, List[AnthropicContentBlockParam]]


AnthropicMessageParam: TypeAlias = Union[AnthropicUserMessage, AnthropicAssistantMessage]


# The `system` prompt is a plain string or a list of text/image/document blocks.
AnthropicSystemParam: TypeAlias = Union[
    str,
    List[Union[AnthropicTextBlock, AnthropicImageBlock, AnthropicDocumentBlock]],
]


# ---------- tools ----------

class AnthropicTool(TypedDict, total=False):
    name: str
    description: str
    input_schema: dict  # arbitrary JSON-schema
    type: Literal["custom"]
    cache_control: AnthropicCacheControl


class AnthropicToolChoiceAuto(TypedDict, total=False):
    type: Literal["auto"]
    disable_parallel_tool_use: bool


class AnthropicToolChoiceAny(TypedDict, total=False):
    type: Literal["any"]
    disable_parallel_tool_use: bool


class AnthropicToolChoiceTool(TypedDict, total=False):
    type: Literal["tool"]
    name: str
    disable_parallel_tool_use: bool


class AnthropicToolChoiceNone(TypedDict, total=False):
    type: Literal["none"]


AnthropicToolChoiceParam: TypeAlias = Union[
    AnthropicToolChoiceAuto,
    AnthropicToolChoiceAny,
    AnthropicToolChoiceTool,
    AnthropicToolChoiceNone,
]


# ---------- thinking ----------

class AnthropicThinkingEnabled(TypedDict, total=False):
    type: Literal["enabled"]
    budget_tokens: int


class AnthropicThinkingAdaptive(TypedDict, total=False):
    type: Literal["adaptive"]
    budget_tokens: int


class AnthropicThinkingDisabled(TypedDict, total=False):
    type: Literal["disabled"]


AnthropicThinkingConfig: TypeAlias = Union[
    AnthropicThinkingEnabled,
    AnthropicThinkingAdaptive,
    AnthropicThinkingDisabled,
]


# ---------- metadata ----------

class AnthropicMetadata(TypedDict, total=False):
    user_id: str


# ---------- structured outputs ----------

class AnthropicOutputFormat(TypedDict, total=False):
    type: Required[Literal["json_schema"]]
    schema: Required[dict]


class AnthropicOutputConfig(TypedDict, total=False):
    format: Required[AnthropicOutputFormat]


# ---------- request ----------

class AnthropicRequest(TypedDict, total=False):
    """Anthropic Messages API request body.

    See https://docs.anthropic.com/en/api/messages
    `model`, `max_tokens` and `messages` are required by the API.
    """

    model: Required[str]
    max_tokens: Required[int]
    messages: Required[List[AnthropicMessageParam]]
    system: AnthropicSystemParam
    temperature: float
    top_p: float
    top_k: int
    stop_sequences: List[str]
    stream: bool
    metadata: AnthropicMetadata
    tools: List[AnthropicTool]
    tool_choice: AnthropicToolChoiceParam
    thinking: AnthropicThinkingConfig
    service_tier: Literal["standard", "priority"]
    cache_control: AnthropicCacheControl
    parallel_tool_calls: bool
    output_config: AnthropicOutputConfig
