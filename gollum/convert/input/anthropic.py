"""
Adapter: OpenAI ChatCompletionRequest <-> Anthropic Messages request.

Reference (Anthropic): https://docs.anthropic.com/en/api/messages
Reference (OpenAI):    https://platform.openai.com/docs/api-reference/chat/create

`to_anthropic_request` (OpenAI -> Anthropic) drops fields with no Anthropic
equivalent (logprobs, n, seed, ...); `from_anthropic_request` (Anthropic ->
OpenAI) drops fields with no ChatCompletions equivalent (top_k, thinking,
service_tier, container, ...) by returning them in a second `extras` dict
instead, mirroring how `to_anthropic_request` accepts them.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

from gollum.types.anthropic import (
    AnthropicContentBlockParam,
    AnthropicImageSource,
    AnthropicMessageParam,
    AnthropicOutputConfig,
    AnthropicRequest,
    AnthropicSystemParam,
    AnthropicThinkingConfig,
    AnthropicTool,
    AnthropicToolChoiceParam,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
)
from gollum.types.chat_completions import ChatCompletionRequest

_DATA_URI = re.compile(r"^data:(?P<media>[^;,]+);base64,(?P<data>.*)$", re.DOTALL)


def _set_if_present(out: dict, key: str, value, *, allow_null: bool = False) -> None:
    """Only add `key` to `out` when `value` is not None."""
    if value is not None or allow_null:
        out[key] = value


# ---------- content ----------

def _image_url_to_source(image_url: Optional[dict]) -> AnthropicImageSource:
    """OpenAI `image_url` part -> Anthropic image source.

    Handles both plain URLs and `data:` URIs; Anthropic needs the mime type and
    base64 payload split apart for local images.
    """
    url = (image_url or {}).get("url")
    if not url:
        raise ValueError("image_url part is missing 'url'")
    match = _DATA_URI.match(url)
    if match:
        return {
            "type": "base64",
            "media_type": match.group("media"),
            "data": match.group("data"),
        }
    return {"type": "url", "url": url}


def _content_part_to_block(part: dict) -> AnthropicContentBlockParam:
    """OpenAI content part -> Anthropic content block."""
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "text", "text": part.get("text") or ""}
    if ptype == "image_url":
        return {"type": "image", "source": _image_url_to_source(part.get("image_url"))}
    raise NotImplementedError(
        f"OpenAI content part type {ptype!r} has no Anthropic equivalent"
    )


def _content_to_anthropic(
    content,
) -> Optional[Union[str, List[AnthropicContentBlockParam]]]:
    """OpenAI message content (str | list[part]) -> Anthropic content (str | list[block])."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_content_part_to_block(p) for p in content]
    raise TypeError(f"Unsupported message content: {type(content).__name__}")


# ---------- messages ----------

def _tool_calls_to_blocks(tool_calls) -> List[AnthropicToolUseBlock]:
    """OpenAI assistant `tool_calls` -> Anthropic `tool_use` blocks.

    OpenAI encodes arguments as a JSON *string*; Anthropic wants a parsed dict.
    """
    blocks: List[AnthropicToolUseBlock] = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        arguments = fn.get("arguments")
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tool_call {call.get('id')!r}: 'arguments' is not valid JSON: {arguments!r}"
            ) from exc
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": fn.get("name", ""),
                "input": parsed,
            }
        )
    return blocks


def _message_to_anthropic(message: dict) -> AnthropicMessageParam:
    """Convert one OpenAI chat message to the Anthropic format."""
    role = message.get("role")

    if role == "user":
        content = _content_to_anthropic(message.get("content"))
        return {"role": "user", "content": content or ""}

    if role == "assistant":
        content = _content_to_anthropic(message.get("content"))
        blocks: List[AnthropicContentBlockParam] = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif content:
            blocks.extend(content)
        blocks.extend(_tool_calls_to_blocks(message.get("tool_calls")))
        # Collapse back to a bare string when there's only plain text.
        if len(blocks) == 1 and blocks[0].get("type") == "text":
            return {"role": "assistant", "content": blocks[0]["text"]}
        return cast(AnthropicMessageParam, {"role": "assistant", "content": blocks})

    if role == "tool":
        result: dict = {
            "type": "tool_result",
            "tool_use_id": message.get("tool_call_id") or "",
        }
        _set_if_present(result, "content", _content_to_anthropic(message.get("content")))
        _set_if_present(result, "is_error", message.get("is_error"))
        # Anthropic nests tool results inside a user message.
        return cast(
            AnthropicMessageParam,
            {"role": "user", "content": [cast(AnthropicToolResultBlock, result)]},
        )

    if role == "function":
        raise NotImplementedError(
            "OpenAI's legacy 'function' role has no Anthropic equivalent; "
            "migrate to tools/tool_calls (tool role) before converting."
        )

    raise ValueError(f"Unsupported message role: {role!r}")


def _extract_system(
    messages: list,
) -> tuple[Optional[AnthropicSystemParam], list]:
    """Pull OpenAI `role: "system"` messages out of the conversation.

    Anthropic has no system *message*; the system prompt lives in the top-level
    `system` field. A single system message becomes a string (or block list);
    multiple system messages are concatenated into a block list.
    """
    system_messages = [m for m in messages if m.get("role") == "system"]
    remaining = [m for m in messages if m.get("role") != "system"]
    if not system_messages:
        return None, remaining
    contents = [_content_to_anthropic(m.get("content")) for m in system_messages]
    if len(contents) == 1:
        return cast(AnthropicSystemParam, contents[0]), remaining
    blocks: List[AnthropicContentBlockParam] = []
    for c in contents:
        if c is None:
            continue
        if isinstance(c, str):
            blocks.append({"type": "text", "text": c})
        else:
            blocks.extend(c)
    return cast(AnthropicSystemParam, blocks), remaining


# ---------- tools ----------

def _tool_to_anthropic(tool: dict) -> AnthropicTool:
    """OpenAI tool ({type: "function", function: {...}}) -> Anthropic tool."""
    if tool.get("type") not in (None, "function"):
        raise NotImplementedError(
            f"OpenAI tool type {tool.get('type')!r} has no Anthropic equivalent"
        )
    fn = tool.get("function") or {}
    out: dict = {
        "name": fn.get("name") or "",
        "input_schema": fn.get("parameters") or {"type": "object"},
    }
    _set_if_present(out, "description", fn.get("description"))
    _set_if_present(out, "type", "custom")
    return cast(AnthropicTool, out)


def _tool_choice_to_anthropic(tool_choice) -> AnthropicToolChoiceParam:
    """OpenAI tool_choice (str | {type: "function", ...}) -> Anthropic tool_choice."""
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "required":
            return {"type": "any"}
        raise ValueError(f"Unknown OpenAI tool_choice string: {tool_choice!r}")
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError(f"Unsupported OpenAI tool_choice: {tool_choice!r}")
        name = (tool_choice.get("function") or {}).get("name")
        out: dict = {"type": "tool", "name": name} if name else {"type": "any"}
        if tool_choice.get("parallel_tool_calls") is False:
            out["disable_parallel_tool_use"] = True
        return cast(AnthropicToolChoiceParam, out)
    raise TypeError(f"Unsupported OpenAI tool_choice type: {type(tool_choice).__name__}")


# ---------- structured outputs ----------

def _response_format_to_output_config(response_format: Optional[dict]) -> Optional[AnthropicOutputConfig]:
    """OpenAI `response_format` -> Anthropic `output_config`.

    See https://platform.claude.com/docs/en/build-with-claude/structured-outputs
    Anthropic's structured outputs only support a JSON-schema-constrained
    format (no schema-less `json_object` equivalent), so anything else raises.
    """
    if response_format is None:
        return None
    rtype = response_format.get("type")
    if rtype == "text":
        return None
    if rtype != "json_schema":
        raise NotImplementedError(
            f"OpenAI response_format type {rtype!r} has no Anthropic equivalent"
        )
    schema = (response_format.get("json_schema") or {}).get("schema")
    if schema is None:
        raise ValueError("response_format.json_schema.schema is required")
    return {"format": {"type": "json_schema", "schema": schema}}


# ---------- request ----------

def to_anthropic_request(
    request: ChatCompletionRequest,
    extras: dict
) -> AnthropicRequest:
    """
    Convert an OpenAI-style ChatCompletionRequest to the Anthropic Messages format.

    `thinking`, `top_k` and `service_tier` have no OpenAI source, so they are
    accepted as explicit keyword-only arguments and forwarded when set.

    Raises:
        ValueError: when `max_tokens`/`max_completion_tokens` is missing (the
            Anthropic API requires `max_tokens`), or on an unmappable value.
        NotImplementedError: for OpenAI features with no Anthropic equivalent
            (audio content parts, the legacy `function` role, non-function tools).
    """


    thinking: Optional[AnthropicThinkingConfig] = extras.get("thinking")
    top_k: Optional[int] = extras.get("top_k")
    service_tier: Optional[Literal["standard", "priority"]] = extras.get("service_tier")

    messages = list(request.get("messages") or [])

    # max_tokens is required by the Anthropic API; accept either OpenAI spelling.
    max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
    if max_tokens is None:
        # raise ValueError(
        #     "Anthropic requires 'max_tokens'; set 'max_completion_tokens' or "
        #     "'max_tokens' on the request before converting."
        # )
        max_tokens = 4096  # NOTE: also litellm's default, see https://docs.litellm.ai/docs/providers/anthropic

    system, messages = _extract_system(messages)

    out: dict = {
        "model": request["model"],
        "max_tokens": max_tokens,
        "messages": [_message_to_anthropic(m) for m in messages],
    }
    _set_if_present(out, "system", system)

    stop = request.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    _set_if_present(out, "stop_sequences", stop)

    for key in ("temperature", "top_p", "stream"):
        _set_if_present(out, key, request.get(key))

    _set_if_present(out, "top_k", top_k)

    tools = request.get("tools")
    if tools:
        out["tools"] = [_tool_to_anthropic(t) for t in tools]

    tool_choice = request.get("tool_choice")
    if tool_choice is not None:
        out["tool_choice"] = _tool_choice_to_anthropic(tool_choice)

    _set_if_present(out, "thinking", thinking)
    _set_if_present(out, "service_tier", service_tier)
    _set_if_present(out, "output_config", _response_format_to_output_config(request.get("response_format")))

    # Anthropic only accepts metadata.user_id; fold OpenAI `user` in as well.
    metadata: dict = {}
    _set_if_present(metadata, "user_id", request.get("user"))
    oai_metadata = request.get("metadata")
    if isinstance(oai_metadata, dict):
        _set_if_present(metadata, "user_id", oai_metadata.get("user_id"))
    _set_if_present(out, "metadata", metadata or None)

    return cast(AnthropicRequest, out)


# ==================== Anthropic -> OpenAI ====================

# ---------- content ----------

def _image_source_to_url(source: Optional[dict]) -> dict:
    """Anthropic image `source` -> OpenAI `image_url` part."""
    source = source or {}
    stype = source.get("type")
    if stype == "base64":
        media_type = source.get("media_type", "")
        data = source.get("data", "")
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    if stype == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    raise NotImplementedError(f"Anthropic image source type {stype!r} has no OpenAI equivalent")


def _block_to_content_part(block: dict) -> dict:
    """Anthropic content block -> OpenAI content part."""
    btype = block.get("type")
    if btype == "text":
        return {"type": "text", "text": block.get("text") or ""}
    if btype == "image":
        return _image_source_to_url(block.get("source"))
    raise NotImplementedError(
        f"Anthropic content block type {btype!r} has no OpenAI equivalent"
    )


def _anthropic_content_to_openai(
    content: Optional[Union[str, List[dict]]],
) -> Optional[Union[str, List[dict]]]:
    """Anthropic content (str | list[block]) -> OpenAI content (str | list[part])."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b for b in content if b.get("type") in ("text", "image")]
        return [_block_to_content_part(b) for b in parts]
    raise TypeError(f"Unsupported Anthropic content: {type(content).__name__}")


# ---------- messages ----------

def _tool_use_blocks_to_calls(blocks: List[dict]) -> List[dict]:
    """Anthropic `tool_use` blocks -> OpenAI assistant `tool_calls`."""
    return [
        {
            "id": b.get("id", ""),
            "type": "function",
            "function": {
                "name": b.get("name", ""),
                "arguments": json.dumps(b.get("input", {}) or {}),
            },
        }
        for b in blocks
        if b.get("type") == "tool_use"
    ]


def _user_message_to_openai(message: dict) -> List[dict]:
    """Anthropic user message -> OpenAI message(s).

    A user message can carry `tool_result` blocks alongside (or instead of)
    plain content; each `tool_result` becomes its own OpenAI `tool` message,
    since OpenAI has no way to nest a tool result inside a user message.
    """
    content = message.get("content")
    if isinstance(content, str) or content is None:
        return [{"role": "user", "content": content or ""}]

    tool_results = [b for b in content if b.get("type") == "tool_result"]
    other_blocks = [b for b in content if b.get("type") != "tool_result"]

    out: List[dict] = []
    if other_blocks or not tool_results:
        out.append({"role": "user", "content": _anthropic_content_to_openai(other_blocks)})
    for b in tool_results:
        tool_message: dict = {
            "role": "tool",
            "tool_call_id": b.get("tool_use_id", ""),
            "content": _anthropic_content_to_openai(b.get("content")) or "",
        }
        _set_if_present(tool_message, "is_error", b.get("is_error"))
        out.append(tool_message)
    return out


def _assistant_message_to_openai(message: dict) -> dict:
    """Anthropic assistant message -> OpenAI assistant message."""
    content = message.get("content")
    if isinstance(content, str) or content is None:
        return {"role": "assistant", "content": content}

    text_blocks = [b for b in content if b.get("type") == "text"]
    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

    text = "".join(b.get("text") or "" for b in text_blocks)
    out: dict = {"role": "assistant", "content": text or None}
    tool_calls = _tool_use_blocks_to_calls(tool_use_blocks)
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _anthropic_message_to_openai(message: dict) -> List[dict]:
    """Convert one Anthropic message to one or more OpenAI chat messages."""
    role = message.get("role")
    if role == "user":
        return _user_message_to_openai(message)
    if role == "assistant":
        return [_assistant_message_to_openai(message)]
    raise ValueError(f"Unsupported Anthropic message role: {role!r}")


def _system_to_message(system: Optional[Union[str, List[dict]]]) -> Optional[dict]:
    """Anthropic top-level `system` -> an OpenAI `role: "system"` message."""
    if system is None:
        return None
    if isinstance(system, str):
        return {"role": "system", "content": system}
    text = "".join(b.get("text") or "" for b in system if b.get("type") == "text")
    return {"role": "system", "content": text}


# ---------- tools ----------

def _tool_from_anthropic(tool: dict) -> dict:
    """Anthropic tool -> OpenAI tool ({type: "function", function: {...}})."""
    if tool.get("type") not in (None, "custom"):
        raise NotImplementedError(
            f"Anthropic tool type {tool.get('type')!r} has no OpenAI equivalent"
        )
    function: dict = {
        "name": tool.get("name") or "",
        "parameters": tool.get("input_schema") or {"type": "object"},
    }
    _set_if_present(function, "description", tool.get("description"))
    return {"type": "function", "function": function}


def _tool_choice_from_anthropic(tool_choice: dict) -> Union[str, dict]:
    """Anthropic tool_choice -> OpenAI tool_choice (str | {type: "function", ...})."""
    ttype = tool_choice.get("type")
    if ttype == "auto":
        return "auto"
    if ttype == "none":
        return "none"
    if ttype == "any":
        return "required"
    if ttype == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name")}}
    raise ValueError(f"Unknown Anthropic tool_choice type: {ttype!r}")


# ---------- structured outputs ----------

def _output_config_to_response_format(
    output_config: Optional[dict],
) -> Optional[dict]:
    """Anthropic `output_config` -> OpenAI `response_format`."""
    if output_config is None:
        return None
    fmt = output_config.get("format") or {}
    if fmt.get("type") != "json_schema":
        raise NotImplementedError(
            f"Anthropic output_config format {fmt.get('type')!r} has no OpenAI equivalent"
        )
    schema = fmt.get("schema")
    if schema is None:
        raise ValueError("output_config.format.schema is required")
    return {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}


# ---------- request ----------

def from_anthropic_request(
    request: AnthropicRequest,
) -> Tuple[ChatCompletionRequest, Dict[str, Any]]:
    """
    Convert an Anthropic Messages request to an OpenAI-style ChatCompletionRequest.

    Returns a `(chat_completion_request, extras)` tuple: `extras` carries
    Anthropic fields with no ChatCompletions equivalent (`thinking`, `top_k`,
    `service_tier`), symmetric with the keyword-only arguments
    `to_anthropic_request` accepts via its own `extras` parameter.

    Raises:
        NotImplementedError: for Anthropic features with no OpenAI equivalent
            (document content blocks, non-function tools).
    """
    messages: List[dict] = []

    system_message = _system_to_message(request.get("system"))
    if system_message is not None:
        messages.append(system_message)

    for m in request.get("messages") or []:
        messages.extend(_anthropic_message_to_openai(m))

    out: dict = {
        "model": request["model"],
        "messages": messages,
        "max_tokens": request["max_tokens"],
    }

    stop_sequences = request.get("stop_sequences")
    if stop_sequences is not None:
        out["stop"] = stop_sequences

    for key in ("temperature", "top_p", "stream"):
        _set_if_present(out, key, request.get(key))

    tools = request.get("tools")
    if tools:
        out["tools"] = [_tool_from_anthropic(t) for t in tools]

    tool_choice = request.get("tool_choice")
    if tool_choice is not None:
        result = _tool_choice_from_anthropic(tool_choice)
        out["tool_choice"] = result
        if isinstance(tool_choice, dict) and tool_choice.get("disable_parallel_tool_use") is True:
            out["parallel_tool_calls"] = False

    _set_if_present(out, "response_format", _output_config_to_response_format(request.get("output_config")))

    metadata = request.get("metadata") or {}
    _set_if_present(out, "user", metadata.get("user_id"))

    extras: Dict[str, Any] = {}
    _set_if_present(extras, "thinking", request.get("thinking"))
    _set_if_present(extras, "top_k", request.get("top_k"))
    _set_if_present(extras, "service_tier", request.get("service_tier"))

    return cast(ChatCompletionRequest, out), extras
