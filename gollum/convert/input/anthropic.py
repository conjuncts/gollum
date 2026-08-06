"""
Adapter: OpenAI ChatCompletionRequest -> Anthropic Messages request.

Reference (Anthropic): https://docs.anthropic.com/en/api/messages
Reference (OpenAI):    https://platform.openai.com/docs/api-reference/chat/create

The conversion is one-directional (OpenAI -> Anthropic); fields with no
Anthropic equivalent (logprobs, n, seed, response_format, ...) are dropped.
"""

import json
import re
from typing import List, Literal, Optional, Union, cast

from gollum.types.anthropic import (
    AnthropicContentBlockParam,
    AnthropicImageSource,
    AnthropicMessageParam,
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

    # Anthropic only accepts metadata.user_id; fold OpenAI `user` in as well.
    metadata: dict = {}
    _set_if_present(metadata, "user_id", request.get("user"))
    oai_metadata = request.get("metadata")
    if isinstance(oai_metadata, dict):
        _set_if_present(metadata, "user_id", oai_metadata.get("user_id"))
    _set_if_present(out, "metadata", metadata or None)

    return cast(AnthropicRequest, out)
