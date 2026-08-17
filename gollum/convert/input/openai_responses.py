"""
Adapter: OpenAI ChatCompletionRequest -> OpenAI Responses request.

Reference (Responses): https://platform.openai.com/docs/api-reference/responses/create
Reference (ChatCompletions): https://platform.openai.com/docs/api-reference/chat/create

The conversion is one-directional (ChatCompletions -> Responses); fields with
no Responses equivalent (logprobs, n, seed, ...) are dropped.
"""

from typing import List, Optional, Union, cast

from gollum.types.chat_completions import ChatCompletionRequest
from gollum.types.openai_responses import (
    ResponsesFunctionCallItem,
    ResponsesFunctionCallOutputItem,
    ResponsesFunctionTool,
    ResponsesInputContentPart,
    ResponsesInputItem,
    ResponsesInputMessage,
    ResponsesReasoningConfig,
    ResponsesRequest,
    ResponsesTextConfig,
    ResponsesToolChoice,
)


def _set_if_present(out: dict, key: str, value, *, allow_null: bool = False) -> None:
    """Only add `key` to `out` when `value` is not None."""
    if value is not None or allow_null:
        out[key] = value


# ---------- content ----------

def _content_part_to_responses(part: dict) -> ResponsesInputContentPart:
    """OpenAI ChatCompletions content part -> Responses input content part."""
    ptype = part.get("type")
    if ptype == "text":
        return {"type": "input_text", "text": part.get("text") or ""}
    if ptype == "image_url":
        image_url = part.get("image_url") or {}
        out: dict = {"type": "input_image", "image_url": image_url.get("url")}
        if image_url.get("detail") is not None:
            out["detail"] = image_url["detail"]
        return cast(ResponsesInputContentPart, out)
    raise NotImplementedError(
        f"ChatCompletions content part type {ptype!r} has no Responses equivalent"
    )


def _content_to_responses(content) -> Union[str, List[ResponsesInputContentPart]]:
    """OpenAI ChatCompletions message content (str | list[part]) -> Responses content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_content_part_to_responses(p) for p in content]
    raise TypeError(f"Unsupported message content: {type(content).__name__}")


# ---------- messages ----------

def _tool_calls_to_items(tool_calls) -> List[ResponsesFunctionCallItem]:
    """OpenAI assistant `tool_calls` -> Responses `function_call` items."""
    items: List[ResponsesFunctionCallItem] = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        items.append({
            "type": "function_call",
            "call_id": call.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "") or "",
        })
    return items


def _message_to_items(message: dict) -> List[ResponsesInputItem]:
    """Convert one ChatCompletions message to zero or more Responses input items."""
    role = message.get("role")

    if role in ("user", "system", "developer"):
        return [cast(ResponsesInputMessage, {
            "type": "message",
            "role": role,
            "content": _content_to_responses(message.get("content")),
        })]

    if role == "assistant":
        items: List[ResponsesInputItem] = []
        content = message.get("content")
        if content is not None and content != "":
            items.append(cast(ResponsesInputMessage, {
                "type": "message",
                "role": "assistant",
                "content": _content_to_responses(content),
            }))
        items.extend(_tool_calls_to_items(message.get("tool_calls")))
        return items

    if role == "tool":
        return [cast(ResponsesFunctionCallOutputItem, {
            "type": "function_call_output",
            "call_id": message.get("tool_call_id") or "",
            "output": _stringify_tool_output(message.get("content")),
        })]

    if role == "function":
        raise NotImplementedError(
            "OpenAI's legacy 'function' role has no Responses equivalent; "
            "migrate to tools/tool_calls (tool role) before converting."
        )

    raise ValueError(f"Unsupported message role: {role!r}")


def _stringify_tool_output(content) -> str:
    """Responses `function_call_output.output` must be a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
        return "".join(parts)
    raise TypeError(f"Unsupported tool message content: {type(content).__name__}")


def _extract_instructions(messages: list) -> tuple[Optional[str], list]:
    """Pull OpenAI `role: "system"` messages out and fold them into `instructions`.

    Responses does support `system`/`developer` role input items directly, but
    the top-level `instructions` field is the more idiomatic destination for a
    ChatCompletions system prompt.
    """
    system_messages = [m for m in messages if m.get("role") == "system"]
    remaining = [m for m in messages if m.get("role") != "system"]
    if not system_messages:
        return None, remaining
    parts = []
    for m in system_messages:
        content = _content_to_responses(m.get("content"))
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append("".join(p.get("text", "") for p in content if p.get("type") == "input_text"))
    return "\n\n".join(p for p in parts if p), remaining


# ---------- tools ----------

def _tool_to_responses(tool: dict) -> ResponsesFunctionTool:
    """OpenAI tool ({type: "function", function: {...}}) -> Responses tool."""
    if tool.get("type") not in (None, "function"):
        raise NotImplementedError(
            f"ChatCompletions tool type {tool.get('type')!r} has no Responses equivalent"
        )
    fn = tool.get("function") or {}
    out: dict = {
        "type": "function",
        "name": fn.get("name") or "",
        "parameters": fn.get("parameters") or {"type": "object"},
    }
    _set_if_present(out, "description", fn.get("description"), allow_null=True)
    _set_if_present(out, "strict", fn.get("strict"))
    return cast(ResponsesFunctionTool, out)


def _tool_choice_to_responses(tool_choice) -> ResponsesToolChoice:
    """OpenAI tool_choice (str | {type: "function", ...}) -> Responses tool_choice."""
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return cast(ResponsesToolChoice, tool_choice)
        raise ValueError(f"Unknown OpenAI tool_choice string: {tool_choice!r}")
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError(f"Unsupported OpenAI tool_choice: {tool_choice!r}")
        name = (tool_choice.get("function") or {}).get("name")
        return {"type": "function", "name": name}
    raise TypeError(f"Unsupported OpenAI tool_choice type: {type(tool_choice).__name__}")


# ---------- structured outputs / reasoning ----------

def _response_format_to_text_config(response_format: Optional[dict]) -> Optional[ResponsesTextConfig]:
    """OpenAI `response_format` -> Responses `text.format`."""
    if response_format is None:
        return None
    rtype = response_format.get("type")
    if rtype == "text":
        return None
    if rtype == "json_object":
        return {"format": {"type": "json_object"}}
    if rtype != "json_schema":
        raise NotImplementedError(
            f"OpenAI response_format type {rtype!r} has no Responses equivalent"
        )
    json_schema = response_format.get("json_schema") or {}
    schema = json_schema.get("schema")
    if schema is None:
        raise ValueError("response_format.json_schema.schema is required")
    out: dict = {"type": "json_schema", "name": json_schema.get("name") or "response", "schema": schema}
    _set_if_present(out, "strict", json_schema.get("strict"))
    return {"format": out}


def _reasoning_effort_to_config(reasoning_effort: Optional[str]) -> Optional[ResponsesReasoningConfig]:
    if reasoning_effort is None:
        return None
    return {"effort": reasoning_effort}


# ---------- request ----------

def to_responses_request(
    request: ChatCompletionRequest,
    extras: dict,
) -> ResponsesRequest:
    """
    Convert an OpenAI-style ChatCompletionRequest to the OpenAI Responses format.

    `previous_response_id`, `store` and `truncation` have no ChatCompletions
    source, so they are accepted via `extras` and forwarded when set.

    Raises:
        NotImplementedError: for ChatCompletions features with no Responses
            equivalent (the legacy `function` role, non-function tools).
    """
    messages = list(request.get("messages") or [])
    instructions, messages = _extract_instructions(messages)

    input_items: List[ResponsesInputItem] = []
    for m in messages:
        input_items.extend(_message_to_items(m))

    out: dict = {
        "model": request["model"],
        "input": input_items,
    }
    _set_if_present(out, "instructions", instructions)

    max_output_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
    _set_if_present(out, "max_output_tokens", max_output_tokens)

    for key in ("temperature", "top_p", "stream", "user", "parallel_tool_calls", "metadata"):
        _set_if_present(out, key, request.get(key))

    tools = request.get("tools")
    if tools:
        out["tools"] = [_tool_to_responses(t) for t in tools]

    tool_choice = request.get("tool_choice")
    if tool_choice is not None:
        out["tool_choice"] = _tool_choice_to_responses(tool_choice)

    _set_if_present(out, "text", _response_format_to_text_config(request.get("response_format")))
    _set_if_present(out, "reasoning", _reasoning_effort_to_config(request.get("reasoning_effort")))

    _set_if_present(out, "previous_response_id", extras.get("previous_response_id"))
    _set_if_present(out, "store", extras.get("store"))
    _set_if_present(out, "truncation", extras.get("truncation"))
    _set_if_present(out, "service_tier", extras.get("service_tier") or request.get("service_tier"))

    return cast(ResponsesRequest, out)
