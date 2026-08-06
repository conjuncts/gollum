import json
import time
from typing import Any, Dict, List, Optional

from gollum.types.chat_completions import (
    ChatCompletionResponse,
    Choice,
    PromptTokensDetails,
    Usage,
)
from gollum.types import GollumResponse

_STOP_REASON_MAP: Dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _map_finish_reason(stop_reason: Optional[str]) -> Optional[str]:
    if stop_reason is None:
        return None
    return _STOP_REASON_MAP.get(stop_reason, "stop")


def _build_usage(anthropic_usage: Optional[Dict[str, Any]]) -> Optional[Usage]:
    if not anthropic_usage:
        return None

    # Pop consumed fields so they don't leak into extras_2 below.
    input_tokens = anthropic_usage.pop("input_tokens", 0) or 0
    output_tokens = anthropic_usage.pop("output_tokens", 0) or 0
    cache_read = anthropic_usage.pop("cache_read_input_tokens", None)

    usage: Usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    if cache_read:
        prompt_details: PromptTokensDetails = {"cached_tokens": cache_read}
        usage["prompt_tokens_details"] = prompt_details

    return usage


def anthropic_message_to_completion(anthropic_message: dict) -> GollumResponse:
    # Work on a copy so the .pop() calls below don't mutate the caller's dict.
    anthropic_message = dict(anthropic_message)
    content_blocks: List[Dict[str, Any]] = anthropic_message.pop("content", []) or []

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    # Boundary layouts: recover per-block slices of the joined strings via
    # `length` (and cumulative offset) without storing the text twice.
    # `content_index` ties each entry back to its position in the original
    # anthropic_message["content"] array.
    text_block_layout: List[Dict[str, Any]] = []
    thinking_block_layout: List[Dict[str, Any]] = []

    extras: Dict[str, Any] = {
        "anthropic.thinking_blocks": [],  # {content_index, signature} (text recoverable via layout)
        "anthropic.redacted_thinking_blocks": [],  # {content_index, data}
        "anthropic.citations": [],  # {content_index, citations}
        "anthropic.server_tool_use_blocks": [],
        "anthropic.web_search_tool_result_blocks": [],
        "anthropic.unhandled_blocks": [],  # unrecognized block types, kept verbatim
    }

    for i, block in enumerate(content_blocks):
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")
            text_parts.append(text)
            text_block_layout.append({"content_index": i, "type": "text", "length": len(text)})
            if block.get("citations"):
                extras["anthropic.citations"].append(
                    {"content_index": i, "citations": block["citations"]}
                )

        elif block_type == "thinking":
            thinking_text = block.get("thinking", "")
            reasoning_parts.append(thinking_text)
            thinking_block_layout.append(
                {"content_index": i, "type": "thinking", "length": len(thinking_text)}
            )
            extras["anthropic.thinking_blocks"].append(
                {"content_index": i, "signature": block.get("signature")}
            )

        elif block_type == "redacted_thinking":
            extras["anthropic.redacted_thinking_blocks"].append(
                {"content_index": i, "data": block.get("data")}
            )

        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}) or {}),
                    },
                }
            )

        elif block_type == "server_tool_use":
            extras["anthropic.server_tool_use_blocks"].append({"content_index": i, "block": block})

        elif block_type == "web_search_tool_result":
            extras["anthropic.web_search_tool_result_blocks"].append(
                {"content_index": i, "block": block}
            )

        else:
            extras["anthropic.unhandled_blocks"].append({"content_index": i, "block": block})

    if text_block_layout:
        extras["anthropic.text_block_layout"] = text_block_layout
    if thinking_block_layout:
        extras["anthropic.thinking_block_layout"] = thinking_block_layout

    # Drop empty lists so extras only reflects what was actually present.
    extras = {k: v for k, v in extras.items() if v}

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }

    if tool_calls:
        message["tool_calls"] = tool_calls

    if reasoning_parts:
        # Convenience convention (LiteLLM/DeepSeek/vLLM-style). Boundaries to
        # recover individual thinking blocks (and their signatures) live in
        # extras["anthropic.thinking_block_layout"] / extras["anthropic.thinking_blocks"].
        message["reasoning_content"] = "".join(reasoning_parts)

    choice: Choice = {
        "index": 0,
        "finish_reason": _map_finish_reason(anthropic_message.get("stop_reason")),
        "message": message,  # type: ignore[typeddict-item]
    }

    response: ChatCompletionResponse = {
        "id": anthropic_message.pop("id", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anthropic_message.pop("model", ""),
        "choices": [choice],
    }

    anthropic_usage = dict(anthropic_message.pop("usage") or {})
    usage = _build_usage(anthropic_usage)
    if usage is not None:
        response["usage"] = usage

    # Response-level fields that ChatCompletionResponse has no slot for at all.
    # Pop each known field so that any remaining (schema-extension) fields can
    # be preserved verbatim below instead of being silently dropped.
    extras_2: Dict[str, Any] = {}

    for key in ("type", "role", "stop_reason", "stop_sequence", "container"):
        value = anthropic_message.pop(key, None)
        if value is not None:
            extras_2[f"anthropic.{key}"] = value

    for key, extras_key in (
        ("cache_creation_input_tokens", "anthropic.cache_creation_input_tokens"),
        ("cache_creation", "anthropic.cache_creation"),
        ("server_tool_use", "anthropic.server_tool_use_usage"),
    ):
        value = anthropic_usage.pop(key, None)
        if value is not None:
            extras_2[extras_key] = value

    # Anything left over is an unknown field; keep it verbatim.
    extras_2.update({f"anthropic.{key}": value for key, value in anthropic_message.items()})
    extras_2.update({f"anthropic.{key}": value for key, value in anthropic_usage.items()})

    extras.update(extras_2)

    return GollumResponse(response=response, extras=extras, metadata={})