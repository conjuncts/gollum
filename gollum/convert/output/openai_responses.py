"""
Adapter: OpenAI Responses response -> OpenAI ChatCompletions response.

Reference (Responses): https://platform.openai.com/docs/api-reference/responses/object
Reference (ChatCompletions): https://platform.openai.com/docs/api-reference/chat/object
"""

import time
from typing import Any, Dict, List, Optional

from gollum.types import GollumResponse
from gollum.types.chat_completions import (
    ChatCompletionResponse,
    Choice,
    CompletionTokensDetails,
    PromptTokensDetails,
    Usage,
)

_STATUS_FINISH_REASON_MAP: Dict[str, str] = {
    "completed": "stop",
    "cancelled": "stop",
    "failed": "stop",
}

_INCOMPLETE_REASON_MAP: Dict[str, str] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


def _map_finish_reason(response: dict, has_tool_calls: bool) -> Optional[str]:
    if has_tool_calls:
        return "tool_calls"
    status = response.get("status")
    if status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return _INCOMPLETE_REASON_MAP.get(reason, "length")
    if status is None:
        return None
    return _STATUS_FINISH_REASON_MAP.get(status, "stop")


def _build_usage(responses_usage: Optional[Dict[str, Any]]) -> Optional[Usage]:
    if not responses_usage:
        return None

    # Pop consumed fields so they don't leak into extras below.
    input_tokens = responses_usage.pop("input_tokens", 0) or 0
    output_tokens = responses_usage.pop("output_tokens", 0) or 0
    input_details = responses_usage.pop("input_tokens_details", None) or {}
    output_details = responses_usage.pop("output_tokens_details", None) or {}

    usage: Usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": responses_usage.pop("total_tokens", None) or (input_tokens + output_tokens),
    }

    cached_tokens = input_details.pop("cached_tokens", None)
    if cached_tokens:
        prompt_details: PromptTokensDetails = {"cached_tokens": cached_tokens}
        usage["prompt_tokens_details"] = prompt_details

    reasoning_tokens = output_details.pop("reasoning_tokens", None)
    if reasoning_tokens:
        completion_details: CompletionTokensDetails = {"reasoning_tokens": reasoning_tokens}
        usage["completion_tokens_details"] = completion_details

    return usage


def responses_response_to_completion(response: dict) -> GollumResponse:
    # Work on a copy so the .pop() calls below don't mutate the caller's dict.
    response = dict(response)
    output_items: List[Dict[str, Any]] = response.pop("output", []) or []

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    extras: Dict[str, Any] = {
        "openai_responses.annotations": [],  # {content_index, annotations}
        "openai_responses.unhandled_items": [],  # unrecognized item types, kept verbatim
    }

    for i, item in enumerate(output_items):
        item_type = item.get("type")

        if item_type == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text") or "")
                    if part.get("annotations"):
                        extras["openai_responses.annotations"].append(
                            {"content_index": i, "annotations": part["annotations"]}
                        )
                elif part.get("type") == "refusal":
                    extras.setdefault("openai_responses.refusals", []).append(
                        {"content_index": i, "refusal": part.get("refusal")}
                    )

        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or "",
                },
            })

        elif item_type == "reasoning":
            summary_texts = [s.get("text", "") for s in (item.get("summary") or [])]
            if summary_texts:
                reasoning_parts.append("".join(summary_texts))
            extras.setdefault("openai_responses.reasoning_items", []).append(
                {"content_index": i, "id": item.get("id")}
            )

        else:
            extras["openai_responses.unhandled_items"].append({"content_index": i, "item": item})

    # Drop empty lists so extras only reflects what was actually present.
    extras = {k: v for k, v in extras.items() if v}

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }

    if tool_calls:
        message["tool_calls"] = tool_calls

    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)

    choice: Choice = {
        "index": 0,
        "finish_reason": _map_finish_reason(response, bool(tool_calls)),
        "message": message,  # type: ignore[typeddict-item]
    }

    completion: ChatCompletionResponse = {
        "id": response.pop("id", ""),
        "object": "chat.completion",
        "created": int(response.pop("created_at", None) or time.time()),
        "model": response.pop("model", ""),
        "choices": [choice],
    }

    responses_usage = dict(response.pop("usage", None) or {})
    usage = _build_usage(responses_usage)
    if usage is not None:
        completion["usage"] = usage

    # Response-level fields that ChatCompletionResponse has no slot for at all.
    # Pop each known field so that any remaining (schema-extension) fields can
    # be preserved verbatim below instead of being silently dropped.
    extras_2: Dict[str, Any] = {}

    for key in (
        "object", "status", "instructions", "metadata", "parallel_tool_calls",
        "previous_response_id", "temperature", "text", "tool_choice", "tools",
        "top_p", "truncation", "user", "incomplete_details", "error",
    ):
        value = response.pop(key, None)
        if value is not None:
            extras_2[f"openai_responses.{key}"] = value

    # Anything left over is an unknown field; keep it verbatim.
    extras_2.update({f"openai_responses.{key}": value for key, value in response.items()})
    extras_2.update({f"openai_responses.usage.{key}": value for key, value in responses_usage.items()})

    extras.update(extras_2)

    return GollumResponse(response=completion, extras=extras, metadata={})
