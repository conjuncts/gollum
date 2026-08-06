# ============================================================
# Round-trip helpers for ChatCompletionResponse
#
# The response shape has far less polymorphism than the request shape:
# the only field that varies in structure is `choices[].message`, which
# is exactly the same {content: str, content_parts: list} / tool_calls /
# function_call split used on the request side. Everything else in the
# response (usage, logprobs, ids, etc.) already matches its Polars
# struct 1:1, so those fields are passed straight through on the way in.
# On the way back out, `usage` needs a field-by-field rebuild
# (_usage_from_flat): a DataFrame round trip fills any struct field the
# source dict omitted with null, and OpenAI expects unset fields to be
# absent rather than null.
#
# NOTE: same caveat as the request-side module — this does not
# distinguish "unset" from "null", so round trip behavior for
# explicit nulls may not be perfect.
#
# NOTE: `reasoning_content` (a non-spec field some providers such as
# DeepSeek / OpenAI OSS / LiteLLM attach to responses) is NOT part of
# ChatCompletionResponseSchema in gollum.types.pl_chat_completions, so
# it is intentionally dropped on the way in and absent on the way out.
# If you need it preserved, add a `"reasoning_content": pl.Utf8` field
# to ChatCompletionResponseSchema and thread it through the two
# functions below the same way `system_fingerprint` is handled.
# ============================================================

from gollum.types.chat_completions import ChatCompletionResponseModel

# Reuse the message split/join logic — it's identical for request and
# response messages (str|list content, tool_calls, function_call, etc.)
from gollum.convert.polars.serialize_request import (
    _message_to_flat,
    _message_from_flat,
    _set_if_present,
)


# ---------- choice: wraps the polymorphic `message` field ----------

def _choice_to_flat(choice: dict) -> dict:
    message = choice.get("message")
    return {
        "index": choice.get("index"),
        "message": _message_to_flat(message) if message is not None else None,
        # logprobs already matches its Polars struct field-for-field
        # (content/refusal lists of {token, logprob, bytes, top_logprobs})
        "logprobs": choice.get("logprobs"),
        "finish_reason": choice.get("finish_reason"),
    }


def _choice_from_flat(choice: dict) -> dict:
    out: dict = {}
    _set_if_present(out, "index", choice.get("index"))
    message = choice.get("message")
    if message is not None:
        out["message"] = _message_from_flat(message)
    _set_if_present(out, "logprobs", choice.get("logprobs"))
    _set_if_present(out, "finish_reason", choice.get("finish_reason"))
    return out


def _usage_from_flat(usage: dict) -> dict:
    """Rebuild an OpenAI-shaped usage dict from a Polars row value.

    `usage` maps 1:1 onto its Polars struct, but a DataFrame round trip
    fills any struct field the source dict omitted with null, which
    `to_dicts()` returns back as explicit `None` entries. OpenAI expects
    unset fields to be absent rather than null, so rebuild the dict
    field-by-field (nested details structs included), dropping nulls.
    """
    out: dict = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _set_if_present(out, key, usage.get(key))

    ctd = usage.get("completion_tokens_details")
    if ctd:
        details: dict = {}
        for key in ("reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"):
            _set_if_present(details, key, ctd.get(key))
        if details:
            out["completion_tokens_details"] = details

    ptd = usage.get("prompt_tokens_details")
    if ptd:
        details = {}
        for key in ("cached_tokens", "audio_tokens"):
            _set_if_present(details, key, ptd.get(key))
        if details:
            out["prompt_tokens_details"] = details

    return out


# ============================================================
# Public round-trip API
# ============================================================

def pl_serialize_chat_response(data: ChatCompletionResponseModel) -> dict:
    """
    Converts a ChatCompletionResponse to a dictionary that is serializable by
    Polars, i.e. one whose shape matches ChatCompletionResponseSchema exactly.

    The only structural transformation needed is on `choices[].message`
    (str|list content, tool_calls, function_call, refusal) — everything
    else (usage, logprobs, id/object/created/model/system_fingerprint/
    service_tier) is already struct-shaped and passed through as-is.

    Missing/absent OpenAI fields become None, which Polars renders as null
    when the dict is fed into `pl.DataFrame([...], schema=ChatCompletionResponseSchema)`.
    """
    data = dict(data)  # TypedDict is just a dict at runtime; avoid mutating caller's copy

    choices = data.get("choices")

    return {
        "id": data.get("id"),
        "object": data.get("object"),
        "created": data.get("created"),
        "model": data.get("model"),
        "choices": [_choice_to_flat(c) for c in choices] if choices is not None else None,
        "usage": data.get("usage"),
        "system_fingerprint": data.get("system_fingerprint"),
        "service_tier": data.get("service_tier"),
    }


def pl_deserialize_chat_response(data: dict) -> ChatCompletionResponseModel:
    """
    Converts a flattened dict (matching ChatCompletionResponseSchema, e.g. one
    row of `df_responses.to_dicts()`) back into an OpenAI-shaped
    ChatCompletionResponse dict. Null/None fields are dropped rather than
    included as `None`, since OpenAI's TypedDict fields are meant to be
    absent, not explicitly null, when unset.
    """
    out: dict = {}

    _set_if_present(out, "id", data.get("id"))
    _set_if_present(out, "object", data.get("object"))
    _set_if_present(out, "created", data.get("created"))
    _set_if_present(out, "model", data.get("model"))

    choices = data.get("choices")
    if choices is not None:
        out["choices"] = [_choice_from_flat(c) for c in choices]

    usage = data.get("usage")
    if usage is not None:
        out["usage"] = _usage_from_flat(usage)
    # _set_if_present(out, "usage", usage)
    _set_if_present(out, "system_fingerprint", data.get("system_fingerprint"))
    _set_if_present(out, "service_tier", data.get("service_tier"))

    return out  # type: ignore[return-value]
