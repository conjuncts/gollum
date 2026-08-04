

# ============================================================
# Round-trip helpers
#
# Every helper below handles one spot where the OpenAI shape is
# polymorphic (str | dict, dict | list, or nested JSON-schema-shaped
# dict) and Polars needs a single fixed struct. Each has a `_to_flat`
# (OpenAI -> Polars-serializable) and `_from_flat` (Polars -> OpenAI)
# counterpart so the split stays symmetric and easy to audit.
# ============================================================

import json

from gollum.types.chat_completions import ChatCompletionRequest


def _set_if_present(out: dict, key: str, value) -> None:
    """Only add `key` to `out` when `value` is not None/empty, so we don't
    reintroduce nulled-out optional fields OpenAI never expects to see."""
    if value is not None:
        out[key] = value


# ---------- content: str | list[content part] ----------

def _content_part_to_flat(part: dict) -> dict:
    ptype = part.get("type")
    flat = {"type": ptype, "text": part.get("text"), "image_url": None}
    if ptype == "image_url":
        image_url = part.get("image_url") or {}
        flat["image_url"] = {
            "url": image_url.get("url"),
            "detail": image_url.get("detail"),
        }
    return flat


def _content_part_from_flat(part: dict) -> dict:
    ptype = part.get("type")
    if ptype == "image_url":
        image_url = part.get("image_url") or {}
        out = {"type": "image_url", "image_url": {}}
        _set_if_present(out["image_url"], "url", image_url.get("url"))
        _set_if_present(out["image_url"], "detail", image_url.get("detail"))
        return out
    # default / "text"
    out = {"type": ptype or "text"}
    _set_if_present(out, "text", part.get("text"))
    return out


def _split_content(content):
    """content: str | list[dict] | None -> (content_str, content_parts)"""
    if content is None:
        return None, None
    if isinstance(content, str):
        return content, None
    if isinstance(content, list):
        return None, [_content_part_to_flat(p) for p in content]
    return None, None


def _join_content(content_str, content_parts):
    if content_parts:
        return [_content_part_from_flat(p) for p in content_parts]
    return content_str


# ---------- messages ----------

def _message_to_flat(msg: dict) -> dict:
    content_str, content_parts = _split_content(msg.get("content"))
    fc = msg.get("function_call")
    flat_fc = None
    if fc:
        flat_fc = {"name": fc.get("name"), "arguments": fc.get("arguments")}
    return {
        "role": msg.get("role"),
        "content": content_str,
        "content_parts": content_parts,
        "name": msg.get("name"),
        "tool_calls": msg.get("tool_calls"),  # already {id, type, function{name,arguments}}
        "tool_call_id": msg.get("tool_call_id"),
        "function_call": flat_fc,
        "refusal": msg.get("refusal"),
    }


def _message_from_flat(msg: dict) -> dict:
    out = {"role": msg.get("role")}
    content = _join_content(msg.get("content"), msg.get("content_parts"))
    _set_if_present(out, "content", content)
    _set_if_present(out, "name", msg.get("name"))
    _set_if_present(out, "tool_calls", msg.get("tool_calls") or None)
    _set_if_present(out, "tool_call_id", msg.get("tool_call_id"))
    fc = msg.get("function_call")
    if fc:
        out["function_call"] = {"name": fc.get("name"), "arguments": fc.get("arguments")}
    _set_if_present(out, "refusal", msg.get("refusal"))
    return out


# ---------- tools / functions (parameters is arbitrary JSON) ----------

def _tool_to_flat(t: dict) -> dict:
    fn = t.get("function") or {}
    params = fn.get("parameters")
    return {
        "type": t.get("type", "function"),
        "function": {
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": json.dumps(params) if params is not None else None,
            "strict": fn.get("strict"),
        },
    }


def _tool_from_flat(t: dict) -> dict:
    fn = t.get("function") or {}
    out_fn = {"name": fn.get("name")}
    _set_if_present(out_fn, "description", fn.get("description"))
    if fn.get("parameters"):
        out_fn["parameters"] = json.loads(fn["parameters"])
    _set_if_present(out_fn, "strict", fn.get("strict"))
    return {"type": t.get("type", "function"), "function": out_fn}


def _function_definition_to_flat(fn: dict) -> dict:
    params = fn.get("parameters")
    return {
        "name": fn.get("name"),
        "description": fn.get("description"),
        "parameters": json.dumps(params) if params is not None else None,
        "strict": fn.get("strict"),
    }


def _function_definition_from_flat(fn: dict) -> dict:
    out = {"name": fn.get("name")}
    _set_if_present(out, "description", fn.get("description"))
    if fn.get("parameters"):
        out["parameters"] = json.loads(fn["parameters"])
    _set_if_present(out, "strict", fn.get("strict"))
    return out


# ---------- tool_choice: "none"|"auto"|"required" | {"type","function":{"name"}} ----------

def _tool_choice_split(tool_choice):
    if tool_choice is None:
        return None, None
    if isinstance(tool_choice, str):
        return tool_choice, None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        return None, {"type": tool_choice.get("type", "function"), "function": {"name": fn.get("name")}}
    return None, None


def _tool_choice_join(tool_choice_str, tool_choice_named):
    if tool_choice_named:
        fn = tool_choice_named.get("function") or {}
        return {"type": tool_choice_named.get("type", "function"), "function": {"name": fn.get("name")}}
    return tool_choice_str


# ---------- function_call (deprecated): "none"|"auto" | {"name"} ----------

def _function_call_split(fc):
    if fc is None:
        return None, None
    if isinstance(fc, str):
        return fc, None
    if isinstance(fc, dict):
        return None, {"name": fc.get("name")}
    return None, None


def _function_call_join(fc_str, fc_named):
    if fc_named:
        return {"name": fc_named.get("name")}
    return fc_str


# ---------- logit_bias: Dict[str, int] <-> list of entries ----------

def _logit_bias_to_list(logit_bias):
    if not logit_bias:
        return None
    return [{"token_id": str(k), "bias": v} for k, v in logit_bias.items()]


def _logit_bias_from_list(entries):
    if not entries:
        return None
    return {e["token_id"]: e["bias"] for e in entries}


# ---------- metadata: Dict[str, str] <-> list of entries ----------

def _metadata_to_list(metadata):
    if not metadata:
        return None
    return [{"key": k, "value": v} for k, v in metadata.items()]


def _metadata_from_list(entries):
    if not entries:
        return None
    return {e["key"]: e["value"] for e in entries}


# ---------- stop: str | list[str] | None <-> list[str] ----------

def _stop_to_list(stop):
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return stop
    return None


def _stop_from_list(stop_list):
    if not stop_list:
        return None
    return stop_list[0] if len(stop_list) == 1 else stop_list


# ---------- audio.voice: str | {"id": str} ----------

def _audio_to_flat(a):
    if not a:
        return None
    voice = a.get("voice")
    voice_str, voice_id = None, None
    if isinstance(voice, str):
        voice_str = voice
    elif isinstance(voice, dict):
        voice_id = {"id": voice.get("id")}
    return {"format": a.get("format"), "voice_str": voice_str, "voice_id": voice_id}


def _audio_from_flat(a):
    if not a:
        return None
    out = {}
    _set_if_present(out, "format", a.get("format"))
    if a.get("voice_id"):
        out["voice"] = {"id": a["voice_id"].get("id")}
    elif a.get("voice_str") is not None:
        out["voice"] = a["voice_str"]
    return out or None


# ---------- prediction.content: str | list[content part] ----------

def _prediction_to_flat(pred):
    if not pred:
        return None
    content_str, content_parts = _split_content(pred.get("content"))
    return {"type": pred.get("type", "content"), "content": content_str, "content_parts": content_parts}


def _prediction_from_flat(pred):
    if not pred:
        return None
    out = {"type": pred.get("type", "content")}
    content = _join_content(pred.get("content"), pred.get("content_parts"))
    _set_if_present(out, "content", content)
    return out


# ---------- response_format.json_schema.schema: arbitrary JSON ----------

def _response_format_to_flat(rf):
    if not rf:
        return None
    js = rf.get("json_schema")
    flat_js = None
    if js:
        schema = js.get("schema")
        flat_js = {
            "name": js.get("name"),
            "description": js.get("description"),
            "schema": json.dumps(schema) if schema is not None else None,
            "strict": js.get("strict"),
        }
    return {"type": rf.get("type"), "json_schema": flat_js}


def _response_format_from_flat(rf):
    if not rf:
        return None
    out = {"type": rf.get("type")}
    js = rf.get("json_schema")
    if js:
        out_js = {}
        _set_if_present(out_js, "name", js.get("name"))
        _set_if_present(out_js, "description", js.get("description"))
        if js.get("schema"):
            out_js["schema"] = json.loads(js["schema"])
        _set_if_present(out_js, "strict", js.get("strict"))
        out["json_schema"] = out_js
    return out


# ---------- web_search_options: nested user_location.approximate ----------

def _web_search_options_to_flat(wso):
    if not wso:
        return None
    loc = wso.get("user_location") or {}
    approx = loc.get("approximate") or {}
    return {
        "search_context_size": wso.get("search_context_size"),
        "user_location_type": loc.get("type"),
        "city": approx.get("city"),
        "country": approx.get("country"),
        "region": approx.get("region"),
        "timezone": approx.get("timezone"),
    }


def _web_search_options_from_flat(wso):
    if not wso:
        return None
    out = {}
    _set_if_present(out, "search_context_size", wso.get("search_context_size"))
    approx = {
        k: wso.get(k) for k in ("city", "country", "region", "timezone") if wso.get(k) is not None
    }
    if approx or wso.get("user_location_type"):
        out["user_location"] = {
            "type": wso.get("user_location_type") or "approximate",
            "approximate": approx,
        }
    return out or None


# ============================================================
# Public round-trip API
# ============================================================

def completion_to_pl_serializable(data: ChatCompletionRequest) -> dict:
    """
    Converts a ChatCompletionRequest to a dictionary that is serializable by Polars,
    i.e. one whose shape matches ChatCompletionRequestSchema exactly (every
    polymorphic OpenAI field split into its fixed-shape counterparts, and every
    arbitrary-JSON field — tool/function parameters, response_format json schema —
    encoded as a JSON string).

    Missing/absent OpenAI fields become None, which Polars renders as null when the
    dict is fed into `pl.DataFrame([...], schema=ChatCompletionRequestSchema)`.
    """
    data = dict(data)  # TypedDict is just a dict at runtime; avoid mutating caller's copy

    tool_choice_str, tool_choice_named = _tool_choice_split(data.get("tool_choice"))
    function_call_str, function_call_named = _function_call_split(data.get("function_call"))

    messages = data.get("messages")
    functions = data.get("functions")
    tools = data.get("tools")

    return {
        "messages": [_message_to_flat(m) for m in messages] if messages else None,
        "model": data.get("model"),
        "audio": _audio_to_flat(data.get("audio")),
        "frequency_penalty": data.get("frequency_penalty"),
        "function_call_str": function_call_str,
        "function_call_named": function_call_named,
        "functions": [_function_definition_to_flat(f) for f in functions] if functions else None,
        "logit_bias": _logit_bias_to_list(data.get("logit_bias")),
        "logprobs": data.get("logprobs"),
        "max_completion_tokens": data.get("max_completion_tokens"),
        "max_tokens": data.get("max_tokens"),
        "metadata": _metadata_to_list(data.get("metadata")),
        "modalities": data.get("modalities"),
        "moderation": data.get("moderation"),
        "n": data.get("n"),
        "parallel_tool_calls": data.get("parallel_tool_calls"),
        "prediction": _prediction_to_flat(data.get("prediction")),
        "presence_penalty": data.get("presence_penalty"),
        "prompt_cache_key": data.get("prompt_cache_key"),
        "prompt_cache_options": data.get("prompt_cache_options"),
        "prompt_cache_retention": data.get("prompt_cache_retention"),
        "reasoning_effort": data.get("reasoning_effort"),
        "response_format": _response_format_to_flat(data.get("response_format")),
        "safety_identifier": data.get("safety_identifier"),
        "seed": data.get("seed"),
        "service_tier": data.get("service_tier"),
        "stop": _stop_to_list(data.get("stop")),
        "store": data.get("store"),
        "stream": data.get("stream"),
        "stream_options": data.get("stream_options"),
        "temperature": data.get("temperature"),
        "tool_choice_str": tool_choice_str,
        "tool_choice_named": tool_choice_named,
        "tools": [_tool_to_flat(t) for t in tools] if tools else None,
        "top_logprobs": data.get("top_logprobs"),
        "top_p": data.get("top_p"),
        "user": data.get("user"),
        "verbosity": data.get("verbosity"),
        "web_search_options": _web_search_options_to_flat(data.get("web_search_options")),
    }


def pl_serializable_to_completion(data: dict) -> ChatCompletionRequest:
    """
    Converts a flattened dict (matching ChatCompletionRequestSchema, e.g. one row
    of `df_requests.to_dicts()`) back into an OpenAI-shaped ChatCompletionRequest
    dict. Null/None fields are dropped rather than included as `None`, since
    OpenAI's TypedDict fields are meant to be absent, not explicitly null, when
    unset.
    """
    out: dict = {}

    messages = data.get("messages")
    if messages:
        out["messages"] = [_message_from_flat(m) for m in messages]
    _set_if_present(out, "model", data.get("model"))
    _set_if_present(out, "audio", _audio_from_flat(data.get("audio")))
    _set_if_present(out, "frequency_penalty", data.get("frequency_penalty"))

    function_call = _function_call_join(data.get("function_call_str"), data.get("function_call_named"))
    _set_if_present(out, "function_call", function_call)

    functions = data.get("functions")
    if functions:
        out["functions"] = [_function_definition_from_flat(f) for f in functions]

    _set_if_present(out, "logit_bias", _logit_bias_from_list(data.get("logit_bias")))
    _set_if_present(out, "logprobs", data.get("logprobs"))
    _set_if_present(out, "max_completion_tokens", data.get("max_completion_tokens"))
    _set_if_present(out, "max_tokens", data.get("max_tokens"))
    _set_if_present(out, "metadata", _metadata_from_list(data.get("metadata")))
    _set_if_present(out, "modalities", data.get("modalities"))
    _set_if_present(out, "moderation", data.get("moderation"))
    _set_if_present(out, "n", data.get("n"))
    _set_if_present(out, "parallel_tool_calls", data.get("parallel_tool_calls"))
    _set_if_present(out, "prediction", _prediction_from_flat(data.get("prediction")))
    _set_if_present(out, "presence_penalty", data.get("presence_penalty"))
    _set_if_present(out, "prompt_cache_key", data.get("prompt_cache_key"))
    _set_if_present(out, "prompt_cache_options", data.get("prompt_cache_options"))
    _set_if_present(out, "prompt_cache_retention", data.get("prompt_cache_retention"))
    _set_if_present(out, "reasoning_effort", data.get("reasoning_effort"))
    _set_if_present(out, "response_format", _response_format_from_flat(data.get("response_format")))
    _set_if_present(out, "safety_identifier", data.get("safety_identifier"))
    _set_if_present(out, "seed", data.get("seed"))
    _set_if_present(out, "service_tier", data.get("service_tier"))
    _set_if_present(out, "stop", _stop_from_list(data.get("stop")))
    _set_if_present(out, "store", data.get("store"))
    _set_if_present(out, "stream", data.get("stream"))
    _set_if_present(out, "stream_options", data.get("stream_options"))
    _set_if_present(out, "temperature", data.get("temperature"))

    tool_choice = _tool_choice_join(data.get("tool_choice_str"), data.get("tool_choice_named"))
    _set_if_present(out, "tool_choice", tool_choice)

    tools = data.get("tools")
    if tools:
        out["tools"] = [_tool_from_flat(t) for t in tools]

    _set_if_present(out, "top_logprobs", data.get("top_logprobs"))
    _set_if_present(out, "top_p", data.get("top_p"))
    _set_if_present(out, "user", data.get("user"))
    _set_if_present(out, "verbosity", data.get("verbosity"))
    _set_if_present(out, "web_search_options", _web_search_options_from_flat(data.get("web_search_options")))

    return out  # type: ignore[return-value]
