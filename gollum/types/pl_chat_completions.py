import polars as pl

from gollum.types.chat_completions import ChatCompletionRequest

# ---------- Reusable nested structs ----------

text_content_part = pl.Struct({
    "type": pl.Utf8,
    "text": pl.Utf8,
})

image_content_part = pl.Struct({
    "type": pl.Utf8,
    "image_url": pl.Struct({
        "url": pl.Utf8,
        "detail": pl.Utf8,  # "auto" | "low" | "high"
    }),
})

# Polars has no true union type, so content parts are modeled as a single
# struct with all possible fields present (nulled when not applicable).
content_part = pl.Struct({
    "type": pl.Utf8,       # "text" | "image_url"
    "text": pl.Utf8,
    "image_url": pl.Struct({
        "url": pl.Utf8,
        "detail": pl.Utf8,
    }),
})

tool_call_function = pl.Struct({
    "name": pl.Utf8,
    "arguments": pl.Utf8,  # JSON-encoded string
})

tool_call = pl.Struct({
    "id": pl.Utf8,
    "type": pl.Utf8,       # "function"
    "function": tool_call_function,
})

function_call = pl.Struct({
    "name": pl.Utf8,
    "arguments": pl.Utf8,
})

# ---------- Message (all roles flattened into one struct) ----------

message = pl.Struct({
    "role": pl.Utf8,                       # "system"|"user"|"assistant"|"tool"|"function"
    "content": pl.Utf8,                    # or use content_parts below for multimodal
    "content_parts": pl.List(content_part),  # populated when content is a list, not a string
    "name": pl.Utf8,
    "tool_calls": pl.List(tool_call),
    "tool_call_id": pl.Utf8,
    "function_call": function_call,        # deprecated
    "refusal": pl.Utf8,
})

# ---------- Tools / function definitions ----------

function_definition = pl.Struct({
    "name": pl.Utf8,
    "description": pl.Utf8,
    "parameters": pl.Utf8,   # JSON Schema, kept as a JSON string (Struct can't hold arbitrary schema)
    "strict": pl.Boolean,
})

tool = pl.Struct({
    "type": pl.Utf8,         # "function"
    "function": function_definition,
})

named_tool_choice = pl.Struct({
    "type": pl.Utf8,
    "function": pl.Struct({"name": pl.Utf8}),
})

# tool_choice itself is polymorphic (str | struct) — store separately if needed:
# tool_choice_str: pl.Utf8   ("none" | "auto" | "required")
# tool_choice_named: named_tool_choice

response_format = pl.Struct({
    "type": pl.Utf8,          # "text" | "json_object" | "json_schema"
    "json_schema": pl.Struct({
        "name": pl.Utf8,
        "description": pl.Utf8,
        "schema": pl.Utf8,    # JSON string
        "strict": pl.Boolean,
    }),
})

stream_options = pl.Struct({
    "include_usage": pl.Boolean,
})

# ---------- Audio ----------

# voice is polymorphic (built-in name | {"id": str}) — split like tool_choice
audio_voice_id = pl.Struct({
    "id": pl.Utf8,
})

audio = pl.Struct({
    "format": pl.Utf8,        # "wav"|"aac"|"mp3"|"flac"|"opus"|"pcm16"
    "voice_str": pl.Utf8,     # built-in voice name
    "voice_id": audio_voice_id,
})

# ---------- Metadata / dicts ----------

# OpenAI metadata is Dict[str, str] — store as list of entries, like logit_bias
metadata_entry = pl.Struct({
    "key": pl.Utf8,
    "value": pl.Utf8,
})

# ---------- Function call (deprecated) ----------

function_call_option = pl.Struct({
    "name": pl.Utf8,
})

# function_call is polymorphic ("none"|"auto" | {"name": str}) — split like tool_choice:
# function_call_str: pl.Utf8
# function_call_named: function_call_option

# ---------- Prediction ----------

prediction = pl.Struct({
    "type": pl.Utf8,          # "content"
    "content": pl.Utf8,       # or use content_parts below when content is a list
    "content_parts": pl.List(pl.Struct({
        "type": pl.Utf8,      # "text"
        "text": pl.Utf8,
    })),
})

# ---------- Moderation ----------

moderation = pl.Struct({
    "model": pl.Utf8,
    "policy_input_mode": pl.Utf8,    # "score" | "block"
    "policy_output_mode": pl.Utf8,   # "score" | "block"
})

# ---------- Prompt cache ----------

prompt_cache_options = pl.Struct({
    "mode": pl.Utf8,          # "implicit" | "explicit"
    "ttl": pl.Utf8,           # "30m"
})

# ---------- Web search ----------

web_search_options = pl.Struct({
    "search_context_size": pl.Utf8,   # "low" | "medium" | "high"
    "user_location_type": pl.Utf8,    # "approximate"
    "city": pl.Utf8,
    "country": pl.Utf8,
    "region": pl.Utf8,
    "timezone": pl.Utf8,
})

# ---------- ChatCompletionRequest schema ----------

ChatCompletionRequestSchema = pl.Schema({
    # required fields first, then alphabetical — matches OpenAI
    "messages": pl.List(message),
    "model": pl.Utf8,
    "audio": audio,
    "frequency_penalty": pl.Float64,
    "function_call_str": pl.Utf8,              # "none" | "auto" (deprecated)
    "function_call_named": function_call_option,
    "functions": pl.List(function_definition),  # deprecated
    "logit_bias": pl.List(pl.Struct({"token_id": pl.Utf8, "bias": pl.Int64})),
    "logprobs": pl.Boolean,
    "max_completion_tokens": pl.Int64,
    "max_tokens": pl.Int64,
    "metadata": pl.List(metadata_entry),        # Dict[str, str] → list of entries
    "modalities": pl.List(pl.Utf8),             # "text" | "audio"
    "moderation": moderation,
    "n": pl.Int64,
    "parallel_tool_calls": pl.Boolean,
    "prediction": prediction,
    "presence_penalty": pl.Float64,
    "prompt_cache_key": pl.Utf8,
    "prompt_cache_options": prompt_cache_options,
    "prompt_cache_retention": pl.Utf8,          # "in_memory" | "24h" (deprecated)
    "reasoning_effort": pl.Utf8,                # "none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"max"
    "response_format": response_format,
    "safety_identifier": pl.Utf8,
    "seed": pl.Int64,
    "service_tier": pl.Utf8,
    "stop": pl.List(pl.Utf8),                   # normalize str|list[str] into a list
    "store": pl.Boolean,
    "stream": pl.Boolean,
    "stream_options": stream_options,
    "temperature": pl.Float64,
    "tool_choice_str": pl.Utf8,                 # "none" | "auto" | "required"
    "tool_choice_named": named_tool_choice,
    "tools": pl.List(tool),
    "top_logprobs": pl.Int64,
    "top_p": pl.Float64,
    "user": pl.Utf8,
    "verbosity": pl.Utf8,                       # "low" | "medium" | "high"
    "web_search_options": web_search_options,
})

# ---------- ChatCompletionResponse schema ----------

top_logprob = pl.Struct({
    "token": pl.Utf8,
    "logprob": pl.Float64,
    "bytes": pl.List(pl.Int64),
})

logprob_content = pl.Struct({
    "token": pl.Utf8,
    "logprob": pl.Float64,
    "bytes": pl.List(pl.Int64),
    "top_logprobs": pl.List(top_logprob),
})

logprobs_struct = pl.Struct({
    "content": pl.List(logprob_content),
    "refusal": pl.List(logprob_content),
})

choice = pl.Struct({
    "index": pl.Int64,
    "message": message,
    "logprobs": logprobs_struct,
    "finish_reason": pl.Utf8,  # "stop"|"length"|"tool_calls"|"content_filter"|"function_call"
})

completion_tokens_details = pl.Struct({
    "reasoning_tokens": pl.Int64,
    "accepted_prediction_tokens": pl.Int64,
    "rejected_prediction_tokens": pl.Int64,
})

prompt_tokens_details = pl.Struct({
    "cached_tokens": pl.Int64,
    "audio_tokens": pl.Int64,
})

usage = pl.Struct({
    "prompt_tokens": pl.Int64,
    "completion_tokens": pl.Int64,
    "total_tokens": pl.Int64,
    "completion_tokens_details": completion_tokens_details,
    "prompt_tokens_details": prompt_tokens_details,
})

ChatCompletionResponseSchema = pl.Schema({
    "id": pl.Utf8,
    "object": pl.Utf8,          # "chat.completion"
    "created": pl.Int64,
    "model": pl.Utf8,
    "choices": pl.List(choice),
    "usage": usage,
    "system_fingerprint": pl.Utf8,
    "service_tier": pl.Utf8,
})

# ---------- Usage example ----------

df_requests = pl.DataFrame(schema=ChatCompletionRequestSchema)
df_responses = pl.DataFrame(schema=ChatCompletionResponseSchema)

# Loading real JSON payloads:
# df_responses = pl.read_json("responses.json", schema=ChatCompletionResponseSchema)
# or for one dict:
# df = pl.DataFrame([response_dict], schema=ChatCompletionResponseSchema)

def completion_to_pl_serializable(data: ChatCompletionRequest) -> dict:
    """
    Converts a ChatCompletionRequest to a dictionary that is serializable by Polars.
    """

def pl_serializable_to_completion(data: dict) -> ChatCompletionRequest:
    """
    Converts a dictionary to a ChatCompletionRequest.
    """