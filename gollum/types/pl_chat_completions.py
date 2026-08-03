import polars as pl

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

# ---------- ChatCompletionRequest schema ----------

ChatCompletionRequestSchema = pl.Schema({
    "model": pl.Utf8,
    "messages": pl.List(message),
    "frequency_penalty": pl.Float64,
    "logit_bias": pl.List(pl.Struct({"token_id": pl.Utf8, "bias": pl.Int64})),
    "logprobs": pl.Boolean,
    "top_logprobs": pl.Int64,
    "max_tokens": pl.Int64,
    "max_completion_tokens": pl.Int64,
    "n": pl.Int64,
    "presence_penalty": pl.Float64,
    "response_format": response_format,
    "seed": pl.Int64,
    "service_tier": pl.Utf8,
    "stop": pl.List(pl.Utf8),         # normalize str|list[str] into a list
    "stream": pl.Boolean,
    "stream_options": stream_options,
    "temperature": pl.Float64,
    "top_p": pl.Float64,
    "tools": pl.List(tool),
    "tool_choice_str": pl.Utf8,       # "none" | "auto" | "required"
    "tool_choice_named": named_tool_choice,
    "parallel_tool_calls": pl.Boolean,
    "user": pl.Utf8,
    "functions": pl.List(function_definition),  # deprecated
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