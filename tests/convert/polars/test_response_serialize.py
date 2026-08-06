import pytest

from gollum.convert.polars.serialize_response import (
    pl_serialize_chat_response,
    pl_deserialize_chat_response,
)
from gollum.types.chat_completions import ChatCompletionResponseModel

sample_responses: list[ChatCompletionResponseModel] = [
    # --- 0. Minimal response --------------------------------------------
    {
        "id": "chatcmpl-0",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
    },

    # --- 1. Usage with token details -------------------------------------
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "42."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {
                "reasoning_tokens": 3,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "cached_tokens": 2,
                "audio_tokens": 0,
            },
        },
    },

    # --- 2. Assistant message with tool_calls -----------------------------
    {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1700000002,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    },

    # --- 3. Logprobs on content and refusal --------------------------------
    {
        "id": "chatcmpl-3",
        "object": "chat.completion",
        "created": 1700000003,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Hi",
                            "logprob": -0.1,
                            "bytes": [72, 105],
                            "top_logprobs": [
                                {"token": "Hi", "logprob": -0.1, "bytes": [72, 105]},
                                {"token": "Hello", "logprob": -2.3, "bytes": [72, 101, 108, 108, 111]},
                            ],
                        }
                    ],
                    "refusal": None,
                },
                "finish_reason": "stop",
            }
        ],
    },

    # --- 4. Multiple choices (n > 1) ---------------------------------------
    {
        "id": "chatcmpl-4",
        "object": "chat.completion",
        "created": 1700000004,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Option A"}, "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": "Option B"}, "finish_reason": "length"},
        ],
    },

    # --- 5. Refusal ----------------------------------------------------------
    {
        "id": "chatcmpl-5",
        "object": "chat.completion",
        "created": 1700000005,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "refusal": "I can't help with that."},
                "finish_reason": "content_filter",
            }
        ],
    },

    # --- 6. system_fingerprint + service_tier --------------------------------
    {
        "id": "chatcmpl-6",
        "object": "chat.completion",
        "created": 1700000006,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}
        ],
        "system_fingerprint": "fp_abc123",
        "service_tier": "default",
    },

    # --- 7. Kitchen sink: everything at once ----------------------------------
    {
        "id": "chatcmpl-7",
        "object": "chat.completion",
        "created": 1700000007,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {"name": "save_result", "arguments": '{"filename": "out.txt"}'},
                        }
                    ],
                },
                "logprobs": {
                    "content": [
                        {
                            "token": "ok",
                            "logprob": -0.05,
                            "bytes": [111, 107],
                            "top_logprobs": [{"token": "ok", "logprob": -0.05, "bytes": [111, 107]}],
                        }
                    ],
                    "refusal": None,
                },
                "finish_reason": "tool_calls",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "Alternative answer."},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 12,
            "total_tokens": 62,
            "completion_tokens_details": {
                "reasoning_tokens": 4,
                "accepted_prediction_tokens": 1,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"cached_tokens": 10, "audio_tokens": 0},
        },
        "system_fingerprint": "fp_kitchen_sink",
        "service_tier": "scale",
    },
]


@pytest.mark.parametrize("sample_index", range(len(sample_responses)))
def test_sample(sample_index):
    orig = sample_responses[sample_index]
    as_polars = pl_serialize_chat_response(orig)
    as_original = pl_deserialize_chat_response(as_polars)
    assert as_original == orig


@pytest.mark.parametrize("sample_index", range(len(sample_responses)))
def test_sample_is_polars_serializable(sample_index):
    import polars as pl
    from gollum.types.pl_chat_completions import ChatCompletionResponseSchema
    """Confirm the flattened dict actually loads into a Polars DataFrame
    under ChatCompletionResponseSchema (i.e. the struct shapes genuinely
    line up, not just that the Python round trip is lossless)."""
    orig = sample_responses[sample_index]
    as_polars = pl_serialize_chat_response(orig)
    df = pl.DataFrame([as_polars], schema=ChatCompletionResponseSchema)
    assert df.height == 1

    # And back out through the DataFrame round trip too.
    row = df.to_dicts()[0]
    as_original = pl_deserialize_chat_response(row)
    assert as_original == orig