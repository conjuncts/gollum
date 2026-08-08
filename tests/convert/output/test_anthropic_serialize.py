import copy
import json

from gollum.convert.output.anthropic import (
    _build_usage,
    _map_finish_reason,
    anthropic_message_to_completion,
)

from _data import anthropic_model_dump_json

# Ground truth: the expected GollumResponse for the real Anthropic payload in
# _data.py. `created` is non-deterministic (int(time.time())), so it's
# normalized to 0 on the actual result before comparison.
GROUND_TRUTH = {
    "response": {
        "id": "msg_011CdmpCmwqn9xtSKvKKSvci",
        "object": "chat.completion",
        "created": 0,
        "model": "claude-haiku-4-5-20251001",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris.",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 10,
            "total_tokens": 24,
        },
    },
    "extras": {
        "anthropic.text_block_layout": [
            {"content_index": 0, "type": "text", "length": 31}
        ],
        "anthropic.type": "message",
        "anthropic.role": "assistant",
        "anthropic.stop_reason": "end_turn",
        "anthropic.cache_creation_input_tokens": 0,
        "anthropic.cache_creation": {
            "ephemeral_1h_input_tokens": 0,
            "ephemeral_5m_input_tokens": 0,
        },
        "anthropic.stop_details": None,
        "anthropic.inference_geo": "not_available",
        "anthropic.output_tokens_details": None,
        "anthropic.service_tier": "standard",
    },
    "metadata": {},
}


def test_anthropic_message_to_completion_ground_truth():
    payload = json.loads(anthropic_model_dump_json)
    result = anthropic_message_to_completion(payload)

    actual = {
        "response": result.chat_completion,
        "extras": result.extras,
        "metadata": result.metadata,
    }
    actual["response"]["created"] = 0  # normalize non-deterministic field

    assert actual == GROUND_TRUTH

    # The converter must not mutate the caller's dict.
    assert payload == json.loads(anthropic_model_dump_json)


# ---------- finish-reason mapping ----------


def test_map_finish_reason_all_cases():
    cases = {
        None: None,
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "pause_turn": "stop",
        "refusal": "content_filter",
        "brand_new_reason": "stop",  # unknown reasons fall back to "stop"
    }
    for stop_reason, expected in cases.items():
        assert _map_finish_reason(stop_reason) == expected


# ---------- usage building ----------


def test_build_usage_empty_or_none():
    assert _build_usage(None) is None
    assert _build_usage({}) is None


def test_build_usage_cache_read_and_missing_token_defaults():
    # Missing input/output tokens default to 0; non-zero cache reads surface
    # as prompt_tokens_details.
    usage = _build_usage({"cache_read_input_tokens": 12})
    assert usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 12},
    }


# ---------- integration: minimal message ----------


def test_minimal_message_defaults():
    # No content, no id/model, no usage: every value falls back to its default.
    payload = {"content": None}
    result = anthropic_message_to_completion(payload)

    assert result.chat_completion["id"] == ""
    assert result.chat_completion["model"] == ""
    assert result.chat_completion["object"] == "chat.completion"
    assert result.chat_completion["choices"] == [
        {
            "index": 0,
            "finish_reason": None,
            "message": {"role": "assistant", "content": None},
        }
    ]
    assert "usage" not in result.chat_completion
    assert result.extras == {}
    assert result.metadata == {}


def test_stop_reason_max_tokens_maps_to_length():
    result = anthropic_message_to_completion(
        {
            "id": "m1",
            "model": "model-1",
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "partial"}],
        }
    )
    assert result.chat_completion["choices"][0]["finish_reason"] == "length"


# ---------- integration: every content-block type ----------

# Exercises: text + citations, thinking, redacted_thinking, tool_use,
# server_tool_use, web_search_tool_result, an unhandled block type, plus
# response-level extras (container, stop_sequence, usage extras).
COMPLEX_PAYLOAD = {
    "id": "msg_complex",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "container": {"id": "container_123"},
    "stop_reason": "tool_use",
    "stop_sequence": "\n",
    "content": [
        {
            "type": "text",
            "text": "Let me compute that.",
            "citations": [
                {"cited_text": "from source document", "document_index": 0}
            ],
        },
        {
            "type": "thinking",
            "thinking": "I need to multiply 6 by 7.",
            "signature": "sig_123",
        },
        {"type": "redacted_thinking", "data": "redacted-xyz"},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "calculator",
            "input": {"expr": "6*7"},
        },
        {
            "type": "server_tool_use",
            "id": "st_1",
            "name": "lookup",
            "input": {},
        },
        {"type": "web_search_tool_result", "id": "ws_1", "title": "Example"},
        {"type": "image", "source": {"type": "base64", "data": "abc"}},
    ],
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 10,
        "cache_creation": {"ephemeral_5m_input_tokens": 10},
        "server_tool_use": {"server_tool_use_input_tokens": 5},
    },
}

COMPLEX_GROUND_TRUTH = {
    "response": {
        "id": "msg_complex",
        "object": "chat.completion",
        "created": 0,  # normalized below
        "model": "claude-sonnet-4-5",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Let me compute that.",
                    "tool_calls": [
                        {
                            "id": "toolu_01",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expr": "6*7"}',
                            },
                        }
                    ],
                    "reasoning_content": "I need to multiply 6 by 7.",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
    },
    "extras": {
        "anthropic.thinking_blocks": [
            {"content_index": 1, "signature": "sig_123"}
        ],
        "anthropic.redacted_thinking_blocks": [
            {"content_index": 2, "data": "redacted-xyz"}
        ],
        "anthropic.citations": [
            {
                "content_index": 0,
                "citations": [
                    {"cited_text": "from source document", "document_index": 0}
                ],
            }
        ],
        "anthropic.server_tool_use_blocks": [
            {
                "content_index": 4,
                "block": {
                    "type": "server_tool_use",
                    "id": "st_1",
                    "name": "lookup",
                    "input": {},
                },
            }
        ],
        "anthropic.web_search_tool_result_blocks": [
            {
                "content_index": 5,
                "block": {
                    "type": "web_search_tool_result",
                    "id": "ws_1",
                    "title": "Example",
                },
            }
        ],
        "anthropic.unhandled_blocks": [
            {
                "content_index": 6,
                "block": {
                    "type": "image",
                    "source": {"type": "base64", "data": "abc"},
                },
            }
        ],
        "anthropic.text_block_layout": [
            {"content_index": 0, "type": "text", "length": 20}
        ],
        "anthropic.thinking_block_layout": [
            {"content_index": 1, "type": "thinking", "length": 26}
        ],
        "anthropic.type": "message",
        "anthropic.role": "assistant",
        "anthropic.stop_reason": "tool_use",
        "anthropic.stop_sequence": "\n",
        "anthropic.container": {"id": "container_123"},
        "anthropic.cache_creation_input_tokens": 10,
        "anthropic.cache_creation": {"ephemeral_5m_input_tokens": 10},
        "anthropic.server_tool_use_usage": {"server_tool_use_input_tokens": 5},
    },
    "metadata": {},
}


def test_complex_content_blocks():
    payload = copy.deepcopy(COMPLEX_PAYLOAD)
    result = anthropic_message_to_completion(payload)

    actual = {
        "response": result.chat_completion,
        "extras": result.extras,
        "metadata": result.metadata,
    }
    actual["response"]["created"] = 0  # normalize non-deterministic field

    assert actual == COMPLEX_GROUND_TRUTH

    # The converter must not mutate the caller's dict.
    assert payload == COMPLEX_PAYLOAD
