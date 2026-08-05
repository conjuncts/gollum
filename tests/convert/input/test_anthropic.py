import pytest

from gollum.convert.input.anthropic import to_anthropic_request
from gollum.types.chat_completions import ChatCompletionRequest


def test_minimal_request():
    req: ChatCompletionRequest = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
    }
    assert to_anthropic_request(req) == {
        "model": "claude-3-5-sonnet",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    }


def test_max_completion_tokens_preferred_over_max_tokens():
    req: ChatCompletionRequest = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 200,
        "max_tokens": 50,
    }
    assert to_anthropic_request(req)["max_tokens"] == 200


def test_missing_max_tokens_raises():
    req: ChatCompletionRequest = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    with pytest.raises(ValueError, match="max_tokens"):
        to_anthropic_request(req)


def test_system_message_hoisted_to_system_field():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
    }
    out = to_anthropic_request(req)
    assert out["system"] == "Be brief."
    assert out["messages"] == [{"role": "user", "content": "Hello"}]


def test_multiple_system_messages_concatenated():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Answer in French."},
        ],
    }
    out = to_anthropic_request(req)
    assert out["system"] == [
        {"type": "text", "text": "Be brief."},
        {"type": "text", "text": "Answer in French."},
    ]
    assert out["messages"] == [{"role": "user", "content": "Hello"}]


def test_stop_normalized_to_list():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "stop": "END",
    }
    assert to_anthropic_request(req)["stop_sequences"] == ["END"]


def test_tools_and_tool_choice_converted():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    out = to_anthropic_request(req)
    assert out["tools"] == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            "type": "custom",
        }
    ]
    assert out["tool_choice"] == {"type": "tool", "name": "get_weather"}


def test_tool_choice_strings():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": "required",
    }
    assert to_anthropic_request(req)["tool_choice"] == {"type": "any"}


def test_tool_calls_and_results_converted():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
        ],
    }
    out = to_anthropic_request(req)
    assert out["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "SF"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": '{"temp": 72}'}
            ],
        },
    ]


def test_image_data_uri_converted():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ],
    }
    out = to_anthropic_request(req)
    assert out["messages"][0]["content"] == [
        {"type": "text", "text": "What is this?"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
        },
    ]


def test_image_plain_url_converted():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    }
                ],
            }
        ],
    }
    out = to_anthropic_request(req)
    assert out["messages"][0]["content"] == [
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/cat.png"},
        }
    ]


def test_metadata_user_id_mapped_and_extra_keys_dropped():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "user": "u_123",
        "metadata": {"user_id": "u_456", "session": "s1"},
    }
    out = to_anthropic_request(req)
    assert out["metadata"] == {"user_id": "u_456"}


def test_thinking_and_top_k_forwarded():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
    }
    out = to_anthropic_request(
        req,
        thinking={"type": "enabled", "budget_tokens": 2048},
        top_k=40,
    )
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert out["top_k"] == 40


def test_audio_part_raises_not_implemented():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}
        ],
    }
    with pytest.raises(NotImplementedError, match="input_audio"):
        to_anthropic_request(req)
