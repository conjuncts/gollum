import pytest

from gollum.convert.input.anthropic import from_anthropic_request, to_anthropic_request
from gollum.types.chat_completions import ChatCompletionRequest


def test_minimal_request():
    req: ChatCompletionRequest = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
    }
    assert to_anthropic_request(req, {}) == {
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
    assert to_anthropic_request(req, {})["max_tokens"] == 200


def test_missing_max_tokens_default_4096():
    req: ChatCompletionRequest = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert to_anthropic_request(req, {})["max_tokens"] == 4096


def test_system_message_hoisted_to_system_field():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
    }
    out = to_anthropic_request(req, {})
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
    out = to_anthropic_request(req, {})
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
    assert to_anthropic_request(req, {})["stop_sequences"] == ["END"]


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
    out = to_anthropic_request(req, {})
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
    assert to_anthropic_request(req, {})["tool_choice"] == {"type": "any"}


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
    out = to_anthropic_request(req, {})
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
    out = to_anthropic_request(req, {})
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
    out = to_anthropic_request(req, {})
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
    out = to_anthropic_request(req, {})
    assert out["metadata"] == {"user_id": "u_456"}


def test_thinking_and_top_k_forwarded():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
    }
    out = to_anthropic_request(
        req,
        extras={
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "top_k": 40,
        }
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
        to_anthropic_request(req, {})


def test_temperature_top_p_stream_forwarded():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    }
    out = to_anthropic_request(req, {})
    assert out["temperature"] == 0.7
    assert out["top_p"] == 0.9
    assert out["stream"] is True


def test_stop_list_passthrough():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "stop": ["END", "STOP"],
    }
    assert to_anthropic_request(req, {})["stop_sequences"] == ["END", "STOP"]


def test_no_stop_omits_stop_sequences():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
    }
    assert "stop_sequences" not in to_anthropic_request(req, {})


def test_service_tier_forwarded():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
    }
    out = to_anthropic_request(req, extras={"service_tier": "priority"})
    assert out["service_tier"] == "priority"


def test_metadata_omitted_when_no_user_info():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
    }
    assert "metadata" not in to_anthropic_request(req, {})


def test_metadata_user_only_from_oai_user_field():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "user": "u_123",
    }
    out = to_anthropic_request(req, {})
    assert out["metadata"] == {"user_id": "u_123"}


def test_tool_choice_auto():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": "auto",
    }
    assert to_anthropic_request(req, {})["tool_choice"] == {"type": "auto"}


def test_tool_choice_none():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": "none",
    }
    assert to_anthropic_request(req, {})["tool_choice"] == {"type": "none"}


def test_tool_choice_unknown_string_raises():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": "bogus",
    }
    with pytest.raises(ValueError, match="bogus"):
        to_anthropic_request(req, {})


def test_tool_choice_function_without_name_becomes_any():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": {"type": "function", "function": {}},
    }
    assert to_anthropic_request(req, {})["tool_choice"] == {"type": "any"}


def test_tool_choice_parallel_tool_calls_false():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": {
            "type": "function",
            "function": {"name": "f"},
            "parallel_tool_calls": False,
        },
    }
    out = to_anthropic_request(req, {})
    assert out["tool_choice"] == {
        "type": "tool",
        "name": "f",
        "disable_parallel_tool_use": True,
    }


def test_tool_choice_unsupported_dict_type_raises():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": {"type": "bogus"},
    }
    with pytest.raises(ValueError):
        to_anthropic_request(req, {})


def test_tool_choice_unsupported_type_raises():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tool_choice": 5,
    }
    with pytest.raises(TypeError):
        to_anthropic_request(req, {})


def test_tool_non_function_type_raises():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "code_interpreter"}],
    }
    with pytest.raises(NotImplementedError, match="code_interpreter"):
        to_anthropic_request(req, {})


def test_tool_missing_description_omitted():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    out = to_anthropic_request(req, {})
    assert out["tools"] == [
        {"name": "f", "input_schema": {"type": "object"}, "type": "custom"}
    ]
    assert "description" not in out["tools"][0]


def test_multiple_tool_calls_in_one_assistant_message():
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
                        "function": {"name": "a", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"},
                    },
                ],
            }
        ],
    }
    out = to_anthropic_request(req, {})
    assert out["messages"][0]["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "a", "input": {}},
        {"type": "tool_use", "id": "call_2", "name": "b", "input": {}},
    ]


def test_assistant_text_plus_tool_calls_not_collapsed():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"},
                    }
                ],
            }
        ],
    }
    out = to_anthropic_request(req, {})
    assert out["messages"][0]["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "call_1", "name": "a", "input": {}},
    ]


def test_assistant_empty_content_and_no_tool_calls():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "assistant", "content": None}],
    }
    out = to_anthropic_request(req, {})
    assert out["messages"][0] == {"role": "assistant", "content": []}


def test_tool_call_invalid_json_arguments_raises():
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
                        "function": {"name": "a", "arguments": "{not json"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="call_1"):
        to_anthropic_request(req, {})


def test_tool_result_is_error_forwarded():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "boom",
                "is_error": True,
            }
        ],
    }
    out = to_anthropic_request(req, {})
    assert out["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "boom",
                    "is_error": True,
                }
            ],
        }
    ]


def test_function_role_raises_not_implemented():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "function", "name": "f", "content": "x"}],
    }
    with pytest.raises(NotImplementedError, match="function"):
        to_anthropic_request(req, {})


def test_unsupported_role_raises_value_error():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "developer", "content": "x"}],
    }
    with pytest.raises(ValueError, match="developer"):
        to_anthropic_request(req, {})


def test_image_url_missing_url_raises():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
        ],
    }
    with pytest.raises(ValueError, match="image_url"):
        to_anthropic_request(req, {})


def test_user_content_none_becomes_empty_string():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": None}],
    }
    out = to_anthropic_request(req, {})
    assert out["messages"][0] == {"role": "user", "content": ""}


def test_system_only_content_none_dropped():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "system", "content": "A"},
            {"role": "system", "content": None},
            {"role": "user", "content": "hi"},
        ],
    }
    out = to_anthropic_request(req, {})
    assert out["system"] == [{"type": "text", "text": "A"}]


def test_empty_messages_list():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [],
    }
    out = to_anthropic_request(req, {})
    assert out["messages"] == []
    assert "system" not in out


def test_max_completion_tokens_falsy_falls_back_to_max_tokens():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_completion_tokens": 0,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "x"}],
    }
    assert to_anthropic_request(req, {})["max_tokens"] == 50


def test_response_format_json_schema_converted_to_output_config():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": schema, "strict": True},
        },
    }
    out = to_anthropic_request(req, {})
    assert out["output_config"] == {"format": {"type": "json_schema", "schema": schema}}


def test_no_response_format_omits_output_config():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
    }
    assert "output_config" not in to_anthropic_request(req, {})


def test_response_format_text_omits_output_config():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {"type": "text"},
    }
    assert "output_config" not in to_anthropic_request(req, {})


def test_response_format_json_object_raises():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {"type": "json_object"},
    }
    with pytest.raises(NotImplementedError, match="json_object"):
        to_anthropic_request(req, {})


def test_response_format_missing_schema_raises():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "weather"}},
    }
    with pytest.raises(ValueError, match="schema"):
        to_anthropic_request(req, {})


# ==================== Anthropic -> OpenAI ====================

def test_from_anthropic_minimal_request():
    anth = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    out, extras = from_anthropic_request(anth)
    assert out == {
        "model": "claude-3-5-sonnet",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert extras == {}


def test_from_anthropic_system_hoisted_to_message():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    out, _ = from_anthropic_request(anth)
    assert out["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]


def test_from_anthropic_system_blocks_joined():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        "messages": [{"role": "user", "content": "Hi"}],
    }
    out, _ = from_anthropic_request(anth)
    assert out["messages"][0] == {"role": "system", "content": "AB"}


def test_from_anthropic_tool_use_and_result():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "SF"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"temp": 72}'}],
            },
        ],
    }
    out, _ = from_anthropic_request(anth)
    assert out["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
    ]


def test_from_anthropic_tools_and_tool_choice():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                "type": "custom",
            }
        ],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    out, _ = from_anthropic_request(anth)
    assert out["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert out["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_from_anthropic_tool_choice_auto_any_none():
    base = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}
    assert from_anthropic_request({**base, "tool_choice": {"type": "auto"}})[0]["tool_choice"] == "auto"
    assert from_anthropic_request({**base, "tool_choice": {"type": "any"}})[0]["tool_choice"] == "required"
    assert from_anthropic_request({**base, "tool_choice": {"type": "none"}})[0]["tool_choice"] == "none"


def test_from_anthropic_image_base64_and_url():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
                    },
                ],
            }
        ],
    }
    out, _ = from_anthropic_request(anth)
    assert out["messages"][0]["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]


def test_from_anthropic_stop_sequences_to_stop():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "stop_sequences": ["END", "STOP"],
    }
    out, _ = from_anthropic_request(anth)
    assert out["stop"] == ["END", "STOP"]


def test_from_anthropic_metadata_user_id_mapped():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "metadata": {"user_id": "u_123"},
    }
    out, _ = from_anthropic_request(anth)
    assert out["user"] == "u_123"


def test_from_anthropic_thinking_top_k_service_tier_go_to_extras():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "top_k": 40,
        "service_tier": "priority",
    }
    out, extras = from_anthropic_request(anth)
    assert "thinking" not in out
    assert "top_k" not in out
    assert "service_tier" not in out
    assert extras == {
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "top_k": 40,
        "service_tier": "priority",
    }


def test_from_anthropic_output_config_to_response_format():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    out, _ = from_anthropic_request(anth)
    assert out["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


def test_from_anthropic_unsupported_tool_type_raises():
    anth = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "f", "type": "bash_20250124"}],
    }
    with pytest.raises(NotImplementedError, match="bash_20250124"):
        from_anthropic_request(anth)


def test_roundtrip_openai_to_anthropic_to_openai():
    req: ChatCompletionRequest = {
        "model": "m",
        "max_tokens": 10,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }
    anth = to_anthropic_request(req, {})
    back, extras = from_anthropic_request(anth)
    assert back == req
    assert extras == {}
