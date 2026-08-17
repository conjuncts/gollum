"""
Ground-truth conversion checks for STANDARD_EXAMPLES.
"""

import json
import re
from types import SimpleNamespace

import pytest

from gollum.convert.input.anthropic import to_anthropic_request
from gollum.convert.output.anthropic import anthropic_message_to_completion
from gollum.testing.examples import STANDARD_EXAMPLES, _TINY_PNG_DATA_URL

_EXAMPLES = {e["id"]: e for e in STANDARD_EXAMPLES}

_PNG_MATCH = re.match(r"^data:(?P<media>[^;,]+);base64,(?P<data>.*)$", _TINY_PNG_DATA_URL)
assert _PNG_MATCH is not None
_PNG_MEDIA_TYPE = _PNG_MATCH.group("media")
_PNG_DATA = _PNG_MATCH.group("data")


# ---------- request: OpenAI-shaped example -> Anthropic wire request ----------

_ANTHROPIC_REQUEST_GT = {
    "simple": {
        "model": "placeholder",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    },
    "system_prompt": {
        "model": "placeholder",
        "max_tokens": 4096,
        "system": "You are a terse assistant. Answer in one word.",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    },
    "image_input": {
        "model": "placeholder",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What shape is in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _PNG_MEDIA_TYPE,
                            "data": _PNG_DATA,
                        },
                    },
                ],
            }
        ],
    },
    "image_url": {
        "model": "placeholder",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://upload.wikimedia.org/wikipedia/commons/6/63/Triangle.png"},
                    },
                ],
            }
        ],
    },
    "function_calling": {
        "model": "placeholder",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "type": "custom",
            }
        ],
        "tool_choice": {"type": "auto"},
    },
    "structured_output": {
        "model": "placeholder",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": _EXAMPLES["structured_output"]["request"]["response_format"]["json_schema"]["schema"],
            }
        },
    },
}


@pytest.mark.parametrize("example_id", _ANTHROPIC_REQUEST_GT.keys())
def test_anthropic_request_conversion(example_id):
    req = dict(_EXAMPLES[example_id]["request"])
    assert to_anthropic_request(req, {}) == _ANTHROPIC_REQUEST_GT[example_id]


# ---------- response: Anthropic wire response -> ChatCompletion shape ----------


def _anthropic_response(example_id: str) -> SimpleNamespace:
    body = {
        "id": f"msg-{example_id}",
        "model": "claude-haiku-4-5",
        "role": "assistant",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    if example_id == "function_calling":
        body["stop_reason"] = "tool_use"
        body["content"] = [{
            "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"},
        }]
    elif example_id == "structured_output":
        body["content"] = [{"type": "text", "text": '{"city": "Paris", "temp_f": 68}'}]
    result = SimpleNamespace(**body)
    result.model_dump = lambda: body
    result.model_dump_json = lambda: json.dumps(body)
    return result


_ANTHROPIC_RESPONSE_GT = {
    "simple": {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "model": "claude-haiku-4-5",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "system_prompt": {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "model": "claude-haiku-4-5",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "image_input": {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "model": "claude-haiku-4-5",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "image_url": {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "model": "claude-haiku-4-5",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "function_calling": {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
            }
        ],
        "model": "claude-haiku-4-5",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "structured_output": {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"city": "Paris", "temp_f": 68}',
                },
            }
        ],
        "model": "claude-haiku-4-5",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
}


@pytest.mark.parametrize("example_id", _ANTHROPIC_RESPONSE_GT.keys())
def test_anthropic_response_conversion(example_id):
    body = _anthropic_response(example_id).model_dump()
    out = anthropic_message_to_completion(body).chat_completion
    assert out["choices"] == _ANTHROPIC_RESPONSE_GT[example_id]["choices"]
    assert out["model"] == _ANTHROPIC_RESPONSE_GT[example_id]["model"]
    assert out["usage"] == _ANTHROPIC_RESPONSE_GT[example_id]["usage"]
