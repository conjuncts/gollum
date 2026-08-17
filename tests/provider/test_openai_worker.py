from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gollum.client.base import GollumClient
from gollum.client.litellm import GollumRouter
from gollum.provider.openai import AsyncOpenAIWorker
from gollum.provider.provider_registry import ProviderRegistry
from gollum.worklist.concurrent_worklist import ConcurrentWorklist
from gollum.worklist.workers.polymorphic_worker import AsyncPolymorphicWorker


def _fake_response(as_dict: dict) -> SimpleNamespace:
    """Stands in for the openai SDK's ChatCompletion pydantic model."""
    result = SimpleNamespace(**as_dict)
    result.model_dump = lambda: as_dict
    return result


def _gollum_client(fake_create: AsyncMock) -> GollumClient:
    fake_openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    registry = ProviderRegistry()
    registry.register_provider("openai", lambda: AsyncOpenAIWorker(client=fake_openai_client))

    worklist = ConcurrentWorklist()
    worklist.enroll_worker(AsyncPolymorphicWorker(provider_registry=registry))
    return GollumClient(worklist)


@pytest.mark.asyncio
async def test_function_calling_forwards_tools_and_returns_tool_calls():
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]

    fake_create = AsyncMock(return_value=_fake_response({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.6-luna",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }],
            },
        }],
    }))

    router = GollumRouter(client=_gollum_client(fake_create))
    response = await router.acompletion(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        tool_choice="auto",
    )

    # gollum must forward tools/tool_choice to the OpenAI SDK untouched
    _, call_kwargs = fake_create.call_args
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"

    # and must surface the resulting tool call back to the caller untouched
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city": "Paris"}'
    assert response.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_structured_output_forwards_response_format_and_returns_content():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "temp_f": {"type": "number"}},
                "required": ["city", "temp_f"],
            },
            "strict": True,
        },
    }

    fake_create = AsyncMock(return_value=_fake_response({
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.6-luna",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"city": "Paris", "temp_f": 68}',
            },
        }],
    }))

    router = GollumRouter(client=_gollum_client(fake_create))
    response = await router.acompletion(
        model="openai/gpt-5.6-luna",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        response_format=response_format,
    )

    # gollum must forward response_format to the OpenAI SDK untouched
    _, call_kwargs = fake_create.call_args
    assert call_kwargs["response_format"] == response_format

    # and must surface the resulting structured content back to the caller untouched
    import json
    parsed = json.loads(response.choices[0].message.content)
    assert parsed == {"city": "Paris", "temp_f": 68}
