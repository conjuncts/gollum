"""
Standard example requests every provider worker should handle correctly.

Whenever a new provider is wired up (a new `gollum/provider/<name>.py`), its
worker test should run against these examples to confirm that the core
feature set survives the round trip: request in (OpenAI ChatCompletions
wire format, gollum's canonical request shape) -> provider SDK call ->
response out (OpenAI ChatCompletion wire format).

Each example is deliberately minimal - just enough to exercise the feature,
not a realistic prompt.
"""

from typing import NotRequired, TypedDict

from gollum.types.chat_completions import ChatCompletionRequest

# A small triangle image, so image examples don't depend on network access.
# Source: https://commons.wikimedia.org/wiki/File:Triangle.png
_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAAAAABVicqIAAABeElEQVR4Ae3ZAUZEUQCF4bPNAR4xxCAeMYQDZg/tYfbQHmYP7WGAUpUbhO8Zgftv4IAPnLx53ByZI7drjsyRObL8w8hdrz6CvbZ7H8Hu2158hHpp2wOPWA/96Kwj1HM/W3WEeuxXJxyR1n53xBHpqT+RyJjDkYiMORyJyKDDEYgMOhyByKDDEYgMOhyByKDDEYgMO2xZZNxhWWTcoYuMO3SRcYcuMu7QRcYdusi4QxcZd+gi4w5dZNyhi4w7dJFxhy4y7tBFxh26yLhDFxl36CLjDl1kwKGIhJFlOCyJhJFrRyYSRvYdkUgYuRS6bBw5FDpsGzmXOm8aWUutW0ZOxU4bRo7Fjj6y9K9cZNyhi4w7dJFxhy4y7tBFxh26yLhDFxl36CLjDl1k3KGLjDt0kXGHLjLu0EXGHbrIuEMXGXfoIuMOXWTcoYuMO3SRcYcuMu7QRcYdusi4QxcZd+gi4w5dZNyhi/w1sltu2m4+ptAcgeaINkfmyDtnNCsk0kajiQAAAABJRU5ErkJggg=="
)


class StandardExample(TypedDict):
    id: str
    description: str
    request: ChatCompletionRequest
    # provider ids (as used in ProviderRegistry) known to not support this
    # example yet - tests should xfail/skip rather than fail for these.
    unsupported: NotRequired[set[str]]


STANDARD_EXAMPLES: list[StandardExample] = [
    {
        "id": "simple",
        "description": "A bare single-turn user message with no extra features.",
        "request": {
            "model": "placeholder",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            # "max_completion_tokens": 100,
        },
    },
    {
        "id": "system_prompt",
        "description": "A system message paired with a user message.",
        "request": {
            "model": "placeholder",
            "messages": [
                {"role": "system", "content": "You are a terse assistant. Answer in one word."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            # "max_completion_tokens": 100,
        },
    },
    {
        "id": "image_input",
        "description": "A user message containing text plus an inline base64 image.",
        "request": {
            "model": "placeholder",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What shape is in this image?"},
                        {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URL}},
                    ],
                }
            ],
            # "max_completion_tokens": 100,
        },
    },
    {
        "id": "image_url",
        "description": "A user message containing text plus a plain (non-data) image URL.",
        "request": {
            "model": "placeholder",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/6/63/Triangle.png"
                        }},
                    ],
                }
            ],
            # "max_completion_tokens": 100,
        },
    },
    {
        "id": "function_calling",
        "description": "A request offering a tool the model may call.",
        "request": {
            "model": "placeholder",
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": [
                {
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
                }
            ],
            "tool_choice": "auto",
            # "max_completion_tokens": 100,
        },
    },
    {
        "id": "structured_output",
        "description": "A request constraining the response to a JSON schema.",
        "request": {
            "model": "placeholder",
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "temp_f": {"type": "number"},
                        },
                        "required": ["city", "temp_f"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            # "max_completion_tokens": 100,
        },
    },
    {
        "id": "web_search",
        "description": "A request offering a server-side web search tool.",
        "request": {
            "model": "placeholder",
            "messages": [{"role": "user", "content": "Who won the most recent F1 race?"}],
            "tools": [{"type": "web_search"}],
            # "max_completion_tokens": 100,
        },
        # OpenAI's Chat Completions API (as opposed to the Responses API) has
        # no server-side web search tool at all, so this is a known gap, not
        # a gollum bug. Anthropic's *output* converter already understands
        # `web_search_tool_result` blocks, but the *input* converter doesn't
        # yet translate an OpenAI-shaped `{"type": "web_search"}` tool into
        # Anthropic's server tool, so it's unsupported end-to-end for now too.
        # Keep the example here anyway: it documents the target request shape
        # for whichever provider adds support first.
        "unsupported": {"openai", "anthropic"},
    },
]
