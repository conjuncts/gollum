from gollum.permacache.cache_method import CacheMethod
from gollum.types import GollumRequest


def test_cache_method():
    method = CacheMethod()
    gr = GollumRequest(
        chat_completion={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        },
        extras={},
        metadata={"gollum_salt": "test"},
        provider_name="openai",
    )
    assert method.generate_cache_key(gr) == "88f6c53b67db1b6c5fa6f0ee5df8078b01a26cba421f34ec13e2d986f4c86f17"