from gollum.types.chat_completions import ChatCompletionResponse


def primitive_to_completion(value: str) -> ChatCompletionResponse: # -> ChatCompletionResponseModel
    return {
        "id": "mock",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": value,
                },
                
            }
        ],
        "created": 0,
        "model": "mock",
        "object": "chat.completion",
    }