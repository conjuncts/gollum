import json

from gollum.convert.output.anthropic import anthropic_message_to_completion

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
