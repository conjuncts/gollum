def guess_provider_type(model_name: str) -> str:
    """
    If a provider prefix (ie. openai/) is not specified, this function will attempt to guess the provider type based on the model name.
    """
    if "/" in model_name:
        return model_name.split("/", 1)[0]

    # from openai.types import AllModels
    openai_prefixes = [
        "gpt-",
        "o1",
        "o3",
        "o4",
        "chatgpt-",
        "codex-",
    ]
    if any(model_name.startswith(prefix) for prefix in openai_prefixes):
        return "openai"

    if model_name.startswith("claude-"):
        return "anthropic"

    if model_name.startswith("gemini-"):
        return "vertex_ai"  # consistent with https://docs.litellm.ai/docs/providers/gemini

    if model_name.startswith("mistral-"):
        return "mistral"
