import base64
import io
from typing import TypeGuard

from PIL import Image

from gollum.types.chat_completions import ChatCompletionMessage

def is_image(obj) -> TypeGuard[Image.Image]:
    """Check if object is a PIL Image."""
    return isinstance(obj, Image.Image)

def to_completion_message(image: Image.Image) -> ChatCompletionMessage:
    """Convert a PIL Image to an OpenAI-style user message.

    The image is embedded as a base64 data URI (PNG, lossless) so it can be
    sent to any OpenAI-compatible endpoint (OpenAI, LiteLLM, Ollama, ...).
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            }
        ],
    }
