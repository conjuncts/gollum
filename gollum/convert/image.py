from typing import TypeGuard

from PIL import Image

from gollum.types.chat_completions import ChatCompletionMessage

def is_image(obj) -> TypeGuard[Image.Image]:
    """Check if object is a PIL Image."""
    return isinstance(obj, Image.Image)

def to_completion_message(image: Image.Image) -> ChatCompletionMessage:
    pass
