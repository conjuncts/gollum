from typing import TYPE_CHECKING, Union

from gollum.convert.image import is_image, to_completion_message
from gollum.types.chat_completions import ChatCompletionMessage

if TYPE_CHECKING:
    from PIL import Image

def primitive_to_message(
    primitive: Union[str, "Image.Image"]
) -> ChatCompletionMessage:
    """
    Convenience method that converts a primitive type (str, PIL.Image, list) to a user message
    """
    if is_image(primitive):
        return to_completion_message(primitive)
    return {
        "role": "user",
        "content": primitive,
    }
