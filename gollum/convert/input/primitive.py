from typing import TYPE_CHECKING, Union

from gollum.convert.image import is_image

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from PIL import Image

def primitive_to_message(
    primitive: Union[str, "Image.Image"]
) -> "ChatCompletionMessageParam":
    """
    Convenience method that converts a primitive type (str, PIL.Image, list) to a user message
    """
    if is_image(primitive):
        raise NotImplementedError
        return {
            "role": "user",
            "content": primitive,
        }
    return {
        "role": "user",
        "content": primitive,
    }
