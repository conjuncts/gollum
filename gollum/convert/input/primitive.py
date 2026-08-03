from typing import Union

from gollum.types.chat_completions import Message


def primitive_to_message(
    primitive: Union[str]
) -> Message:
    """
    Convenience method that converts a primitive type (str, PIL.Image, list) to a user message
    """
    return {
        "role": "user",
        "content": primitive,
    }