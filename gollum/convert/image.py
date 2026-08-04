from typing import TypeGuard

from PIL import Image

def is_image(obj) -> TypeGuard[Image.Image]:
    """Check if object is a PIL Image."""
    return isinstance(obj, Image.Image)