"""
A global client

"""

from gollum.client.base import GollumClient
from gollum.types import GollumRequest
from gollum.types.chat_completions import ChatCompletionRequest


_singleton = None
def get_singleton_client() -> GollumClient:
    global _singleton
    if _singleton is None:
        _singleton = GollumClient()
    return _singleton

def completion(
    request: ChatCompletionRequest
):
    as_standard_format = GollumRequest(
        request=request,
        extras={},
        metadata={},
    )
    return get_singleton_client().worklist.enroll(as_standard_format)
