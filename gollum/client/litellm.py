from types import SimpleNamespace
from typing import Any, Generator

from gollum.types.chat_completions import ChatCompletionResponse, ChatCompletionResponseModel
from gollum.worklist.base import WorklistEntry

def _nested_simple_namespace(data: ChatCompletionResponse) -> ChatCompletionResponseModel:
    if isinstance(data, dict):
        return SimpleNamespace(**{k: _nested_simple_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [_nested_simple_namespace(item) for item in data]
    else:
        return data


class LiteLLMWorklistEntry:
    def __init__(self, worklist_entry: WorklistEntry):
        self.worklist_entry = worklist_entry

    def __await__(self) -> Generator[Any, None, ChatCompletionResponseModel]:
        # wrap the gollum response in a SimpleNamespace

        async def wrap_response():
            response = await self.worklist_entry.wait()
            return _nested_simple_namespace(response.response)

        return wrap_response().__await__()
