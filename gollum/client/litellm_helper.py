from typing import Any, Generator

from openai.types.chat.chat_completion import ChatCompletion

from gollum.types.chat_completions import ChatCompletionResponseModel
from gollum.worklist.base import WorklistEntry


class LiteLLMWorklistEntry:
    def __init__(self, worklist_entry: WorklistEntry):
        self.worklist_entry = worklist_entry

    def __await__(self) -> Generator[Any, None, ChatCompletionResponseModel]:
        async def wrap_response():
            response = await self.worklist_entry.wait()
            return ChatCompletion.model_validate(response.chat_completion)

        return wrap_response().__await__()
