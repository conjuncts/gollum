from typing import TYPE_CHECKING

from gollum.types import GollumResponse
from gollum.types.chat_completions import ChatCompletionResponseModel
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker


if TYPE_CHECKING:
    from openai import AsyncOpenAI

class AsyncOpenAIWorker(Worker):
    def __init__(self, client: "AsyncOpenAI"):
        self.client = client

    async def process(self, worklist_entry: WorklistEntry) -> None:
        kwargs = worklist_entry.request.request
        # drop any that are None. (model and messages are required)
        kwargs = {k: v for k, v in kwargs.items() if k in ["model", "messages"] or v is not None}
        if isinstance(kwargs["model"], str):
            kwargs["model"] = kwargs["model"].removeprefix("openai/")
        result: ChatCompletionResponseModel = await self.client.chat.completions.create(**kwargs)
        as_dict = result.model_dump()
        worklist_entry.finish(GollumResponse(as_dict, extras={}, metadata={}))
