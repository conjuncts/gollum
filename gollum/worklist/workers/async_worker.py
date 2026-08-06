from typing import TYPE_CHECKING

from gollum.types import GollumResponse
from gollum.types.chat_completions import ChatCompletionResponseModel
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Provider, Worker


if TYPE_CHECKING:
    from openai import AsyncOpenAI

class AsyncOpenAIWorker(Provider):
    def __init__(self, client: "AsyncOpenAI"):
        self.client = client

    async def process(self, worklist_entry: WorklistEntry) -> None:
        compl = worklist_entry.request.request
        # drop any that are None. (model and messages are required)
        compl = {k: v for k, v in compl.items() if k in ["model", "messages"] or v is not None}
        if isinstance(compl["model"], str):
            compl["model"] = compl["model"].removeprefix("openai/")
        result: ChatCompletionResponseModel = await self.client.chat.completions.create(**compl)
        as_dict = result.model_dump()
        worklist_entry.finish(GollumResponse(as_dict, extras={}, metadata={}))

    def supports(self, worklist_entry: WorklistEntry) -> bool:
        # from openai.types import AllModels
        model_name = worklist_entry.request.request["model"]
        valid_prefixes = [
            "openai/",
            "gpt-",
            "o1",
            "o3",
            "o4",
            "chatgpt-",
            "codex-",
        ]
        return any(model_name.startswith(prefix) for prefix in valid_prefixes)
