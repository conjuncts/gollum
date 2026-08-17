from gollum.convert.input.openai_responses import to_responses_request
from gollum.convert.output.openai_responses import responses_response_to_completion
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.responses import Response


class AsyncOpenAIResponsesWorker(Worker):
    """Routes requests through the OpenAI Responses API instead of ChatCompletions."""

    def __init__(self, client: "AsyncOpenAI", *, store_original=True):
        self.client = client
        self.store_original = store_original
        """Set to false to save some space"""

    async def process(self, worklist_entry: WorklistEntry) -> bool:
        compl = worklist_entry.request.chat_completion
        as_responses_request = to_responses_request(compl, extras=worklist_entry.request.extras)
        if isinstance(as_responses_request.get("model"), str):
            as_responses_request["model"] = as_responses_request["model"].removeprefix("openai/")
        result: "Response" = await self.client.responses.create(**as_responses_request)
        resp = responses_response_to_completion(result.model_dump())
        if self.store_original:
            resp.original = result.model_dump_json()
        worklist_entry.finish(resp)
        return True
