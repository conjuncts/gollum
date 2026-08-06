from gollum.convert.input.anthropic import to_anthropic_request
from gollum.convert.output.anthropic import anthropic_message_to_completion
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import Worker

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message


class AsyncAnthropicWorker(Worker):
    def __init__(self, client: "AsyncAnthropic"):
        self.client = client

    async def process(self, worklist_entry: WorklistEntry) -> None:
        compl = worklist_entry.request.chat_completion
        # drop any that are None. (model and messages are required)
        as_anthropic_request = to_anthropic_request(compl, extras=worklist_entry.request.extras)
        result: "Message" = await self.client.messages.create(**as_anthropic_request)
        resp = anthropic_message_to_completion(result.model_dump())
        worklist_entry.finish(resp)
