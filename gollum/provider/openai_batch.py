import asyncio
import json
import io
from typing import TYPE_CHECKING, Any

from gollum.batch.job import BatchJob
from gollum.batch.result import BatchResult
from gollum.types import GollumResponse
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import BatchWorker

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.batch import Batch


class BatchOpenAIWorker(BatchWorker):
    def __init__(self, client: "AsyncOpenAI", completion_window: str = "24h"):
        self.client = client
        self.completion_window = completion_window

    def _prepare_request_body(self, worklist_entry: WorklistEntry) -> dict[str, Any]:
        """Formats and cleans the chat completion request payload."""
        compl = worklist_entry.request.chat_completion
        
        # Drop keys with None values (keep required ones like model & messages)
        compl = {k: v for k, v in compl.items() if k in ["model", "messages"] or v is not None}
        
        # Strip provider prefix if present
        if isinstance(compl.get("model"), str):
            compl["model"] = compl["model"].removeprefix("openai/")
            
        return compl

    async def send_batch(self, worklist_entries: list[WorklistEntry]) -> BatchJob:
        """
        Serializes entries into a JSONL file, uploads it to OpenAI, 
        and creates a batch processing job.
        """
        if not worklist_entries:
            return None

        # Build JSONL lines formatted for OpenAI Batch API
        jsonl_lines = []
        for index, entry in enumerate(worklist_entries):
            body = self._prepare_request_body(entry)
            line_item = {
                "custom_id": f"req_{index}_{id(entry)}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            jsonl_lines.append(json.dumps(line_item))

        jsonl_data = "\n".join(jsonl_lines).encode("utf-8")

        # Upload JSONL file to OpenAI
        file_obj = await self.client.files.create(
            file=("batch_requests.jsonl", io.BytesIO(jsonl_data)),
            purpose="batch"
        )

        # Create Batch job
        batch: "Batch" = await self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=self.completion_window,
        )

        return BatchJob(id=batch.id) # , raw_job=batch)

    async def check_batch(self, batch_job: BatchJob) -> BatchResult:
        """Retrieves the current status of the batch job."""
        batch: "Batch" = await self.client.batches.retrieve(batch_job.id)

        if batch.status not in ["completed", "failed", "cancelled", "expired"]:
            return BatchResult(status=batch.status, completed=[], errors=[])

        completed_results = []
        error_results = []

        # Process successful outputs if available
        if batch.output_file_id:
            file_response = await self.client.files.content(batch.output_file_id)
            content = file_response.text
            for line in content.strip().split("\n"):
                if not line:
                    continue
                data = json.loads(line)
                response_body = data.get("response", {}).get("body", {})
                gollum_resp = GollumResponse(response_body, extras={}, metadata={}, original=None)
                completed_results.append(gollum_resp)

        # Process error outputs if available
        if batch.error_file_id:
            file_response = await self.client.files.content(batch.error_file_id)
            content = file_response.text
            for line in content.strip().split("\n"):
                if not line:
                    continue
                data = json.loads(line)
                error_results.append(data)

        return BatchResult(
            status=batch.status,
            results=completed_results,
            errors=error_results
        )

    async def await_batch(
        self, batch_job: BatchJob, poll_interval: float = 10.0
    ) -> BatchResult:
        """Polls until the batch job reaches a terminal state."""
        while True:
            result = await self.check_batch(batch_job)
            if result.status in ["completed", "failed", "cancelled", "expired"]:
                return result
            await asyncio.sleep(poll_interval)
