import asyncio
import json
import io
import logging
from typing import TYPE_CHECKING, Any, Optional

from gollum.batch.job import BatchJob
from gollum.batch.result import BatchResult
from gollum.permacache.cache_method import CacheMethod
from gollum.types import GollumResponse
from gollum.worklist.base import WorklistEntry
from gollum.worklist.worker import BatchWorker

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.batch import Batch

logger = logging.getLogger(__name__)


class BatchOpenAIWorker(BatchWorker):
    def __init__(
        self,
        client: "AsyncOpenAI",
        completion_window: str = "24h",
        cache_method: Optional[CacheMethod] = None,
    ):
        self.client = client
        self.completion_window = completion_window
        # Must match the CacheMethod used by the BatchHandler this worker is
        # wired to -- custom_id embeds the cache_key so check_batch() can
        # report back which cache_key each result/error belongs to (see
        # _prepare_request_body's custom_id and check_batch below).
        self.cache_method = cache_method if cache_method is not None else CacheMethod()

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

        # Build JSONL lines formatted for OpenAI Batch API. custom_id embeds
        # this entry's cache_key (plus an index, since two entries can
        # legitimately share a cache_key -- e.g. duplicate requests within
        # the same batch -- and OpenAI requires custom_id to be unique per
        # file) so check_batch() can report cache_keys/cache_keys_errors
        # back to BatchHandler in the same shape it uses everywhere else.
        jsonl_lines = []
        for index, entry in enumerate(worklist_entries):
            body = self._prepare_request_body(entry)
            cache_key = self.cache_method.generate_cache_key(entry.request)
            line_item = {
                "custom_id": f"{cache_key}__{index}",
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

        logger.info("Submitted batch %s (%d entries)", batch.id, len(worklist_entries))

        return BatchJob(batch_id=batch.id, provider_name="openai")

    async def check_batch(self, batch_job: BatchJob) -> BatchResult:
        """Retrieves the current status of the batch job."""
        batch: "Batch" = await self.client.batches.retrieve(batch_job.batch_id)

        if batch.status not in ["completed", "failed", "cancelled", "expired"]:
            return BatchResult(status="pending", cache_keys=[], results=[])

        logger.info("Downloaded batch %s (status=%s)", batch.id, batch.status)

        completed_cache_keys = []
        completed_results = []
        error_cache_keys = []
        error_results = []

        # Process successful outputs if available
        if batch.output_file_id:
            file_response = await self.client.files.content(batch.output_file_id)
            content = file_response.text
            for line in content.strip().split("\n"):
                if not line:
                    continue
                data = json.loads(line)
                cache_key = data["custom_id"].rsplit("__", 1)[0]
                response_body = data.get("response", {}).get("body", {})
                gollum_resp = GollumResponse(response_body, extras={}, metadata={}, original=None)
                completed_cache_keys.append(cache_key)
                completed_results.append(gollum_resp)

        # Process error outputs if available
        if batch.error_file_id:
            file_response = await self.client.files.content(batch.error_file_id)
            content = file_response.text
            for line in content.strip().split("\n"):
                if not line:
                    continue
                data = json.loads(line)
                cache_key = data["custom_id"].rsplit("__", 1)[0]
                error_cache_keys.append(cache_key)
                error_results.append(data)

        return BatchResult(
            status="completed" if batch.status == "completed" else "error",
            cache_keys=completed_cache_keys,
            results=completed_results,
            cache_keys_errors=error_cache_keys,
            errors=error_results,
        )

    async def await_batch(
        self, batch_job: BatchJob, poll_interval: float = 10.0
    ) -> BatchResult:
        """Polls until the batch job reaches a terminal state."""
        while True:
            result = await self.check_batch(batch_job)
            if result.status != "pending":
                return result
            await asyncio.sleep(poll_interval)
