import hashlib

import pytest

from gollum.folder.file_manager import FileManager
from gollum.permacache.pl_permacache import PolarsPermacache
from gollum.types import GollumResponse


def _key(text: str) -> str:
    """Realistic cache keys: sha256 hex digests, like CacheMethod produces."""
    return hashlib.sha256(text.encode()).hexdigest()


def _resp(content: str = "hi") -> GollumResponse:
    return GollumResponse(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        },
        extras={"temperature": 0.7},
        metadata={"gollum.salt": "test"},
    )

@pytest.mark.asyncio
async def test_round_trip_serves_from_memory(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm)
    resp = _resp("hello")
    await pc.store(resp, _key("hello"), likely_partition="")
    # Same object identity proves the in-memory mirror answers, no disk read.
    assert await pc.retrieve(_key("hello"), "") is resp

@pytest.mark.asyncio
async def test_miss_returns_none(tmp_path):
    pc = PolarsPermacache(FileManager(tmp_path))
    assert await pc.retrieve(_key("nope"), "") is None

@pytest.mark.asyncio
async def test_persists_across_instances(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=1)
    resp = _resp("hello")
    await pc.store(resp, _key("hello"), "")
    await pc.flush()

    pc2 = PolarsPermacache(fm)
    got = await pc2.retrieve(_key("hello"), "")
    assert got is not None
    # Note: the polars struct round trip fills unset fields with None, so
    # compare meaningful fields rather than exact dict equality.
    assert got.chat_completion["model"] == resp.chat_completion["model"]
    assert got.chat_completion["choices"][0]["message"]["role"] == "assistant"
    assert got.chat_completion["choices"][0]["message"]["content"] == "hello"
    assert got.extras == resp.extras
    assert got.metadata == resp.metadata
    assert got.original == resp.original

@pytest.mark.asyncio
async def test_auto_flush_at_threshold(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=2)
    await pc.store(_resp("a"), _key("a"), "")
    assert not list(fm.path_permacache().glob("**/*.parquet"))
    await pc.store(_resp("b"), _key("b"), "")
    assert list(fm.path_permacache().glob("**/*.parquet"))

@pytest.mark.asyncio
async def test_shards_spread_keys_and_round_trip(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=5, num_shards=2)
    n = 50
    keys = [_key(f"k{i}") for i in range(n)]
    for i, k in enumerate(keys):
        await pc.store(_resp(f"msg{i}"), k, "")
    await pc.flush()

    shards = list(fm.path_permacache().glob("polars/v1/*.parquet"))
    assert len(shards) > 1  # keys actually spread across shards

    pc2 = PolarsPermacache(fm, num_shards=2)
    for i, k in enumerate(keys):
        got = await pc2.retrieve(k, "")
        assert got is not None
        assert got.chat_completion["choices"][0]["message"]["content"] == f"msg{i}"


@pytest.mark.asyncio
async def test_flush_at_exit_persists_pending(tmp_path):
    """Rows buffered below the flush threshold survive interpreter exit."""
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=10)
    await pc.store(_resp("hello"), _key("hello"), "")
    assert not list(fm.path_permacache().glob("**/*.parquet"))

    # Simulate interpreter exit: fire the registered atexit hook directly.
    pc._flush_at_exit()
    assert list(fm.path_permacache().glob("**/*.parquet"))

    pc2 = PolarsPermacache(fm)
    got = await pc2.retrieve(_key("hello"), "")
    assert got is not None
    assert got.chat_completion["choices"][0]["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_dedupe_keeps_latest(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=10)
    k = _key("dup")
    await pc.store(_resp("v1"), k, "")
    await pc.store(_resp("v2"), k, "")
    assert (await pc.retrieve(k, "")).chat_completion["choices"][0]["message"]["content"] == "v2"
    await pc.flush()

    pc2 = PolarsPermacache(fm)
    assert (await pc2.retrieve(k, "")).chat_completion["choices"][0]["message"]["content"] == "v2"
