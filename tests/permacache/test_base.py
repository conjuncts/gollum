import hashlib

from gollum.folder.file_manager import FileManager
from gollum.permacache.pl_permacache import PolarsPermacache
from gollum.types import GollumRequest


def _key(text: str) -> str:
    """Realistic cache keys: sha256 hex digests, like CacheMethod produces."""
    return hashlib.sha256(text.encode()).hexdigest()


def _req(content: str = "hi") -> GollumRequest:
    return GollumRequest(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": content}]},
        extras={"temperature": 0.7},
        metadata={"gollum.salt": "test"},
        provider_name="openai",
    )


def test_round_trip_serves_from_memory(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm)
    req = _req("hello")
    pc.store(req, _key("hello"), likely_partition="")
    # Same object identity proves the in-memory mirror answers, no disk read.
    assert pc.retrieve(_key("hello"), "") is req


def test_miss_returns_none(tmp_path):
    pc = PolarsPermacache(FileManager(tmp_path))
    assert pc.retrieve(_key("nope"), "") is None


def test_persists_across_instances(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=1)
    req = _req("hello")
    pc.store(req, _key("hello"), "")
    pc.flush()

    pc2 = PolarsPermacache(fm)
    got = pc2.retrieve(_key("hello"), "")
    assert got is not None
    # Note: the polars struct round trip fills unset fields with None, so
    # compare meaningful fields rather than exact dict equality.
    assert got.chat_completion["model"] == req.chat_completion["model"]
    assert got.chat_completion["messages"][0]["role"] == "user"
    assert got.chat_completion["messages"][0]["content"] == "hello"
    assert got.extras == req.extras
    assert got.metadata == req.metadata
    assert got.provider_name == req.provider_name


def test_auto_flush_at_threshold(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=2)
    pc.store(_req("a"), _key("a"), "")
    assert not list(fm.path_permacache().glob("**/*.parquet"))
    pc.store(_req("b"), _key("b"), "")
    assert list(fm.path_permacache().glob("**/*.parquet"))


def test_shards_spread_keys_and_round_trip(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=5, num_shards=2)
    n = 50
    keys = [_key(f"k{i}") for i in range(n)]
    for i, k in enumerate(keys):
        pc.store(_req(f"msg{i}"), k, "")
    pc.flush()

    shards = list(fm.path_permacache().glob("polars/v1/*.parquet"))
    assert len(shards) > 1  # keys actually spread across shards

    pc2 = PolarsPermacache(fm, num_shards=2)
    for i, k in enumerate(keys):
        got = pc2.retrieve(k, "")
        assert got is not None
        assert got.chat_completion["messages"][0]["content"] == f"msg{i}"


def test_dedupe_keeps_latest(tmp_path):
    fm = FileManager(tmp_path)
    pc = PolarsPermacache(fm, flush_threshold=10)
    k = _key("dup")
    pc.store(_req("v1"), k, "")
    pc.store(_req("v2"), k, "")
    assert pc.retrieve(k, "").chat_completion["messages"][0]["content"] == "v2"
    pc.flush()

    pc2 = PolarsPermacache(fm)
    assert pc2.retrieve(k, "").chat_completion["messages"][0]["content"] == "v2"
