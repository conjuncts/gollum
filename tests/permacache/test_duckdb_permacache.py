import hashlib
import shutil
from pathlib import Path

import pytest

from gollum.folder.file_manager import FileManager
from gollum.permacache.duckdb_permacache import DuckDBPermacache
from gollum.types import GollumResponse

pytestmark = pytest.mark.duckdb


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
        metadata={"gollum_salt": "test"},
    )


@pytest.mark.asyncio
async def test_round_trip_serves_from_memory(tmp_path):
    fm = FileManager(tmp_path)
    pc = DuckDBPermacache(fm)
    pc._flush_at_exit = lambda: None
    resp = _resp("hello")
    await pc.store(resp, _key("hello"), likely_partition="")
    # Same object identity proves the in-memory pending index answers, no disk read.
    assert await pc.retrieve(_key("hello"), "") is resp


@pytest.mark.asyncio
async def test_miss_returns_none(tmp_path):
    pc = DuckDBPermacache(FileManager(tmp_path))
    pc._flush_at_exit = lambda: None
    assert await pc.retrieve(_key("nope"), "") is None


@pytest.mark.asyncio
async def test_persists_across_instances(tmp_path):
    fm = FileManager(tmp_path)
    pc = DuckDBPermacache(fm, flush_threshold=1)
    pc._flush_at_exit = lambda: None
    resp = _resp("hello")
    await pc.store(resp, _key("hello"), "")
    await pc.flush()
    pc.close()

    pc2 = DuckDBPermacache(fm)
    pc2._flush_at_exit = lambda: None
    got = await pc2.retrieve(_key("hello"), "")
    assert got is not None
    assert got.chat_completion["model"] == resp.chat_completion["model"]
    assert got.chat_completion["choices"][0]["message"]["role"] == "assistant"
    assert got.chat_completion["choices"][0]["message"]["content"] == "hello"
    assert got.extras == resp.extras
    assert got.metadata == resp.metadata
    assert got.original == resp.original
    pc2.close()


@pytest.mark.asyncio
async def test_auto_flush_at_threshold(tmp_path):
    fm = FileManager(tmp_path)
    pc = DuckDBPermacache(fm, flush_threshold=2)
    pc._flush_at_exit = lambda: None
    await pc.store(_resp("a"), _key("a"), "")
    assert pc._con.execute("SELECT count(*) FROM cache").fetchone()[0] == 0
    await pc.store(_resp("b"), _key("b"), "")
    assert pc._con.execute("SELECT count(*) FROM cache").fetchone()[0] == 2
    pc.close()


@pytest.mark.asyncio
async def test_flush_at_exit_persists_pending(tmp_path):
    """Rows buffered below the flush threshold survive interpreter exit."""
    fm = FileManager(tmp_path)
    pc = DuckDBPermacache(fm, flush_threshold=10)
    await pc.store(_resp("hello"), _key("hello"), "")
    assert pc._con.execute("SELECT count(*) FROM cache").fetchone()[0] == 0

    # Simulate interpreter exit: fire the registered atexit hook directly.
    pc._flush_at_exit()
    assert pc._con.execute("SELECT count(*) FROM cache").fetchone()[0] == 1
    pc.close()

    pc2 = DuckDBPermacache(fm)
    pc2._flush_at_exit = lambda: None
    got = await pc2.retrieve(_key("hello"), "")
    assert got is not None
    assert got.chat_completion["choices"][0]["message"]["content"] == "hello"
    pc2.close()


@pytest.mark.asyncio
async def test_dedupe_keeps_latest(tmp_path):
    """cache_key isn't unique any more, but retrieve() still answers with
    the most recently stored value for a given key."""
    fm = FileManager(tmp_path)
    pc = DuckDBPermacache(fm, flush_threshold=10)
    pc._flush_at_exit = lambda: None
    k = _key("dup")
    await pc.store(_resp("v1"), k, "")
    await pc.store(_resp("v2"), k, "")
    assert (await pc.retrieve(k, "")).chat_completion["choices"][0]["message"]["content"] == "v2"
    await pc.flush()

    pc2 = DuckDBPermacache(fm)
    pc2._flush_at_exit = lambda: None
    assert (await pc2.retrieve(k, "")).chat_completion["choices"][0]["message"]["content"] == "v2"
    pc2.close()


@pytest.mark.asyncio
async def test_history_rows_are_appended_not_upserted(tmp_path):
    """Every store() appends a new row (cache_key has no uniqueness
    constraint), so a key's write history is kept on disk."""
    fm = FileManager(tmp_path)
    pc = DuckDBPermacache(fm, flush_threshold=10)
    pc._flush_at_exit = lambda: None
    k = _key("dup")
    await pc.store(_resp("v1"), k, "")
    await pc.store(_resp("v2"), k, "")
    await pc.flush()

    rows = pc._con.execute(
        "SELECT cache_key FROM cache WHERE cache_key = ?", [k]
    ).fetchall()
    assert len(rows) == 2
    pc.close()


@pytest.mark.asyncio
async def test_session_id_and_seq_recorded_per_store(tmp_path):
    fm = FileManager(tmp_path)
    pc = DuckDBPermacache(fm, flush_threshold=10)
    pc._flush_at_exit = lambda: None
    await pc.store(_resp("a"), _key("a"), "")
    await pc.store(_resp("b"), _key("b"), "")
    await pc.flush()

    rows = pc._con.execute(
        "SELECT session_id, seq FROM cache ORDER BY seq"
    ).fetchall()
    assert len(rows) == 2
    (session_a, seq_a), (session_b, seq_b) = rows
    # Both rows came from the same DuckDBPermacache instance/session.
    assert session_a == session_b == pc._session_id
    # The per-session counter increments with each store().
    assert (seq_a, seq_b) == (1, 2)
    pc.close()


@pytest.mark.asyncio
async def test_new_instance_gets_new_session_id(tmp_path):
    fm = FileManager(tmp_path)
    pc1 = DuckDBPermacache(fm, flush_threshold=1)
    pc1._flush_at_exit = lambda: None
    await pc1.store(_resp("a"), _key("a"), "")
    await pc1.flush()
    pc1.close()

    pc2 = DuckDBPermacache(fm, flush_threshold=1)
    pc2._flush_at_exit = lambda: None
    await pc2.store(_resp("b"), _key("b"), "")
    await pc2.flush()

    assert pc1._session_id != pc2._session_id
    rows = dict(
        pc2._con.execute("SELECT cache_key, seq FROM cache").fetchall()
    )
    # Each session's own counter restarts at 1.
    assert rows[_key("a")] == 1
    assert rows[_key("b")] == 1
    pc2.close()


@pytest.mark.asyncio
async def test_no_primary_key_on_cache_key(tmp_path):
    """cache_key must not carry a PRIMARY KEY / uniqueness constraint any
    more, since rows are appended rather than upserted."""
    pc = DuckDBPermacache(FileManager(tmp_path))
    pc._flush_at_exit = lambda: None
    constraints = pc._con.execute(
        "SELECT constraint_type FROM duckdb_constraints() WHERE table_name = 'cache'"
    ).fetchall()
    assert ("PRIMARY KEY",) not in constraints
    pc.close()


@pytest.mark.asyncio
async def test_migrates_legacy_cache_with_primary_key(tmp_path):
    """Opening an older on-disk cache (created back when cache_key was
    PRIMARY KEY) must drop that constraint so duplicate keys can be
    appended, and must preserve the rows already on disk."""
    fm = FileManager(tmp_path)
    dest_dir = Path(tmp_path) / "permacache/duckdb/v1"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/data/cache.duckdb", dest_dir / "cache.duckdb")

    pc = DuckDBPermacache(fm, flush_threshold=1)
    pc._flush_at_exit = lambda: None

    constraints = pc._con.execute(
        "SELECT constraint_type FROM duckdb_constraints() WHERE table_name = 'cache'"
    ).fetchall()
    assert ("PRIMARY KEY",) not in constraints

    pre_migration_count = pc._con.execute("SELECT count(*) FROM cache").fetchone()[0]
    assert pre_migration_count > 0

    existing_key = pc._con.execute("SELECT cache_key FROM cache LIMIT 1").fetchone()[0]
    await pc.store(_resp("post-migration"), existing_key, "")
    await pc.flush()

    post_count = pc._con.execute("SELECT count(*) FROM cache").fetchone()[0]
    assert post_count == pre_migration_count + 1

    got = await pc.retrieve(existing_key, "")
    assert got.chat_completion["choices"][0]["message"]["content"] == "post-migration"
    pc.close()
