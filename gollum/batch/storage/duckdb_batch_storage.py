from typing import Optional

from gollum.batch.storage.batch_storage import BatchStorage
from gollum.batch.job import BatchJob
from gollum.folder.file_manager import FileManager

import duckdb

import atexit
import threading
from pathlib import Path


# ============================================================
# Schema
#
# Two tables, kept deliberately separate from the permacache DB
# (different file, different lifecycle -- batch records are
# short-lived bookkeeping, not a durable response cache):
#
#   batches:     one row per BatchJob (batch_id PRIMARY KEY)
#   batch_keys:  one row per cache_key, pointing at the batch_id
#                that currently "owns" it (cache_key PRIMARY KEY,
#                since a key should only ever be in flight in one
#                batch at a time)
#
# retrieve() is a join; free_batch() deletes from both tables.
# ============================================================

_BATCHES_TABLE = "batches"
_BATCH_KEYS_TABLE = "batch_keys"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_BATCHES_TABLE} (
    batch_id VARCHAR PRIMARY KEY,
    provider_name VARCHAR
);

CREATE TABLE IF NOT EXISTS {_BATCH_KEYS_TABLE} (
    cache_key VARCHAR PRIMARY KEY,
    batch_id VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_batch_keys_batch_id
    ON {_BATCH_KEYS_TABLE} (batch_id);
"""


class DuckDBBatchStorage(BatchStorage):
    """
    DuckDB-backed BatchCache: tracks which in-flight batch job "owns"
    each cache_key, so gollum can avoid resubmitting work that's
    already been sent off to a provider's batch API.

    Unlike DuckDBPermacache, this cache is not append-mostly and
    doesn't benefit from write-batching -- record_batch() and
    free_batch() are comparatively rare, bursty operations (once per
    batch submission / completion, potentially touching thousands of
    cache_keys at once), so writes go straight to disk via a single
    transaction rather than being buffered behind a flush_threshold.
    """

    def __init__(self, fm: FileManager):
        super().__init__()
        self.fm = fm
        self._lock = threading.RLock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._db_path))
        self._con.execute(_DDL)

        atexit.register(self.close)

    # ---------- paths ----------

    @property
    def _db_path(self) -> Path:
        # Assumes FileManager exposes a batch-cache-specific path,
        # analogous to path_permacache(). Kept in its own file so a
        # long-lived permacache DB and a churny batch-tracking DB
        # never contend for the same file handle / WAL.
        return self.fm.path_batchcache() / "duckdb" / "v1" / "batches.duckdb"

    # ---------- public API ----------

    async def record_batch(self, batch: BatchJob, cache_keys: list[str]):
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._con.execute(
                    f"""
                    INSERT INTO {_BATCHES_TABLE} (batch_id, provider_name)
                    VALUES (?, ?)
                    ON CONFLICT (batch_id) DO UPDATE
                        SET provider_name = excluded.provider_name
                    """,
                    [batch.batch_id, batch.provider_name],
                )
                if cache_keys:
                    self._con.executemany(
                        f"""
                        INSERT INTO {_BATCH_KEYS_TABLE} (cache_key, batch_id)
                        VALUES (?, ?)
                        ON CONFLICT (cache_key) DO UPDATE
                            SET batch_id = excluded.batch_id
                        """,
                        [[cache_key, batch.batch_id] for cache_key in cache_keys],
                    )
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise

    async def retrieve(self, cache_key: str, likely_partition: str) -> Optional[BatchJob]:
        with self._lock:
            row = self._con.execute(
                f"""
                SELECT b.batch_id, b.provider_name
                FROM {_BATCH_KEYS_TABLE} bk
                JOIN {_BATCHES_TABLE} b ON b.batch_id = bk.batch_id
                WHERE bk.cache_key = ?
                """,
                [cache_key],
            ).fetchone()
            if row is None:
                return None
            batch_id, provider_name = row
            return BatchJob(batch_id, provider_name)

    async def free_batch(self, batch: BatchJob):
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._con.execute(
                    f"DELETE FROM {_BATCH_KEYS_TABLE} WHERE batch_id = ?",
                    [batch.batch_id],
                )
                self._con.execute(
                    f"DELETE FROM {_BATCHES_TABLE} WHERE batch_id = ?",
                    [batch.batch_id],
                )
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise

    async def get_all_batches(self) -> list[BatchJob]:
        with self._lock:
            rows = self._con.execute(
                f"SELECT batch_id, provider_name FROM {_BATCHES_TABLE}"
            ).fetchall()
            return [
                BatchJob(batch_id, provider_name)
                for batch_id, provider_name in rows
            ]

    def close(self):
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None
