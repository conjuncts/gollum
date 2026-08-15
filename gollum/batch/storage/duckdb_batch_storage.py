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
#   batches:     one row per BatchJob (batch_id PRIMARY KEY), plus a
#                `completed` flag. complete_batch() sets the flag;
#                it does NOT delete the row. Only free_completed()
#                deletes rows (and their owned batch_keys), and it
#                only ever touches rows where completed = TRUE.
#   batch_keys:  one row per (cache_key, batch_id) pairing. cache_key
#                is no longer unique/PRIMARY KEY -- a key can now be
#                owned by multiple in-flight batches at once -- so
#                this table is a plain association table with an
#                index on batch_id for the delete side and an
#                implicit scan on cache_key for lookups.
#
# retrieve_batch() is a join; free_completed() deletes from both
# tables, scoped to completed batches only.
# ============================================================

_BATCHES_TABLE = "batches"
_BATCH_KEYS_TABLE = "batch_keys"

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_BATCHES_TABLE} (
    batch_id VARCHAR PRIMARY KEY,
    provider_name VARCHAR,
    completed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS {_BATCH_KEYS_TABLE} (
    cache_key VARCHAR,
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

    Completion is a two-phase process, matching BatchStorage's split:
      - complete_batch() marks a batch's row as completed, but leaves
        the row and its batch_keys in place.
      - free_completed() sweeps all batches marked completed and
        deletes them (and their owned batch_keys) in one go, typically
        called at the start of the next session.

    Unlike DuckDBPermacache, this cache is not append-mostly and
    doesn't benefit from write-batching -- record_batch(),
    complete_batch(), and free_completed() are comparatively rare,
    bursty operations (once per batch submission / completion,
    potentially touching thousands of cache_keys at once), so writes
    go straight to disk via a single transaction rather than being
    buffered behind a flush_threshold.
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
                    INSERT INTO {_BATCHES_TABLE} (batch_id, provider_name, completed)
                    VALUES (?, ?, FALSE)
                    ON CONFLICT (batch_id) DO UPDATE
                        SET provider_name = excluded.provider_name
                    """,
                    [batch.batch_id, batch.provider_name],
                )
                if cache_keys:
                    # There's nothing to conflict on anymore, so this is a
                    # plain insert rather than an upsert.
                    self._con.executemany(
                        f"""
                        INSERT INTO {_BATCH_KEYS_TABLE} (cache_key, batch_id)
                        VALUES (?, ?)
                        """,
                        [[cache_key, batch.batch_id] for cache_key in cache_keys],
                    )
                self._con.execute("COMMIT")
            except Exception:
                self._con.execute("ROLLBACK")
                raise

    async def retrieve_batch(self, cache_key: str, likely_partition: str) -> Optional[BatchJob]:
        """
        NOTE: cache_key is no longer unique, so a given cache_key can now be
        owned by more than one in-flight batch at once. When that happens
        this returns an arbitrary (not-completed-preferring) match rather
        than all of them, since the interface only allows one BatchJob back.
        Uncompleted batches are preferred so a still-in-flight owner isn't
        shadowed by one that's already completed and just awaiting sweep.
        """
        with self._lock:
            row = self._con.execute(
                f"""
                SELECT b.batch_id, b.provider_name
                FROM {_BATCH_KEYS_TABLE} bk
                JOIN {_BATCHES_TABLE} b ON b.batch_id = bk.batch_id
                WHERE bk.cache_key = ?
                ORDER BY b.completed ASC
                LIMIT 1
                """,
                [cache_key],
            ).fetchone()
            if row is None:
                return None
            batch_id, provider_name = row
            return BatchJob(batch_id, provider_name)

    async def complete_batch(self, batch: BatchJob):
        """
        Marks `batch` as completed. Does NOT delete its row or its
        batch_keys -- those are only removed later, in bulk, by
        free_completed().
        """
        with self._lock:
            self._con.execute(
                f"UPDATE {_BATCHES_TABLE} SET completed = TRUE WHERE batch_id = ?",
                [batch.batch_id],
            )

    async def free_completed(self):
        """
        Deletes all batches (and their owned batch_keys) that have
        been marked completed via complete_batch().
        """
        with self._lock:
            self._con.execute("BEGIN TRANSACTION")
            try:
                self._con.execute(
                    f"""
                    DELETE FROM {_BATCH_KEYS_TABLE}
                    WHERE batch_id IN (
                        SELECT batch_id FROM {_BATCHES_TABLE} WHERE completed = TRUE
                    )
                    """
                )
                self._con.execute(
                    f"DELETE FROM {_BATCHES_TABLE} WHERE completed = TRUE"
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