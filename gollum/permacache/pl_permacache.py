from typing import Optional

from gollum.folder.file_manager import FileManager
from gollum.permacache.base import Permacache
from gollum.types import GollumResponse
from gollum.types.pl_chat_completions import ChatCompletionResponseSchema


import polars as pl


import atexit
import json
import threading
from pathlib import Path


class PolarsPermacache(Permacache):
    """
    Polars-backed permacache with an in-memory mirror and per-key sharding.

    Compared to the naive single-file implementation:
    - Rows are sharded across `num_shards` parquet files by a cheap hash of
      the cache_key (a sha256 hex digest, so the leading hex chars are already
      uniformly distributed). Any single read/write touches only 1/num_shards
      of the data; `likely_partition` is accepted for interface compatibility
      but bucketing is derived from the key itself.
    - Every stored row is mirrored in an in-memory dict, so repeated lookups
      never touch disk. Shards are loaded lazily into the dict, only when a
      miss would otherwise require reading that shard.
    - Writes are batched: rows accumulate in memory and are flushed to their
      shard once `flush_threshold` rows are pending (or on explicit flush()).
      Each flush rewrites only the dirty shards and dedupes by cache_key
      (keeping the latest), so shards stay compact without a separate
      compaction pass.
    - A re-entrant lock makes store/retrieve/flush safe within a process.
      Cross-process safety (file locking / a single writer) is the caller's
      responsibility.
    """

    def __init__(
        self,
        fm: FileManager,
        *,
        num_shards: int = 1,
        flush_threshold: int = 64,
    ):
        super().__init__(fm)
        self._num_shards = num_shards
        self._flush_threshold = flush_threshold
        self._lock = threading.RLock()

        # cache_key -> latest GollumResponse; authoritative for all reads.
        self._cache: dict[str, GollumResponse] = {}
        # Buckets whose parquet files have already been loaded into _cache.
        self._loaded_shards: set[int] = set()
        # Pending rows (dicts) awaiting flush, keyed by bucket.
        self._pending: dict[int, list[dict]] = {}
        self._pending_count = 0

        # Persist any rows still buffered in memory when the process exits
        # (e.g. an interpreter exit before an explicit flush() is called).
        atexit.register(self._flush_at_exit)


    # ---------- paths / bucketing ----------

    @property
    def _shard_dir(self) -> Path:
        return self.fm.path_permacache() / "polars" / "v1"

    def _shard_path(self, bucket: int) -> Path:
        return self._shard_dir / f"{bucket:03d}.parquet"

    def _bucket(self, cache_key: str) -> int:
        # cache_key is a sha256 hex digest; the leading hex chars are a cheap,
        # stable, uniformly distributed hash, so no extra hashing is needed.
        return int(cache_key[:8], 16) % self._num_shards

    # ---------- public API ----------

    async def store(self, response: GollumResponse, cache_key: str, likely_partition: str):
        with self._lock:
            self._cache[cache_key] = response
            self._pending.setdefault(self._bucket(cache_key), []).append({
                "cache_key": cache_key,
                "chat_completion": response.chat_completion,
                "extras": json.dumps(response.extras),
                "metadata": json.dumps(response.metadata),
                # "provider_name": response.provider_name,
                # "original": response.original,
                "original": None,
            })
            self._pending_count += 1
            if self._pending_count >= self._flush_threshold:
                self._flush()

    async def retrieve(self, cache_key: str, likely_partition: str) -> Optional[GollumResponse]:
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            self._load_shard(self._bucket(cache_key))
            return self._cache.get(cache_key)

    async def flush(self):
        """Force any buffered stores out to disk."""
        with self._lock:
            self._flush()

    def _flush_at_exit(self):
        """atexit hook: persist pending rows so buffered stores are never lost."""
        with self._lock:
            self._flush()

    # ---------- internals ----------

    def _flush(self):
        if not self._pending:
            return
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        for bucket, rows in self._pending.items():
            addendum = pl.DataFrame(rows, schema_overrides={
                "cache_key": pl.Utf8,
                "chat_completion": pl.Struct(ChatCompletionResponseSchema),
                "extras": pl.Utf8,
                "metadata": pl.Utf8,
                "original": pl.Utf8,
                # "provider_name": pl.Utf8,
            })
            path = self._shard_path(bucket)
            if path.exists():
                combined = pl.concat(
                    [pl.read_parquet(path), addendum], how="diagonal_relaxed"
                )
            else:
                combined = addendum
            # Only the latest row per key is kept, so files stay compact.
            combined = combined.unique(subset=["cache_key"], keep="last")
            combined.write_parquet(path)
        self._pending = {}
        self._pending_count = 0

    def _load_shard(self, bucket: int):
        if bucket in self._loaded_shards:
            return
        path = self._shard_path(bucket)
        if path.exists():
            for row in pl.read_parquet(path).iter_rows(named=True):
                # _cache is authoritative: it may hold a newer pending value
                # for a key whose older version is still in the file.
                if row["cache_key"] not in self._cache:
                    self._cache[row["cache_key"]] = GollumResponse(
                        row["chat_completion"],
                        json.loads(row["extras"]),
                        json.loads(row["metadata"]),
                        # row["provider_name"],
                        original=row["original"],
                    )
        self._loaded_shards.add(bucket)
