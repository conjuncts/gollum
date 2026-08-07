from typing import Optional

from gollum.folder.file_manager import FileManager
from gollum.permacache.base import Permacache
from gollum.types import GollumResponse
from gollum.types.pl_chat_completions import ChatCompletionResponseSchema

import polars as pl
import duckdb

import atexit
import json
import threading
from pathlib import Path


# ============================================================
# Polars schema -> DuckDB DDL
#
# Keeps the Polars schema as ground truth (same one PolarsPermacache
# uses) and derives the DuckDB column types from it, so the two
# backends never drift apart. Only the primitive/List/Struct dtypes
# actually used in ChatCompletionResponseSchema are handled; anything
# else raises loudly rather than silently guessing.
# ============================================================

_PRIMITIVE_MAP = {
    pl.Utf8: "VARCHAR",
    pl.Boolean: "BOOLEAN",
    pl.Int8: "TINYINT",
    pl.Int16: "SMALLINT",
    pl.Int32: "INTEGER",
    pl.Int64: "BIGINT",
    pl.UInt8: "UTINYINT",
    pl.UInt16: "USMALLINT",
    pl.UInt32: "UINTEGER",
    pl.UInt64: "UBIGINT",
    pl.Float32: "FLOAT",
    pl.Float64: "DOUBLE",
}


def _pl_dtype_to_duckdb(dtype: pl.DataType) -> str:
    """Recursively map a Polars dtype (as used in pl_chat_completions.py) to
    the equivalent DuckDB type string, so STRUCT/LIST nesting round-trips."""
    for pl_type, duck_type in _PRIMITIVE_MAP.items():
        if dtype == pl_type:
            return duck_type
    if isinstance(dtype, pl.List):
        return f"{_pl_dtype_to_duckdb(dtype.inner)}[]"
    if isinstance(dtype, pl.Struct):
        fields = ", ".join(
            f'"{f.name}" {_pl_dtype_to_duckdb(f.dtype)}' for f in dtype.fields
        )
        return f"STRUCT({fields})"
    raise TypeError(f"No DuckDB mapping defined for polars dtype: {dtype!r}")


def pl_schema_to_duckdb_ddl(
    schema: pl.Schema, table_name: str, *, primary_key: Optional[str] = None
) -> str:
    """CREATE TABLE ... statement whose columns mirror `schema` 1:1.

    `primary_key`, if given, must name a column in `schema`; DuckDB builds
    an ART index on it, giving real O(log n) point lookups (vs. Parquet's
    full-shard scan).
    """
    cols = []
    for name, dtype in schema.items():
        col_sql = f'"{name}" {_pl_dtype_to_duckdb(dtype)}'
        if primary_key == name:
            col_sql += " PRIMARY KEY"
        cols.append(col_sql)
    cols_sql = ",\n    ".join(cols)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {cols_sql}\n)"


# The on-disk row shape: cache_key + the same payload fields
# PolarsPermacache writes, all derived from the same schema module.
_CACHE_TABLE = "cache"
_CACHE_SCHEMA = pl.Schema({
    "cache_key": pl.Utf8,
    "chat_completion": pl.Struct(ChatCompletionResponseSchema),
    "extras": pl.Utf8,
    "metadata": pl.Utf8,
    "original": pl.Utf8,
})
_PAYLOAD_COLUMNS = [c for c in _CACHE_SCHEMA if c != "cache_key"]
_UPSERT_SET_CLAUSE = ", ".join(f"{c} = excluded.{c}" for c in _PAYLOAD_COLUMNS)


class DuckDBPermacache(Permacache):
    """
    DuckDB-backed permacache: same on-disk shape and columns as
    PolarsPermacache, but stored in a single native `.duckdb` file with
    `cache_key` declared PRIMARY KEY. DuckDB builds an ART index on that
    column, so `retrieve()` is a real indexed point lookup (~O(log n),
    effectively O(1) in practice) instead of PolarsPermacache's
    read-the-whole-shard-into-memory approach.

    Compared to PolarsPermacache:
    - No manual sharding: DuckDB's own file format handles large single
      files well, and it does its own block-level compression (dictionary
      encoding, RLE, FSST, bit-packing) comparable to Parquet's.
    - No manual "concat + unique(keep='last')" dedup pass: upserts are
      expressed directly as `INSERT ... ON CONFLICT (cache_key) DO UPDATE`.
    - Writes are still batched in memory up to `flush_threshold` and
      flushed as one vectorized upsert (via a Polars DataFrame handed to
      DuckDB through its zero-copy Arrow/Polars interop), rather than
      row-by-row inserts.
    - Rows written but not yet flushed are kept in `_pending_index` so
      `retrieve()` sees them immediately, same as PolarsPermacache's
      in-memory mirror -- just scoped to the unflushed batch instead of
      the whole cache, since DuckDB itself is now fast enough to serve
      everything else.

    You can still export to Parquet for archival/portability at any time
    with `COPY cache TO 'archive.parquet'` on `self._con`.
    """

    def __init__(
        self,
        fm: FileManager,
        *,
        flush_threshold: int = 64,
    ):
        super().__init__(fm)
        self._flush_threshold = flush_threshold
        self._lock = threading.RLock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self._db_path))
        self._con.execute(
            pl_schema_to_duckdb_ddl(_CACHE_SCHEMA, _CACHE_TABLE, primary_key="cache_key")
        )

        # Rows accumulated since the last flush, keyed by cache_key, so
        # retrieve() can see just-stored values before they hit disk.
        self._pending_rows: list[dict] = []
        self._pending_index: dict[str, GollumResponse] = {}

        atexit.register(self._flush_at_exit)

    # ---------- paths ----------

    @property
    def _db_path(self) -> Path:
        return self.fm.path_permacache() / "duckdb" / "v1" / "cache.duckdb"

    # ---------- public API ----------

    async def store(self, response: GollumResponse, cache_key: str, likely_partition: str):
        with self._lock:
            self._pending_index[cache_key] = response
            self._pending_rows.append({
                "cache_key": cache_key,
                "chat_completion": response.chat_completion,
                "extras": json.dumps(response.extras),
                "metadata": json.dumps(response.metadata),
                "original": None,
            })
            if len(self._pending_rows) >= self._flush_threshold:
                self._flush()

    async def retrieve(self, cache_key: str, likely_partition: str) -> Optional[GollumResponse]:
        with self._lock:
            if cache_key in self._pending_index:
                return self._pending_index[cache_key]
            row = self._con.execute(
                f"SELECT chat_completion, extras, metadata, original "
                f"FROM {_CACHE_TABLE} WHERE cache_key = ?",
                [cache_key],
            ).fetchone()
            if row is None:
                return None
            chat_completion, extras, metadata, original = row
            return GollumResponse(
                chat_completion,
                json.loads(extras),
                json.loads(metadata),
                original=original,
            )

    async def flush(self):
        """Force any buffered stores out to disk."""
        with self._lock:
            self._flush()

    def close(self):
        with self._lock:
            self._flush()
            self._con.close()

    def _flush_at_exit(self):
        """atexit hook: persist pending rows so buffered stores are never lost."""
        with self._lock:
            self._flush()

    # ---------- internals ----------

    def _flush(self):
        if not self._pending_rows:
            return
        # `batch_df` is picked up by DuckDB's replacement scan (it inspects
        # the calling frame for a matching local/global name) -- no need to
        # register it explicitly.
        batch_df = pl.DataFrame(self._pending_rows, schema_overrides=_CACHE_SCHEMA)  # noqa: F841
        self._con.execute(
            f"INSERT INTO {_CACHE_TABLE} SELECT * FROM batch_df "
            f"ON CONFLICT (cache_key) DO UPDATE SET {_UPSERT_SET_CLAUSE}"
        )
        self._pending_rows = []
        self._pending_index = {}
