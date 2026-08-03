import json
import threading

from gollum.folder.file_manager import FileManager
from gollum.types import GollumRequest

import polars as pl

from gollum.types.pl_chat_completions import ChatCompletionRequestSchema


class Permacache:
    """
    A permacache is a cache that stores LLM responses to disk.
    """
    def __init__(self, fm: FileManager):
        self.fm = fm

    def store(self, request: GollumRequest, cache_key: str, likely_partition: str):
        pass

    def retrieve(self, cache_key: str, likely_partition: str) -> GollumRequest:
        pass


class PolarsPermacache(Permacache):
    """
    Naive permacache implementation with polars
    """
    def __init__(self, fm: FileManager):
        super().__init__(fm)
        self._lock = threading.Lock()

    @property
    def _parquet_loc(self):
        return self.fm.path_permacache() / "polars/v1.parquet"

    @property
    def _check_loc(self):
        return self.fm.path_permacache() / "polars/v1.parquet"


    def store(self, request: GollumRequest, cache_key: str, likely_partition: str):
        with self._lock:
            self._parquet_loc.parent.mkdir(parents=True, exist_ok=True)

            # optional: markdirty based on timestamp deposited to
            # now = str(int(time.time()))
            # with open(self._check_loc, "w+") as f:
            #     f.write(now)
            # self.version = now
            addendum = pl.DataFrame({
                "cache_key": [cache_key],
                "chat_completion": [request.request],
                "extras": [json.dumps(request.extras)],
                "metadata": [json.dumps(request.metadata)],
            }, schema_overrides={
                "cache_key": pl.Utf8,
                "chat_completion": ChatCompletionRequestSchema,
                "extras": pl.Utf8,
                "metadata": pl.Utf8
            })
            df = pl.read_parquet(self._parquet_loc)
            combined = pl.concat([df, addendum], how="diagonal_relaxed")
            combined.write_parquet(self._parquet_loc)

    def retrieve(self, cache_key: str, likely_partition: str) -> GollumRequest:
        # self._check_loc.parent.mkdir(parents=True, exist_ok=True)
        # with open(self._check_loc, "r") as f:
        #     check = f.readline()
        #     self.version == check
        with self._lock:
            df = pl.read_parquet(self._parquet_loc)
            df = df.filter(
                pl.col("cache_key") == cache_key
            )
            completion, extras, metadata = df.select(
                "chat_completion", "extras", "metadata"
            ).row(0, named=True)
            return GollumRequest(completion, json.loads(extras), json.loads(metadata))



        
    

