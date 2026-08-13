from pathlib import Path
from typing import Union


class FileManager:
    def __init__(self, root: Union[Path, str]):
        self.root = Path(root)

    def path_permacache(self) -> Path:
        return self.root / "permacache"

    def path_batchcache(self) -> Path:
        return self.root / "batchcache"
