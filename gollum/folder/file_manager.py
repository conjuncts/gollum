from pathlib import Path


class FileManager:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path_permacache(self) -> Path:
        return self.root / "permacache"
