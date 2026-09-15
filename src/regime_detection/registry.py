from pathlib import Path

from .model import GaussianHMM


class ModelRegistry:
    def __init__(self, root="models"):
        self.root = Path(root)

    def load_latest(self):
        pointer = self.root / "latest"
        return GaussianHMM.load(pointer.resolve()) if pointer.exists() else None

    def save(self, model, version, metadata=None):
        directory = self.root / version
        model.save(directory, {"version": version, **(metadata or {})})
        latest = self.root / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(directory.name)
        return directory