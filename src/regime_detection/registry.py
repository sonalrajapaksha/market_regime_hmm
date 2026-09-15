from pathlib import Path

from .model import GaussianHMM


class ModelRegistry:
    def __init__(self, root="models"):
        self.root = Path(root)

    def load_latest(self):
        pointer = self.root / "latest"
        return GaussianHMM.load(pointer.resolve()) if pointer.exists() else None

    def save(self, model, version, metadata=None, promote=True):
        directory = self.root / version
        if directory.exists():
            raise FileExistsError(f"model version already exists: {version}")
        model.save(directory, {"version": version, **(metadata or {})})
        if promote:
            self.promote(version)
        return directory

    def promote(self, version):
        directory = self.root / version
        if not directory.is_dir():
            raise FileNotFoundError(f"model version does not exist: {version}")
        latest = self.root / "latest"
        temporary = self.root / ".latest.tmp"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(directory.name)
        temporary.replace(latest)

    def versions(self):
        return (
            sorted(path.name for path in self.root.iterdir() if path.is_dir() and path.name != "latest")
            if self.root.exists()
            else []
        )
