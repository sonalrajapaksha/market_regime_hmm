import numpy as np
import pytest

from regime_detection.model import GaussianHMM
from regime_detection.registry import ModelRegistry


def test_registry_promotes_and_rejects_duplicate_versions(tmp_path):
    model = GaussianHMM(2, n_iter=5).fit(np.random.default_rng(12).normal(size=(30, 2)), verbose=False)
    registry = ModelRegistry(tmp_path)
    registry.save(model, "v1", promote=True)
    assert registry.load_latest().metadata["version"] == "v1"
    with pytest.raises(FileExistsError):
        registry.save(model, "v1")
