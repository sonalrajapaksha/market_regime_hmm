import numpy as np

from regime_detection.model import GaussianHMM
from regime_detection.retraining import evaluate_drift


def test_retraining_drift_gate_uses_saved_baseline():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(300, 2))
    model = GaussianHMM(2, n_iter=10).fit(reference, verbose=False, feature_names=["a", "b"])
    model.metadata = {
        "drift_reference": reference.tolist(),
        "reference_loglik": [model.score(chunk) / len(chunk) for chunk in np.array_split(reference, 6)],
    }
    assert not evaluate_drift(model, reference.copy())["drift_detected"]
    assert evaluate_drift(model, reference + 20)["drift_detected"]
