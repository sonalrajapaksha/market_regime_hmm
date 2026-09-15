import numpy as np


def psi(reference, current, bins=10):
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        scale = max(float(np.std(reference)), 1e-12)
        return 0.0 if abs(float(np.mean(current) - np.mean(reference))) <= scale else 1.0
    expected, _ = np.histogram(reference, bins=edges)
    actual, _ = np.histogram(current, bins=edges)
    expected = (expected + 1e-6) / (expected.sum() + 1e-6 * len(expected))
    actual = (actual + 1e-6) / (actual.sum() + 1e-6 * len(actual))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(reference, current, feature_names=None, threshold=0.2):
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    if reference.ndim != 2 or current.ndim != 2 or reference.shape[1] != current.shape[1]:
        raise ValueError("reference and current must be 2D arrays with matching features")
    names = feature_names or [f"feature_{i}" for i in range(reference.shape[1])]
    features = {name: {"score": psi(reference[:, i], current[:, i]), "drifted": False} for i, name in enumerate(names)}
    for result in features.values():
        result["drifted"] = result["score"] >= threshold
    score = max((result["score"] for result in features.values()), default=0.0)
    return {"drift_detected": score >= threshold, "score": score, "threshold": threshold, "features": features}