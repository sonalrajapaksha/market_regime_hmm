import numpy as np

from regime_detection.drift import drift_report


def test_identical_distributions_are_not_drifted():
    values = np.random.default_rng(2).normal(size=(100, 2))
    result = drift_report(values, values.copy(), ["a", "b"])
    assert not result["drift_detected"]


def test_shifted_distribution_is_drifted():
    reference = np.zeros((100, 1))
    current = np.ones((100, 1))
    assert drift_report(reference, current, ["x"])["drift_detected"]