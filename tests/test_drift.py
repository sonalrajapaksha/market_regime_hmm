import numpy as np

from regime_detection.drift import drift_report, likelihood_drift, psi


def test_identical_distributions_are_not_drifted():
    values = np.random.default_rng(2).normal(size=(100, 2))
    result = drift_report(values, values.copy(), ["a", "b"])
    assert not result["drift_detected"]


def test_shifted_distribution_is_drifted():
    reference = np.zeros((100, 1))
    current = np.ones((100, 1))
    assert drift_report(reference, current, ["x"])["drift_detected"]


def test_likelihood_drop_is_drifted():
    result = likelihood_drift([10, 11, 9, 10], [3, 4])
    assert result["drifted"]
    assert drift_report([[0], [1]], [[0], [1]], ["x"], reference_loglik=[10, 11, 9, 10], current_loglik=[3, 4])[
        "drift_detected"
    ]


def test_psi_counts_values_outside_reference_range():
    reference = np.arange(100, dtype=float)
    assert psi(reference, reference + 1000) > 0.2
