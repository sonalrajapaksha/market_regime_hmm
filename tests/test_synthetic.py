import numpy as np

from regime_detection.model import GaussianHMM
from regime_detection.validation import validate_convergence


def test_hmm_recovers_known_synthetic_regimes():
    rng = np.random.default_rng(0)
    transition = np.array([[0.97, 0.03], [0.05, 0.95]])
    states = np.zeros(1200, dtype=int)
    for index in range(1, len(states)):
        states[index] = rng.choice(2, p=transition[states[index - 1]])
    means = np.array([[-0.02, 0.03], [0.015, 0.01]])
    standard_deviations = np.array([[0.015, 0.01], [0.008, 0.005]])
    observations = np.array([rng.normal(means[state], standard_deviations[state]) for state in states])

    model = GaussianHMM(2, n_iter=80, tol=1e-6, random_state=1).fit(observations, verbose=False)
    predicted = model.predict(observations)
    accuracy = max(np.mean(predicted == states), np.mean(predicted == (1 - states)))
    assert accuracy > 0.99
    assert validate_convergence(model.loglik_history_)


def test_edge_cases_are_rejected_or_stable():
    model = GaussianHMM(2)
    with np.testing.assert_raises(ValueError):
        model.fit(np.ones((10, 2)), verbose=False)
    fitted = GaussianHMM(2, n_iter=10).fit(np.random.default_rng(1).normal(size=(20, 2)), verbose=False)
    assert np.isfinite(fitted.predict([[0.1, -0.2]])).all()
    assert np.isfinite(fitted.predict_proba([[1e100, -1e100]])).all()
