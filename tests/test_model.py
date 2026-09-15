import numpy as np
import pytest

from regime_detection.model import GaussianHMM


def test_hmm_probabilities_and_likelihood_are_finite():
    rng = np.random.default_rng(7)
    observations = np.r_[rng.normal(-1, 0.2, (80, 2)), rng.normal(1, 0.2, (80, 2))]
    model = GaussianHMM(2, n_iter=20).fit(observations, verbose=False)
    probabilities = model.predict_proba(observations)
    assert np.isfinite(model.loglik_history_).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(model.score(observations))


def test_model_rejects_nan_and_unfitted_prediction():
    model = GaussianHMM(2)
    with pytest.raises(RuntimeError):
        model.predict([[1.0]])
    with pytest.raises(ValueError):
        model.fit([[1.0, np.nan], [2.0, 3.0]])