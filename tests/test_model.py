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


def test_filter_starts_with_initial_distribution_and_can_be_stateful():
    rng = np.random.default_rng(8)
    model = GaussianHMM(2, n_iter=10).fit(rng.normal(size=(50, 2)), verbose=False)
    observation = np.array([[0.2, -0.1]])
    emission = model._emission_prob(observation)[0]
    expected = model.pi * emission
    expected /= expected.sum()
    probabilities, state = model.filter_from_state(observation)
    assert np.allclose(probabilities[0], expected)
    continued, _ = model.filter_from_state(observation, state)
    assert not np.allclose(continued[0], probabilities[0])


def test_model_round_trip_and_corrupt_parameters(tmp_path):
    model = GaussianHMM(2, n_iter=10).fit(np.random.default_rng(9).normal(size=(50, 2)), verbose=False)
    model.save(tmp_path)
    loaded = GaussianHMM.load(tmp_path)
    assert np.allclose(model.predict_proba([[0.1, 0.2]]), loaded.predict_proba([[0.1, 0.2]]))
    np.savez(tmp_path / "parameters.npz", pi=[1, 0], A=np.eye(2), means=np.zeros((2, 2)), covars=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        GaussianHMM.load(tmp_path)
