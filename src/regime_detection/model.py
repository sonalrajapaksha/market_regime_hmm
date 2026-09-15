import json
from pathlib import Path

import numpy as np


class GaussianHMM:
    def __init__(self, n_states, n_iter=200, tol=1e-5, random_state=42, cov_reg=1e-6):
        if n_states < 1:
            raise ValueError("n_states must be positive")
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.cov_reg = cov_reg
        self.pi = self.A = self.means = self.covars = None
        self.loglik_history_ = []
        self.feature_names = None
        self.metadata = {}
        self._filtered_state = None

    @property
    def fitted_(self):
        return self.pi is not None

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if self.fitted_ and X.shape[1] != self.means.shape[1]:
            raise ValueError(f"expected {self.means.shape[1]} features, got {X.shape[1]}")
        return X

    def _require_fitted(self):
        if not self.fitted_:
            raise RuntimeError("model has not been fitted")

    def _init_params(self, X):
        if len(X) < self.n_states:
            raise ValueError("X must contain at least n_states observations")
        rng = np.random.default_rng(self.random_state)
        self.pi = np.full(self.n_states, 1 / self.n_states)
        off_diag = 0.1 / max(self.n_states - 1, 1)
        self.A = np.full((self.n_states, self.n_states), off_diag)
        np.fill_diagonal(self.A, 0.9)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.means = X[rng.choice(len(X), self.n_states, replace=False)].copy()
        self.covars = np.tile(X.var(axis=0) + self.cov_reg, (self.n_states, 1))

    def _emission_prob(self, X):
        self._require_fitted()
        n_features = X.shape[1]
        result = np.zeros((len(X), self.n_states))
        for state in range(self.n_states):
            variance = self.covars[state] + self.cov_reg
            diff = X - self.means[state]
            exponent = -0.5 * np.sum(diff**2 / variance, axis=1)
            normalizer = 1 / np.sqrt(((2 * np.pi) ** n_features) * np.prod(variance))
            result[:, state] = normalizer * np.exp(exponent)
        return np.clip(result, 1e-300, None)

    def _forward(self, emissions):
        alpha = np.zeros_like(emissions)
        scales = np.zeros(len(emissions))
        alpha[0] = self.pi * emissions[0]
        scales[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scales[0]
        for index in range(1, len(emissions)):
            alpha[index] = (alpha[index - 1] @ self.A) * emissions[index]
            scales[index] = max(alpha[index].sum(), 1e-300)
            alpha[index] /= scales[index]
        return alpha, scales

    def _backward(self, emissions, scales):
        beta = np.zeros_like(emissions)
        beta[-1] = 1.0
        for index in range(len(emissions) - 2, -1, -1):
            beta[index] = (self.A @ (emissions[index + 1] * beta[index + 1])) / scales[index + 1]
        return beta

    def fit(self, X, verbose=True, feature_names=None):
        X = self._validate_X(X)
        self._init_params(X)
        self.feature_names = list(feature_names) if feature_names else None
        self.loglik_history_ = []
        previous = -np.inf
        for iteration in range(self.n_iter):
            emissions = self._emission_prob(X)
            alpha, scales = self._forward(emissions)
            beta = self._backward(emissions, scales)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = np.zeros((len(X) - 1, self.n_states, self.n_states))
            for index in range(len(X) - 1):
                xi[index] = (alpha[index][:, None] * self.A * emissions[index + 1][None, :] * beta[index + 1][None, :]) / scales[index + 1]
            self.pi = gamma[0]
            self.A = xi.sum(axis=0) / np.maximum(gamma[:-1].sum(axis=0)[:, None], 1e-300)
            self.A /= np.maximum(self.A.sum(axis=1, keepdims=True), 1e-300)
            for state in range(self.n_states):
                weights = gamma[:, state]
                total = max(weights.sum(), 1e-300)
                self.means[state] = (weights[:, None] * X).sum(axis=0) / total
                difference = X - self.means[state]
                self.covars[state] = (weights[:, None] * difference**2).sum(axis=0) / total + self.cov_reg
            loglik = float(np.log(scales).sum())
            self.loglik_history_.append(loglik)
            if verbose and (iteration % 10 == 0 or iteration == self.n_iter - 1):
                print(f"  iter {iteration:3d}  log-likelihood = {loglik:.4f}")
            if abs(loglik - previous) < self.tol:
                break
            previous = loglik
        self._filtered_state = None
        return self

    def predict_proba(self, X):
        X = self._validate_X(X)
        emissions = self._emission_prob(X)
        alpha, scales = self._forward(emissions)
        posterior = alpha * self._backward(emissions, scales)
        return posterior / posterior.sum(axis=1, keepdims=True)

    def filter(self, X, reset=True):
        X = self._validate_X(X)
        state = self.pi.copy() if reset or self._filtered_state is None else self._filtered_state.copy()
        probabilities = []
        for observation in X:
            emission = self._emission_prob(observation.reshape(1, -1))[0]
            state = (state @ self.A) * emission
            state /= max(state.sum(), 1e-300)
            probabilities.append(state.copy())
        self._filtered_state = state
        return np.asarray(probabilities)

    def predict(self, X):
        X = self._validate_X(X)
        emissions = self._emission_prob(X)
        log_A, log_B, log_pi = np.log(self.A + 1e-300), np.log(emissions + 1e-300), np.log(self.pi + 1e-300)
        delta = np.zeros_like(log_B)
        backpointers = np.zeros((len(X), self.n_states), dtype=int)
        delta[0] = log_pi + log_B[0]
        for index in range(1, len(X)):
            transitions = delta[index - 1][:, None] + log_A
            backpointers[index] = np.argmax(transitions, axis=0)
            delta[index] = np.max(transitions, axis=0) + log_B[index]
        states = np.zeros(len(X), dtype=int)
        states[-1] = np.argmax(delta[-1])
        for index in range(len(X) - 2, -1, -1):
            states[index] = backpointers[index + 1, states[index + 1]]
        return states

    def score(self, X):
        X = self._validate_X(X)
        _, scales = self._forward(self._emission_prob(X))
        return float(np.log(scales).sum())

    def save(self, directory, metadata=None):
        self._require_fitted()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "parameters.npz", pi=self.pi, A=self.A, means=self.means, covars=self.covars)
        payload = {"n_states": self.n_states, "n_iter": self.n_iter, "tol": self.tol, "cov_reg": self.cov_reg, "feature_names": self.feature_names, **self.metadata, **(metadata or {})}
        (path / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(metadata["n_states"], metadata.get("n_iter", 200), metadata.get("tol", 1e-5), cov_reg=metadata.get("cov_reg", 1e-6))
        parameters = np.load(path / "parameters.npz")
        model.pi, model.A, model.means, model.covars = parameters["pi"], parameters["A"], parameters["means"], parameters["covars"]
        model.feature_names, model.metadata = metadata.get("feature_names"), metadata
        return model