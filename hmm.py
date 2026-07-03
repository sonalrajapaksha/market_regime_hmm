import numpy as np


class GaussianHMM:
    def __init__(self, n_states, n_iter=200, tol=1e-5, random_state=42, cov_reg=1e-6):
        """
        n_states   : number of hidden states
        n_iter     : max EM iterations
        tol        : stop early if log-likelihood improves by less than this
        cov_reg    : small constant added to variances for numerical stability
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.cov_reg = cov_reg

        self.pi = None  # shape (n_states,)
        self.A = None  # shape (n_states, n_states)
        self.means = None  # shape (n_states, n_features)
        self.covars = None  # shape (n_states, n_features), diagonal covariance

        self.loglik_history_ = []

    def _init_params(self, X):
        rng = np.random.default_rng(self.random_state)
        T, n_features = X.shape

        self.pi = np.full(self.n_states, 1.0 / self.n_states)

        # Bias toward self-transition: real-world regimes tend to persist.
        off_diag = 0.1 / max(self.n_states - 1, 1)
        self.A = np.full((self.n_states, self.n_states), off_diag)
        np.fill_diagonal(self.A, 0.9)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Initial means: random distinct data points (simple seeding, not full k-means++)
        idx = rng.choice(T, self.n_states, replace=False)
        self.means = X[idx].copy()

        overall_var = X.var(axis=0) + self.cov_reg
        self.covars = np.tile(overall_var, (self.n_states, 1))

    def _emission_prob(self, X):
        T, n_features = X.shape
        B = np.zeros((T, self.n_states))
        for s in range(self.n_states):
            var = self.covars[s] + self.cov_reg
            diff = X - self.means[s]
            exponent = -0.5 * np.sum((diff**2) / var, axis=1)
            norm_const = 1.0 / np.sqrt(((2 * np.pi) ** n_features) * np.prod(var))
            B[:, s] = norm_const * np.exp(exponent)
        return np.clip(B, 1e-300, None)

    def _forward(self, B):
        T, n_states = B.shape
        alpha = np.zeros((T, n_states))
        c = np.zeros(T)

        alpha[0] = self.pi * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]

        return alpha, c

    def _backward(self, B, c):
        T, n_states = B.shape
        beta = np.zeros((T, n_states))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        return beta

    def fit(self, X, verbose=True):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._init_params(X)
        T, n_features = X.shape
        prev_loglik = -np.inf
        self.loglik_history_ = []

        for iteration in range(self.n_iter):
            B = self._emission_prob(X)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)

            # E-step
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                xi[t] = (
                    alpha[t][:, None]
                    * self.A
                    * B[t + 1][None, :]
                    * beta[t + 1][None, :]
                ) / c[t + 1]

            # M-step
            self.pi = gamma[0]
            self.A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
            self.A /= self.A.sum(axis=1, keepdims=True)

            for s in range(self.n_states):
                w = gamma[:, s]
                w_sum = w.sum()
                self.means[s] = (w[:, None] * X).sum(axis=0) / w_sum
                diff = X - self.means[s]
                self.covars[s] = (w[:, None] * diff**2).sum(
                    axis=0
                ) / w_sum + self.cov_reg

            loglik = np.sum(np.log(c))
            self.loglik_history_.append(loglik)

            if verbose and (iteration % 10 == 0 or iteration == self.n_iter - 1):
                print(f"  iter {iteration:3d}  log-likelihood = {loglik:.4f}")

            if abs(loglik - prev_loglik) < self.tol:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break
            prev_loglik = loglik

        return self

    def predict_proba(self, X):
        """Posterior P(state=s at t | all data), shape (T, n_states)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        B = self._emission_prob(X)
        alpha, c = self._forward(B)
        beta = self._backward(B, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def predict(self, X):
        """Viterbi: single most likely state sequence, shape (T,)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        B = self._emission_prob(X)
        T, n_states = B.shape

        log_A = np.log(self.A + 1e-300)
        log_B = np.log(B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        delta = np.zeros((T, n_states))
        psi = np.zeros((T, n_states), dtype=int)

        delta[0] = log_pi + log_B[0]
        for t in range(1, T):
            trans = delta[t - 1][:, None] + log_A  # (n_states_prev, n_states_next)
            psi[t] = np.argmax(trans, axis=0)
            delta[t] = np.max(trans, axis=0) + log_B[t]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def score(self, X):
        """Total log-likelihood of X under current parameters (for model comparison)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        B = self._emission_prob(X)
        _, c = self._forward(B)
        return float(np.sum(np.log(c)))
