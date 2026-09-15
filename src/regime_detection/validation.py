import numpy as np


def validate_training_data(X, n_states):
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or len(values) < max(n_states * 2, 3):
        raise ValueError("training data must be 2D and contain enough observations")
    if not np.isfinite(values).all():
        raise ValueError("training data contains NaN or infinite values")
    variances = values.var(axis=0)
    if (variances <= 0).any():
        raise ValueError("training data contains a zero-variance feature")


def validate_convergence(loglik_history, tolerance=1e-8):
    values = np.asarray(loglik_history, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return False
    return bool(np.all(np.diff(values) >= -tolerance))