import numpy as np
from hmm import GaussianHMM

rng = np.random.default_rng(0)

# ---- Generate synthetic 2-state data ----
true_A = np.array([[0.97, 0.03], [0.05, 0.95]])
true_pi = np.array([0.5, 0.5])
true_means = np.array(
    [[-0.02, 0.03], [0.015, 0.01]]
)  # state 0 = "bear-like", state 1 = "bull-like"
true_stds = np.array([[0.015, 0.01], [0.008, 0.005]])

T = 3000
states = np.zeros(T, dtype=int)
states[0] = rng.choice(2, p=true_pi)
for t in range(1, T):
    states[t] = rng.choice(2, p=true_A[states[t - 1]])

X = np.zeros((T, 2))
for t in range(T):
    s = states[t]
    X[t] = rng.normal(true_means[s], true_stds[s])

# ---- Fit our HMM ----
print("Fitting GaussianHMM on synthetic data...")
model = GaussianHMM(n_states=2, n_iter=150, tol=1e-6, random_state=1)
model.fit(X)

pred_states = model.predict(X)

# States are unlabeled / order may be swapped vs. ground truth — align them.
# Simple fix: check both possible label mappings and take whichever matches better.
acc_direct = np.mean(pred_states == states)
acc_swapped = np.mean(pred_states == (1 - states))
best_acc = max(acc_direct, acc_swapped)
swapped = acc_swapped > acc_direct

print(f"\nRecovered means:\n{model.means}")
print(f"True means:\n{true_means}")
print(f"\nRecovered transition matrix:\n{model.A}")
print(f"True transition matrix:\n{true_A}")

print(f"\nViterbi state-recovery accuracy vs ground truth: {best_acc:.2%}")
print("(Label order may differ — this is normal and accounted for above.)")

assert best_acc > 0.85, (
    "HMM failed to recover states above 85% accuracy — check implementation."
)
print("\nPASSED: implementation recovers hidden states from synthetic data correctly.")
