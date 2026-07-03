import matplotlib.pyplot as plt


def plot_price_with_regimes(
    df,
    regime_col="Regime",
    price_col="Close",
    colors=None,
    title="Price with Detected Regimes",
):
    if colors is None:
        colors = {"Bull": "#2ca02c", "Bear": "#d62728", "State2": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df[price_col], color="black", linewidth=0.8, zorder=3)

    regimes = df[regime_col].values
    dates = df.index
    start_idx = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[start_idx]:
            regime = regimes[start_idx]
            ax.axvspan(
                dates[start_idx],
                dates[i - 1],
                color=colors.get(regime, "gray"),
                alpha=0.2,
            )
            start_idx = i

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_state_probabilities(
    df, gamma, state_labels, title="Posterior State Probabilities"
):
    fig, ax = plt.subplots(figsize=(14, 4))
    for s in range(gamma.shape[1]):
        ax.plot(df.index, gamma[:, s], label=state_labels.get(s, f"State {s}"))
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(state | data)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_strategy_vs_buyhold(
    df, title="Strategy vs Buy & Hold (Cumulative Growth of $1)"
):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        df.index, df["StrategyCumulative"], label="HMM Regime Strategy", linewidth=1.5
    )
    ax.plot(
        df.index,
        df["BuyHoldCumulative"],
        label="Buy & Hold",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_ylabel("Cumulative Growth of $1")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_transition_matrix(A, state_labels, title="Learned Transition Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    n = A.shape[0]
    labels = [state_labels.get(s, f"S{s}") for s in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{A[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if A[i, j] > 0.5 else "black",
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_loglik_convergence(
    loglik_history, title="EM Convergence (Log-Likelihood per Iteration)"
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loglik_history, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title(title)
    fig.tight_layout()
    return fig
