import os
import matplotlib.pyplot as plt

from hmm import GaussianHMM
from data import load_price_data, add_features, train_test_split_by_date
from backtest import label_regimes, run_strategy, performance_metrics
from visualize import (
    plot_price_with_regimes,
    plot_state_probabilities,
    plot_strategy_vs_buyhold,
    plot_transition_matrix,
    plot_loglik_convergence,
)

TICKER = "^GSPC"
START = "2005-01-01"
END = "2024-01-01"
SPLIT_DATE = "2018-01-01"  # train before this date, test (out-of-sample) after
N_STATES = 2  # try 3 for Bull / Bear / Sideways
FEATURE_COLS = ["LogReturn", "Volatility"]

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("1. Loading data...")
    df = load_price_data(TICKER, START, END)
    df = add_features(df)
    train_df, test_df = train_test_split_by_date(df, SPLIT_DATE)
    print(
        f"   Train: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} rows)"
    )
    print(
        f"   Test:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} rows)"
    )

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values
    X_full = df[FEATURE_COLS].values

    print("\n2. Fitting HMM on TRAINING data only (unsupervised EM)...")
    model = GaussianHMM(n_states=N_STATES, n_iter=200, tol=1e-5, random_state=42)
    model.fit(X_train)

    state_labels = label_regimes(model.means, return_col_index=0)
    print(f"\n   Learned state labels: {state_labels}")
    print(f"   Means (LogReturn, Volatility) per state:\n{model.means}")
    print(f"   Transition matrix:\n{model.A}")

    print("\n3. Decoding regimes OUT-OF-SAMPLE on TEST data (model never saw this)...")
    test_df = test_df.copy()
    test_df["State"] = model.predict(X_test)
    test_df["Regime"] = test_df["State"].map(state_labels)

    print("\n4. Backtesting regime-following strategy on TEST data...")
    test_bt = run_strategy(test_df, long_regimes=("Bull",))
    strat_metrics = performance_metrics(test_bt["StrategyReturn"])
    bh_metrics = performance_metrics(test_bt["BuyHoldReturn"])

    print("\n   --- Strategy Performance (out-of-sample, 2018-2024) ---")
    for k, v in strat_metrics.items():
        print(f"   {k:22s}: {v}")
    print("\n   --- Buy & Hold Performance (out-of-sample, 2018-2024) ---")
    for k, v in bh_metrics.items():
        print(f"   {k:22s}: {v}")

    print("\n5. Refitting on FULL history for visualization purposes...")
    full_model = GaussianHMM(n_states=N_STATES, n_iter=200, tol=1e-5, random_state=42)
    full_model.fit(X_full, verbose=False)
    full_labels = label_regimes(full_model.means, return_col_index=0)

    df_viz = df.copy()
    df_viz["State"] = full_model.predict(X_full)
    df_viz["Regime"] = df_viz["State"].map(full_labels)
    full_gamma = full_model.predict_proba(X_full)
    df_viz_bt = run_strategy(df_viz, long_regimes=("Bull",))

    print("\n6. Generating plots...")
    fig1 = plot_price_with_regimes(
        df_viz_bt, title=f"{TICKER} Price with HMM-Detected Regimes (Full History)"
    )
    fig1.savefig(f"{OUT_DIR}/price_regimes.png", dpi=150)

    fig2 = plot_state_probabilities(df_viz, full_gamma, full_labels)
    fig2.savefig(f"{OUT_DIR}/state_probabilities.png", dpi=150)

    fig3 = plot_strategy_vs_buyhold(
        df_viz_bt, title=f"{TICKER}: HMM Strategy vs Buy & Hold (Full History)"
    )
    fig3.savefig(f"{OUT_DIR}/strategy_vs_buyhold.png", dpi=150)

    fig4 = plot_transition_matrix(full_model.A, full_labels)
    fig4.savefig(f"{OUT_DIR}/transition_matrix.png", dpi=150)

    fig5 = plot_loglik_convergence(full_model.loglik_history_)
    fig5.savefig(f"{OUT_DIR}/loglik_convergence.png", dpi=150)

    print(f"\n   Saved 5 plots to ./{OUT_DIR}/")
    plt.show()


if __name__ == "__main__":
    main()
