import numpy as np


def label_regimes(means, return_col_index=0):
    returns_by_state = means[:, return_col_index]
    bull_state = int(np.argmax(returns_by_state))
    bear_state = int(np.argmin(returns_by_state))

    labels = {}
    for s in range(len(means)):
        if s == bull_state:
            labels[s] = "Bull"
        elif s == bear_state:
            labels[s] = "Bear"
        else:
            labels[s] = f"State{s}"  # for n_states > 2, e.g. "Sideways"
    return labels


def run_strategy(df, regime_col="Regime", long_regimes=("Bull",)):
    df = df.copy()
    df["Position"] = df[regime_col].isin(long_regimes).astype(int)
    df["StrategyReturn"] = df["Position"].shift(1).fillna(0) * df["LogReturn"]
    df["BuyHoldReturn"] = df["LogReturn"]

    df["StrategyCumulative"] = df["StrategyReturn"].cumsum().apply(np.exp)
    df["BuyHoldCumulative"] = df["BuyHoldReturn"].cumsum().apply(np.exp)
    return df


def performance_metrics(returns, periods_per_year=252):
    returns = returns.dropna()
    total_return = np.exp(returns.sum()) - 1
    annual_return = np.exp(returns.mean() * periods_per_year) - 1
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    cumulative = returns.cumsum().apply(np.exp)
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annual_return:.2%}",
        "Annualized Volatility": f"{annual_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
    }
