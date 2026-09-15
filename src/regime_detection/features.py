import numpy as np
import pandas as pd


FEATURE_NAMES = ["LogReturn", "Volatility"]


def load_price_data(ticker="^GSPC", start="2005-01-01", end=None, period=None, interval="1d"):
    import yfinance as yf

    options = {"progress": False, "auto_adjust": False, "interval": interval}
    if period is not None:
        options["period"] = period
    else:
        options.update({"start": start, "end": end})
    frame = yf.download(ticker, **options)
    if frame.empty:
        raise ValueError(f"no price data returned for {ticker}")
    close = frame["Close"] if "Close" in frame else frame
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename("Close").to_frame().dropna()


def prices_to_features(prices, timestamps=None, volatility_window=10):
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or len(values) < volatility_window + 1:
        raise ValueError(f"prices must contain at least {volatility_window + 1} values")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite positive values")
    if timestamps is not None and len(timestamps) != len(values):
        raise ValueError("timestamps and prices must have the same length")
    index = None
    if timestamps is not None:
        index = pd.to_datetime(timestamps, utc=True, errors="raise")
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError("timestamps must be unique and strictly increasing")
    frame = pd.DataFrame({"Close": values}, index=index)
    frame["LogReturn"] = np.log(frame["Close"] / frame["Close"].shift(1))
    frame["Volatility"] = frame["LogReturn"].rolling(volatility_window).std()
    return frame.dropna().reset_index(drop=True)


def add_features(frame, volatility_window=10):
    if "Close" not in frame:
        raise ValueError("input data must contain a Close column")
    result = frame.copy()
    result["LogReturn"] = np.log(result["Close"] / result["Close"].shift(1))
    result["Volatility"] = result["LogReturn"].rolling(volatility_window).std()
    return result.dropna()


def train_test_split_by_date(frame, split_date):
    train = frame[frame.index < split_date]
    test = frame[frame.index >= split_date]
    if train.empty or test.empty:
        raise ValueError("split_date must produce non-empty train and test sets")
    return train, test
