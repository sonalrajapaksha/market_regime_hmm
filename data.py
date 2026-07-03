import numpy as np
import yfinance as yf


def load_price_data(ticker="^GSPC", start="2005-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Close"]].dropna()
    df.columns = ["Close"]
    return df


def add_features(df):
    df = df.copy()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["LogReturn"].rolling(10).std()
    df = df.dropna()
    return df


def train_test_split_by_date(df, split_date):
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    return train, test
