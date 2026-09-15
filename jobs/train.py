import argparse
from datetime import datetime, timezone

from data import add_features, load_price_data
from regime_detection.registry import ModelRegistry
from regime_detection.model import GaussianHMM


def train(ticker="^GSPC", start="2005-01-01", end="2024-01-01", model_dir="models", states=2):
    frame = add_features(load_price_data(ticker, start, end))
    features = ["LogReturn", "Volatility"]
    model = GaussianHMM(states, n_iter=200, tol=1e-5, random_state=42)
    model.fit(frame[features].values, verbose=False, feature_names=features)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {"ticker": ticker, "training_start": start, "training_end": end, "state_labels": {str(i): f"State{i}" for i in range(states)}}
    ModelRegistry(model_dir).save(model, version, metadata)
    print(f"registered model {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--states", type=int, default=2)
    args = parser.parse_args()
    train(args.ticker, args.start, args.end, args.model_dir, args.states)