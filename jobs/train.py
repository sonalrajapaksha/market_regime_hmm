import argparse
from regime_detection.retraining import train_and_register


def train(ticker="^GSPC", start="2005-01-01", end="2024-01-01", model_dir="models", states=2):
    version = train_and_register(ticker, start, end, model_dir, states)
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