import argparse
import asyncio

from regime_detection.retraining import scheduled_retraining


def main():
    parser = argparse.ArgumentParser(description="Run periodic HMM retraining")
    parser.add_argument("--interval-hours", type=float, default=24)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--states", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(scheduled_retraining(args.interval_hours, model_dir=args.model_dir, ticker=args.ticker, start=args.start, states=args.states))


if __name__ == "__main__":
    main()