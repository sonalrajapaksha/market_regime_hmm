import argparse
import json
import logging
import time
from datetime import timezone

from redis import Redis
from redis.exceptions import RedisError

from regime_detection.features import FEATURE_NAMES, add_features, load_price_data

logger = logging.getLogger(__name__)


def publish_latest(redis_client, stream, ticker, period="5d", interval="5m", last_timestamp=None):
    frame = add_features(load_price_data(ticker, period=period, interval=interval))
    latest = frame.iloc[-1]
    observed_at = frame.index[-1]
    if last_timestamp is not None and observed_at <= last_timestamp:
        return None, last_timestamp
    timestamp = observed_at.to_pydatetime() if hasattr(observed_at, "to_pydatetime") else observed_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    payload = {
        "features": [float(latest[name]) for name in FEATURE_NAMES],
        "timestamp": timestamp.isoformat(),
        "series_id": ticker,
    }
    redis_client.xadd(stream, {"payload": json.dumps(payload)}, maxlen=10000, approximate=True)
    return payload, observed_at


def main():
    parser = argparse.ArgumentParser(description="Poll market prices and publish Redis Stream observations")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--stream", default="market-observations")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--period", default="5d")
    parser.add_argument("--market-interval", default="5m")
    args = parser.parse_args()
    client = Redis.from_url(args.redis_url)
    last_timestamp = None
    while True:
        try:
            _, last_timestamp = publish_latest(
                client, args.stream, args.ticker, args.period, args.market_interval, last_timestamp
            )
        except (ConnectionError, KeyError, RedisError, TimeoutError, ValueError) as error:
            logger.warning("poll failed", extra={"error": str(error)})
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
