import argparse
import json
import logging
import time
from datetime import datetime, timezone

from redis import Redis
from redis.exceptions import RedisError

from regime_detection.features import FEATURE_NAMES, add_features, load_price_data

logger = logging.getLogger(__name__)


def publish_latest(redis_client, stream, ticker):
    frame = add_features(load_price_data(ticker, start="30 days ago"))
    latest = frame.iloc[-1]
    payload = {"features": [float(latest[name]) for name in FEATURE_NAMES], "timestamp": datetime.now(timezone.utc).isoformat()}
    redis_client.xadd(stream, {"payload": json.dumps(payload)}, maxlen=10000, approximate=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Poll market prices and publish Redis Stream observations")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--stream", default="market-observations")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    client = Redis.from_url(args.redis_url)
    while True:
        try:
            publish_latest(client, args.stream, args.ticker)
        except (ConnectionError, KeyError, RedisError, TimeoutError, ValueError) as error:
            logger.warning("poll failed", extra={"error": str(error)})
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()