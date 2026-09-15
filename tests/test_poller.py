import pandas as pd

from jobs import poll


class FakeRedis:
    def __init__(self):
        self.messages = []

    def xadd(self, stream, payload, **_kwargs):
        self.messages.append((stream, payload))


def test_poller_publishes_each_market_bar_once(monkeypatch):
    index = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
    frame = pd.DataFrame({"Close": range(100, 120)}, index=index)
    monkeypatch.setattr(poll, "load_price_data", lambda *_args, **_kwargs: frame)
    redis = FakeRedis()
    payload, observed_at = poll.publish_latest(redis, "prices", "TEST")
    duplicate, _ = poll.publish_latest(redis, "prices", "TEST", last_timestamp=observed_at)
    assert payload["series_id"] == "TEST"
    assert duplicate is None
    assert len(redis.messages) == 1
