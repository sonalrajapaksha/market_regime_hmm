import numpy as np
from fastapi.testclient import TestClient

from regime_detection.model import GaussianHMM
from api import main


def setup_module():
    rng = np.random.default_rng(4)
    observations = np.r_[rng.normal(-1, 0.2, (30, 2)), rng.normal(1, 0.2, (30, 2))]
    main.model = GaussianHMM(2, n_iter=10).fit(observations, verbose=False, feature_names=["a", "b"])
    main.model.metadata = {"version": "test", "state_labels": {"0": "Bear", "1": "Bull"}}


def test_health_and_prediction():
    client = TestClient(main.app)
    assert client.get("/api/v1/health").status_code == 200
    response = client.post("/api/v1/predict", json={"features": [0.1, 0.2]})
    assert response.status_code == 200
    assert len(response.json()["probabilities"]) == 2
    assert response.json()["latency_ms"] >= 0


def test_invalid_features_are_rejected():
    client = TestClient(main.app)
    assert client.post("/api/v1/predict", json={"features": ["bad"]}).status_code == 400
    assert client.post("/api/v1/predict", json={"features": [0.1]}).status_code == 400
    assert client.post("/api/v1/predict", json={"features": [0.1, 0.2, 0.3]}).status_code == 400


def test_raw_prices_are_converted_to_predictions():
    client = TestClient(main.app)
    response = client.post("/api/v1/predict/prices", json={"prices": list(range(1, 13))})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_raw_prices_require_ordered_unique_timestamps():
    client = TestClient(main.app)
    timestamps = [f"2026-01-{day:02d}T00:00:00Z" for day in range(1, 13)]
    timestamps[-1] = timestamps[-2]
    response = client.post("/api/v1/predict/prices", json={"prices": list(range(1, 13)), "timestamps": timestamps})
    assert response.status_code == 400


def test_stateless_requests_do_not_share_filter_state():
    client = TestClient(main.app)
    payload = {"features": [0.1, 0.2]}
    first = client.post("/api/v1/predict", json=payload).json()["probabilities"]
    second = client.post("/api/v1/predict", json=payload).json()["probabilities"]
    assert np.allclose(first, second)


def test_series_requests_keep_separate_state():
    client = TestClient(main.app)
    first = client.post("/api/v1/predict", json={"features": [0.1, 0.2], "series_id": "one"}).json()
    continued = client.post("/api/v1/predict", json={"features": [0.1, 0.2], "series_id": "one"}).json()
    separate = client.post("/api/v1/predict", json={"features": [0.1, 0.2], "series_id": "two"}).json()
    assert not np.allclose(first["probabilities"], continued["probabilities"])
    assert np.allclose(first["probabilities"], separate["probabilities"])


def test_admin_reload_is_disabled_without_token(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert TestClient(main.app).post("/api/v1/admin/reload-model").status_code == 503
