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


def test_invalid_features_are_rejected():
    client = TestClient(main.app)
    assert client.post("/api/v1/predict", json={"features": ["bad"]}).status_code == 422