import os
from datetime import datetime, timezone
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from regime_detection.drift import drift_report
from regime_detection.registry import ModelRegistry

FEATURE_NAMES = ["LogReturn", "Volatility"]
registry = ModelRegistry(os.getenv("MODEL_DIR", "models"))
model = registry.load_latest()
model_lock = Lock()

app = FastAPI(title="Real-time Market Regime Detection", version="1.0.0")


class Observation(BaseModel):
    features: list[float] = Field(..., min_length=1)
    timestamp: datetime | None = None


class BatchRequest(BaseModel):
    observations: list[Observation] = Field(..., min_length=1)
    reset: bool = True


class DriftRequest(BaseModel):
    reference: list[list[float]] = Field(..., min_length=2)
    current: list[list[float]] = Field(..., min_length=2)
    threshold: float = Field(0.2, gt=0)


def _require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="no model is loaded")
    return model


def _prediction(features, timestamp=None):
    active_model = _require_model()
    with model_lock:
        probabilities = active_model.filter(np.asarray([features]), reset=False)[0]
    state = int(np.argmax(probabilities))
    labels = active_model.metadata.get("state_labels", {})
    return {
        "model_version": active_model.metadata.get("version", "unknown"),
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "state": state,
        "regime": labels.get(str(state), labels.get(state, f"State{state}")),
        "probabilities": probabilities.tolist(),
        "confidence": float(probabilities[state]),
    }


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/ready")
def ready(response: Response):
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False}
    return {"ready": True, "model_version": model.metadata.get("version", "unknown")}


@app.get("/api/v1/model")
def model_info():
    active_model = _require_model()
    return {"version": active_model.metadata.get("version", "unknown"), "features": active_model.feature_names, "n_states": active_model.n_states}


@app.post("/api/v1/predict")
def predict(observation: Observation):
    return _prediction(observation.features, observation.timestamp)


@app.post("/api/v1/predict/batch")
def predict_batch(request: BatchRequest):
    active_model = _require_model()
    with model_lock:
        probabilities = active_model.filter(np.asarray([item.features for item in request.observations]), reset=request.reset)
    labels = active_model.metadata.get("state_labels", {})
    results = []
    for item, values in zip(request.observations, probabilities):
        state = int(np.argmax(values))
        results.append({"model_version": active_model.metadata.get("version", "unknown"), "timestamp": (item.timestamp or datetime.now(timezone.utc)).isoformat(), "state": state, "regime": labels.get(str(state), f"State{state}"), "probabilities": values.tolist(), "confidence": float(values[state])})
    return {"predictions": results}


@app.post("/api/v1/drift/check")
def check_drift(request: DriftRequest):
    active_model = _require_model()
    return drift_report(request.reference, request.current, active_model.feature_names, request.threshold)


@app.post("/api/v1/admin/reload-model")
def reload_model():
    global model
    loaded = registry.load_latest()
    if loaded is None:
        raise HTTPException(status_code=404, detail="no registered model found")
    with model_lock:
        model = loaded
    return {"reloaded": True, "version": model.metadata.get("version", "unknown")}