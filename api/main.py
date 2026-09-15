import os
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from threading import Lock
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from regime_detection.drift import drift_report
from regime_detection.ingestion import consume_stream
from regime_detection.metrics import ACTIVE_MODEL, DRIFT_SCORE, PREDICTION_COUNT, REQUEST_COUNT, REQUEST_LATENCY, metrics_payload
from regime_detection.registry import ModelRegistry

FEATURE_NAMES = ["LogReturn", "Volatility"]
registry = ModelRegistry(os.getenv("MODEL_DIR", "models"))
model = registry.load_latest()
model_lock = Lock()
stream_task = None


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"level": record.levelname, "message": record.getMessage(), "logger": record.name, "time": self.formatTime(record, self.datefmt)})


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)


async def _handle_stream(payload):
    observation = Observation.model_validate(payload)
    _prediction(observation.features, observation.timestamp)


@asynccontextmanager
async def lifespan(_app):
    global stream_task
    ACTIVE_MODEL.set(1 if model is not None else 0)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        stream_task = asyncio.create_task(consume_stream(redis_url, os.getenv("REDIS_STREAM", "market-observations"), os.getenv("REDIS_GROUP", "regime-api"), os.getenv("REDIS_CONSUMER", "api"), _handle_stream))
    yield
    if stream_task:
        stream_task.cancel()
        await asyncio.gather(stream_task, return_exceptions=True)

app = FastAPI(title="Real-time Market Regime Detection", version="1.0.0", lifespan=lifespan)


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
    PREDICTION_COUNT.labels(str(state), active_model.metadata.get("version", "unknown")).inc()
    labels = active_model.metadata.get("state_labels", {})
    return {
        "model_version": active_model.metadata.get("version", "unknown"),
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "state": state,
        "regime": labels.get(str(state), labels.get(state, f"State{state}")),
        "probabilities": probabilities.tolist(),
        "confidence": float(probabilities[state]),
    }


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - started
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(duration)
    return response


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/ready")
async def ready(response: Response):
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False}
    return {"ready": True, "model_version": model.metadata.get("version", "unknown")}


@app.get("/api/v1/model")
async def model_info():
    active_model = _require_model()
    return {"version": active_model.metadata.get("version", "unknown"), "features": active_model.feature_names, "n_states": active_model.n_states}


@app.post("/api/v1/predict")
async def predict(observation: Observation):
    return _prediction(observation.features, observation.timestamp)


@app.post("/api/v1/predict/batch")
async def predict_batch(request: BatchRequest):
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
async def check_drift(request: DriftRequest):
    active_model = _require_model()
    result = drift_report(request.reference, request.current, active_model.feature_names, request.threshold)
    DRIFT_SCORE.set(result["score"])
    return result


@app.post("/api/v1/admin/reload-model")
async def reload_model():
    global model
    loaded = registry.load_latest()
    if loaded is None:
        raise HTTPException(status_code=404, detail="no registered model found")
    with model_lock:
        model = loaded
        ACTIVE_MODEL.set(1)
    return {"reloaded": True, "version": model.metadata.get("version", "unknown")}


@app.get("/metrics")
async def metrics():
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type.split(";", 1)[0])