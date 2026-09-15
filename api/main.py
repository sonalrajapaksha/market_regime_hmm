import os
import asyncio
import json
import logging
import secrets
import time
from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from regime_detection.drift import drift_report
from regime_detection.features import prices_to_features
from regime_detection.ingestion import consume_stream
from regime_detection.metrics import (
    ACTIVE_MODEL,
    DRIFT_SCORE,
    PREDICTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    metrics_payload,
)
from regime_detection.monitoring import record_prediction
from regime_detection.registry import ModelRegistry

FEATURE_NAMES = ["LogReturn", "Volatility"]
registry = ModelRegistry(os.getenv("MODEL_DIR", "models"))
model = registry.load_latest()
model_lock = Lock()
stream_task = None
reload_task = None
sequence_states = OrderedDict()
series_state_limit = max(int(os.getenv("SERIES_STATE_LIMIT", "1000")), 1)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "time": self.formatTime(record, self.datefmt),
            }
        )


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)


async def _handle_stream(payload):
    observation = Observation.model_validate(payload)
    _prediction(observation.features, observation.timestamp, observation.series_id)


def _install_model(loaded):
    global model
    with model_lock:
        model = loaded
        sequence_states.clear()
        ACTIVE_MODEL.set(1 if loaded is not None else 0)


async def _watch_registry():
    while True:
        try:
            loaded = await asyncio.to_thread(registry.load_latest)
            loaded_version = loaded.metadata.get("version") if loaded else None
            current_version = model.metadata.get("version") if model else None
            if loaded is not None and loaded_version != current_version:
                _install_model(loaded)
                logging.getLogger(__name__).info("loaded promoted model %s", loaded_version)
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            logging.getLogger(__name__).exception("model reload check failed")
        await asyncio.sleep(float(os.getenv("MODEL_RELOAD_SECONDS", "5")))


@asynccontextmanager
async def lifespan(_app):
    global stream_task, reload_task
    ACTIVE_MODEL.set(1 if model is not None else 0)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        stream_task = asyncio.create_task(
            consume_stream(
                redis_url,
                os.getenv("REDIS_STREAM", "market-observations"),
                os.getenv("REDIS_GROUP", "regime-api"),
                os.getenv("REDIS_CONSUMER", "api"),
                _handle_stream,
            )
        )
    reload_task = asyncio.create_task(_watch_registry())
    yield
    tasks = [task for task in (stream_task, reload_task) if task]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Real-time Market Regime Detection", version="1.0.0", lifespan=lifespan)


class Observation(BaseModel):
    features: list[float] = Field(..., min_length=1)
    timestamp: datetime | None = None
    series_id: str | None = Field(None, min_length=1, max_length=100)


class BatchRequest(BaseModel):
    observations: list[Observation] = Field(..., min_length=1)
    reset: bool = True


class DriftRequest(BaseModel):
    reference: list[list[float]] = Field(..., min_length=2)
    current: list[list[float]] = Field(..., min_length=2)
    threshold: float = Field(0.2, gt=0)
    reference_loglik: list[float] | None = None
    current_loglik: list[float] | None = None
    likelihood_threshold: float = Field(2.0, gt=0)


class RawPriceRequest(BaseModel):
    prices: list[float] = Field(..., min_length=11)
    timestamps: list[datetime] | None = None


class PredictionResponse(BaseModel):
    model_version: str
    timestamp: datetime
    state: int
    regime: str
    probabilities: list[float]
    confidence: float
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


def _require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="no model is loaded")
    return model


def _format_prediction(active_model, probabilities, timestamp, latency_ms):
    state = int(np.argmax(probabilities))
    version = active_model.metadata.get("version", "unknown")
    labels = active_model.metadata.get("state_labels", {})
    PREDICTION_COUNT.labels(str(state), version).inc()
    result = {
        "model_version": version,
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "state": state,
        "regime": labels.get(str(state), labels.get(state, f"State{state}")),
        "probabilities": probabilities.tolist(),
        "confidence": float(probabilities[state]),
        "latency_ms": latency_ms,
    }
    record_prediction(result)
    return result


def _prediction(features, timestamp=None, series_id=None):
    started = time.perf_counter()
    active_model = _require_model()
    with model_lock:
        initial_state = sequence_states.get(series_id) if series_id else None
        values, final_state = active_model.filter_from_state(np.asarray([features]), initial_state)
        if series_id:
            sequence_states.pop(series_id, None)
            sequence_states[series_id] = final_state
            while len(sequence_states) > series_state_limit:
                sequence_states.popitem(last=False)
    latency_ms = (time.perf_counter() - started) * 1000
    return _format_prediction(active_model, values[0], timestamp, latency_ms)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - started
    REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(duration)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(_request, _error):
    return JSONResponse(status_code=400, content={"detail": "invalid request payload"})


@app.exception_handler(ValueError)
async def value_error(_request, error):
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "model_version": model.metadata.get("version", "unknown") if model else None}


@app.get("/api/v1/ready")
async def ready(response: Response):
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False}
    return {"ready": True, "model_version": model.metadata.get("version", "unknown")}


@app.get("/api/v1/model")
async def model_info():
    active_model = _require_model()
    return {
        "version": active_model.metadata.get("version", "unknown"),
        "features": active_model.feature_names,
        "n_states": active_model.n_states,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(observation: Observation):
    return _prediction(observation.features, observation.timestamp, observation.series_id)


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchRequest):
    started = time.perf_counter()
    active_model = _require_model()
    with model_lock:
        series_ids = {item.series_id for item in request.observations}
        if not request.reset and (None in series_ids or len(series_ids) != 1):
            raise ValueError("a non-reset batch requires one shared series_id")
        series_id = next(iter(series_ids)) if len(series_ids) == 1 else None
        initial_state = None if request.reset else sequence_states.get(series_id)
        probabilities, final_state = active_model.filter_from_state(
            np.asarray([item.features for item in request.observations]), initial_state
        )
        if not request.reset:
            sequence_states.pop(series_id, None)
            sequence_states[series_id] = final_state
            while len(sequence_states) > series_state_limit:
                sequence_states.popitem(last=False)
    elapsed = (time.perf_counter() - started) * 1000
    results = [
        _format_prediction(active_model, values, item.timestamp, elapsed / len(probabilities))
        for item, values in zip(request.observations, probabilities)
    ]
    return {"predictions": results}


@app.post("/api/v1/drift/check")
async def check_drift(request: DriftRequest):
    active_model = _require_model()
    result = drift_report(
        request.reference,
        request.current,
        active_model.feature_names,
        request.threshold,
        request.reference_loglik,
        request.current_loglik,
        request.likelihood_threshold,
    )
    DRIFT_SCORE.set(result["score"])
    return result


@app.post("/api/v1/predict/prices", response_model=BatchPredictionResponse)
async def predict_prices(request: RawPriceRequest):
    started = time.perf_counter()
    active_model = _require_model()
    frame = prices_to_features(request.prices, request.timestamps)
    with model_lock:
        probabilities, _ = active_model.filter_from_state(frame[["LogReturn", "Volatility"]].values)
    elapsed = (time.perf_counter() - started) * 1000
    results = [
        _format_prediction(
            active_model,
            values,
            request.timestamps[index + 10] if request.timestamps else None,
            elapsed / len(probabilities),
        )
        for index, values in enumerate(probabilities)
    ]
    return {"predictions": results}


@app.post("/api/v1/admin/reload-model")
async def reload_model(x_admin_token: str | None = Header(None)):
    configured_token = os.getenv("ADMIN_TOKEN")
    if not configured_token:
        raise HTTPException(status_code=503, detail="administrative reload is disabled")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, configured_token):
        raise HTTPException(status_code=401, detail="invalid administrative token")
    loaded = registry.load_latest()
    if loaded is None:
        raise HTTPException(status_code=404, detail="no registered model found")
    _install_model(loaded)
    return {"reloaded": True, "version": model.metadata.get("version", "unknown")}


@app.get("/metrics")
async def metrics():
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type.split(";", 1)[0])
