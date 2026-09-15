from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_COUNT = Counter("regime_api_requests_total", "API requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("regime_api_request_latency_seconds", "API request latency", ["path"])
PREDICTION_COUNT = Counter("regime_predictions_total", "Predictions by state", ["state", "model_version"])
DRIFT_SCORE = Gauge("regime_drift_score", "Latest drift score")
ACTIVE_MODEL = Gauge("regime_active_model", "Whether a model is loaded")


def metrics_payload():
    return generate_latest(), CONTENT_TYPE_LATEST