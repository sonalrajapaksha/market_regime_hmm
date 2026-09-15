import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .drift import drift_report
from .features import FEATURE_NAMES, add_features, load_price_data
from .model import GaussianHMM
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


def _window_scores(model, values, window=50):
    return [
        model.score(values[index : index + window]) / len(values[index : index + window])
        for index in range(0, len(values), window)
        if len(values[index : index + window]) >= 2
    ]


def _state_labels(model, feature_names):
    if model.n_states == 2 and "LogReturn" in feature_names:
        return {
            str(index): (
                "Bear" if index == int(np.argmin(model.means[:, feature_names.index("LogReturn")])) else "Bull"
            )
            for index in range(model.n_states)
        }
    return {str(index): f"State{index}" for index in range(model.n_states)}


def evaluate_drift(model, current, threshold=0.2, likelihood_threshold=2.0):
    reference = np.asarray(model.metadata.get("drift_reference", []), dtype=float)
    if reference.ndim != 2 or reference.shape[1:] != current.shape[1:] or len(reference) < 2:
        return {"drift_detected": True, "reason": "missing compatible drift baseline"}
    sample_size = min(len(reference), len(current))
    reference_scores = model.metadata.get("reference_loglik", [])
    current_scores = _window_scores(model, current[-sample_size:])
    return drift_report(
        reference[-sample_size:],
        current[-sample_size:],
        model.feature_names,
        threshold,
        reference_scores if len(reference_scores) >= 2 else None,
        current_scores if len(current_scores) >= 1 else None,
        likelihood_threshold,
    )


def _log_mlflow(model, directory, version, metadata):
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow is unavailable; model remains registered in the filesystem")
        return None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "market-regime-hmm"))
    with mlflow.start_run(run_name=version) as run:
        mlflow.log_params(
            {"ticker": metadata["ticker"], "n_states": model.n_states, "n_iter": len(model.loglik_history_)}
        )
        mlflow.log_metrics(
            {
                "training_log_likelihood": model.training_score_,
                "candidate_validation_score": metadata["candidate_validation_score"],
            }
        )
        mlflow.set_tags({"promoted": str(metadata["promoted"]).lower(), "model_version": version})
        mlflow.log_artifacts(str(directory), artifact_path="portable_model")
        if metadata["promoted"]:

            class HMMPyFuncModel(mlflow.pyfunc.PythonModel):
                def __init__(self, hmm):
                    self.hmm = hmm

                def predict(self, context, model_input: np.ndarray, params: dict | None = None) -> np.ndarray:
                    return self.hmm.predict_proba(np.asarray(model_input, dtype=float))

            mlflow.pyfunc.log_model(
                name="model",
                python_model=HMMPyFuncModel(model),
                registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL", "market-regime-hmm"),
                input_example=np.zeros((1, model.means.shape[1])),
            )
        return run.info.run_id


def train_and_register(
    ticker="^GSPC",
    start="2005-01-01",
    end=None,
    model_dir="models",
    states=2,
    require_drift=False,
    drift_threshold=0.2,
    likelihood_threshold=2.0,
):
    end = end or datetime.now(timezone.utc).date().isoformat()
    frame = add_features(load_price_data(ticker, start, end))
    feature_names = FEATURE_NAMES
    registry = ModelRegistry(model_dir)
    production = registry.load_latest()
    drift = None
    if require_drift and production is not None:
        drift = evaluate_drift(
            production,
            frame[feature_names].values,
            threshold=drift_threshold,
            likelihood_threshold=likelihood_threshold,
        )
        if not drift["drift_detected"]:
            logger.info("retraining skipped because no drift was detected", extra={"drift_score": drift.get("score")})
            return None
    split = max(int(len(frame) * 0.8), states * 2)
    train_frame, validation_frame = frame.iloc[:split], frame.iloc[split:]
    if validation_frame.empty:
        raise ValueError("not enough observations for a held-out validation window")
    model = GaussianHMM(states, n_iter=200, tol=1e-5, random_state=42)
    model.fit(train_frame[feature_names].values, verbose=False, feature_names=feature_names)
    candidate_score = model.score(validation_frame[feature_names].values) / len(validation_frame)
    production_score = (
        production.score(validation_frame[feature_names].values) / len(validation_frame)
        if production is not None
        else None
    )
    promoted = production_score is None or candidate_score >= production_score
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    metadata = {
        "ticker": ticker,
        "training_start": start,
        "training_end": end,
        "training_rows": len(frame),
        "candidate_validation_score": candidate_score,
        "production_validation_score": production_score,
        "promoted": promoted,
        "state_labels": _state_labels(model, feature_names),
        "drift_at_retraining": drift,
        "drift_reference": train_frame[feature_names].tail(1000).values.tolist(),
        "reference_loglik": _window_scores(model, train_frame[feature_names].tail(1000).values),
        "training_quantiles": {
            name: frame[name].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).tolist() for name in feature_names
        },
    }
    directory = registry.save(model, version, metadata, promote=promoted)
    run_id = _log_mlflow(model, directory, version, metadata)
    if run_id:
        metadata_path = Path(directory) / "metadata.json"
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_metadata["mlflow_run_id"] = run_id
        metadata_path.write_text(json.dumps(saved_metadata, indent=2), encoding="utf-8")
    logger.info("registered model", extra={"version": version, "path": str(directory)})
    return version


async def scheduled_retraining(interval_hours, **kwargs):
    while True:
        try:
            await asyncio.to_thread(train_and_register, require_drift=True, **kwargs)
        except Exception:
            logger.exception("scheduled retraining failed")
        await asyncio.sleep(interval_hours * 3600)
