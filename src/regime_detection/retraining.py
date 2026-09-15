import asyncio
import logging
from datetime import datetime, timezone

from .features import FEATURE_NAMES, add_features, load_price_data
from .model import GaussianHMM
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


def train_and_register(ticker="^GSPC", start="2005-01-01", end=None, model_dir="models", states=2):
    end = end or datetime.now(timezone.utc).date().isoformat()
    frame = add_features(load_price_data(ticker, start, end))
    feature_names = FEATURE_NAMES
    split = max(int(len(frame) * 0.8), states * 2)
    train_frame, validation_frame = frame.iloc[:split], frame.iloc[split:]
    model = GaussianHMM(states, n_iter=200, tol=1e-5, random_state=42)
    model.fit(train_frame[feature_names].values, verbose=False, feature_names=feature_names)
    candidate_score = model.score(validation_frame[feature_names].values) if len(validation_frame) else model.training_score_
    production = ModelRegistry(model_dir).load_latest()
    production_score = production.score(validation_frame[feature_names].values) if production is not None and len(validation_frame) else None
    promoted = production_score is None or candidate_score >= production_score
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "ticker": ticker,
        "training_start": start,
        "training_end": end,
        "training_rows": len(frame),
        "candidate_validation_score": candidate_score,
        "production_validation_score": production_score,
        "promoted": promoted,
        "state_labels": {str(index): f"State{index}" for index in range(states)},
        "training_quantiles": {
            name: frame[name].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).tolist()
            for name in feature_names
        },
    }
    directory = ModelRegistry(model_dir).save(model, version, metadata, promote=promoted)
    logger.info("registered model", extra={"version": version, "path": str(directory)})
    return version


async def scheduled_retraining(interval_hours, **kwargs):
    while True:
        try:
            await asyncio.to_thread(train_and_register, **kwargs)
        except Exception:
            logger.exception("scheduled retraining failed")
        await asyncio.sleep(interval_hours * 3600)