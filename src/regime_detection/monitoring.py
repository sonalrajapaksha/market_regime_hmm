import json
import os
from pathlib import Path


def record_prediction(payload):
    path = Path(os.getenv("PREDICTION_LOG", "var/predictions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload) + "\n")


def read_predictions(path=None):
    source = Path(path or os.getenv("PREDICTION_LOG", "var/predictions.jsonl"))
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]