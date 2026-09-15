import json
import os
from pathlib import Path
from threading import Lock


_write_lock = Lock()


def record_prediction(payload):
    path = Path(os.getenv("PREDICTION_LOG", "var/predictions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    with _write_lock, path.open("a", encoding="utf-8") as output:
        output.write(line)


def read_predictions(path=None):
    source = Path(path or os.getenv("PREDICTION_LOG", "var/predictions.jsonl"))
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
