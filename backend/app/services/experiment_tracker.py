from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def _events_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "reports" / "experiment_events.jsonl"


def log_experiment_event(event_type: str, payload: dict[str, object]) -> Path:
    path = _events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_time": datetime.now(tz=UTC).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, separators=(",", ":")))
        handle.write("\n")
    return path
