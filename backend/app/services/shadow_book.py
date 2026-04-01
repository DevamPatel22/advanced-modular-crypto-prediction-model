"""Helpers for frozen champion shadow-trading artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings


def _resolve(path_value: str) -> Path:
    """Resolve configured relative paths from the backend project root."""
    root = Path(__file__).resolve().parents[2]
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def shadow_champion_path() -> Path:
    """Return frozen champion metadata path."""
    return _resolve(get_settings().shadow_champion_path)


def shadow_predictions_path() -> Path:
    """Return JSONL path for captured shadow predictions."""
    return _resolve(get_settings().shadow_predictions_path)


def shadow_report_path() -> Path:
    """Return latest aggregated shadow report path."""
    return _resolve(get_settings().shadow_report_path)


def read_shadow_champion() -> dict[str, object]:
    """Load frozen champion metadata if available."""
    path = shadow_champion_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_shadow_champion(payload: dict[str, object]) -> Path:
    """Persist frozen champion metadata."""
    path = shadow_champion_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now(tz=UTC).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_shadow_predictions() -> list[dict[str, object]]:
    """Load all captured shadow rows from JSONL."""
    path = shadow_predictions_path()
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_shadow_predictions(rows: list[dict[str, object]]) -> Path:
    """Rewrite the shadow JSONL file from normalized rows."""
    path = shadow_predictions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def append_shadow_prediction(row: dict[str, object]) -> Path:
    """Append one shadow prediction row to the JSONL ledger."""
    path = shadow_predictions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path
