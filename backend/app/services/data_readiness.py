"""Data quality and source-health accessors used by health endpoints and guards."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import get_settings


def _reports_root() -> Path:
    """Internal helper to compute reports root."""
    return Path(__file__).resolve().parents[2] / "reports"


def _db_path() -> Path:
    """Internal helper to compute database path."""
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _safe_load_json(path: Path) -> dict[str, object] | None:
    """Internal helper to compute safe load JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_data_quality_report() -> dict[str, object]:
    """Compute latest data quality report."""
    root = _reports_root()
    candidates = sorted(
        list(root.glob("data_quality_*.json")) + list(root.glob("data_quality_report*.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {
            "available": False,
            "message": "No data quality reports found in reports directory",
        }

    path = candidates[0]
    payload = _safe_load_json(path)
    if payload is None:
        return {
            "available": False,
            "message": f"Latest data quality report is unreadable: {path.name}",
            "path": str(path),
        }

    return {
        "available": True,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "gate_passed": payload.get("gate_passed"),
        "failing_pair_count": payload.get("failing_pair_count"),
        "pair_count": payload.get("pair_count"),
        "symbol_summary": payload.get("symbol_summary", {}),
    }


def latest_scorecard_report() -> dict[str, object]:
    """Compute latest scorecard report."""
    root = _reports_root()
    candidates = sorted(
        list(root.glob("scorecard_*.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {
            "available": False,
            "message": "No scorecard reports found in reports directory",
        }

    path = candidates[0]
    payload = _safe_load_json(path)
    if payload is None:
        return {
            "available": False,
            "message": f"Latest scorecard report is unreadable: {path.name}",
            "path": str(path),
        }

    return {
        "available": True,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "model_version": payload.get("model_version"),
        "status": payload.get("status"),
        "core_summary": payload.get("core_summary", {}),
        "overall_summary": payload.get("overall_summary", {}),
    }


def latest_robust_validation_report() -> dict[str, object]:
    """Load the newest robust validation bundle if available."""
    root = _reports_root()
    candidates = sorted(list(root.glob("robust_validation*.json")), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return {
            "available": False,
            "message": "No robust validation reports found in reports directory",
        }
    path = candidates[0]
    payload = _safe_load(path)
    if payload is None:
        return {
            "available": False,
            "message": f"Latest robust validation report is unreadable: {path.name}",
            "path": str(path),
        }
    return {
        "available": True,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "model_version": payload.get("model_version"),
        "promoted_pair_count": payload.get("promoted_pair_count"),
        "robust_alpha_gate_passed": payload.get("robust_alpha_gate_passed"),
        "oos_summary": payload.get("oos_summary", {}),
        "shadow_validation": payload.get("shadow_validation", {}),
    }


def latest_shadow_report() -> dict[str, object]:
    """Load the newest shadow trading report if available."""
    root = _reports_root()
    candidates = sorted(list(root.glob("shadow_report*.json")), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return {
            "available": False,
            "message": "No shadow trading reports found in reports directory",
        }
    path = candidates[0]
    payload = _safe_load(path)
    if payload is None:
        return {
            "available": False,
            "message": f"Latest shadow report is unreadable: {path.name}",
            "path": str(path),
        }
    return {
        "available": True,
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "model_version": payload.get("model_version"),
        "overall": payload.get("overall", {}),
    }


def source_health_summary(hours: int = 24) -> dict[str, object]:
    """Compute source health summary."""
    db_path = _db_path()
    if not db_path.exists():
        return {
            "available": False,
            "message": f"Market data DB not found: {db_path}",
        }

    horizon_hours = max(int(hours), 1)
    since = datetime.now(tz=UTC) - timedelta(hours=horizon_hours)
    since_iso = since.isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        total_row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM source_health_events
            WHERE datetime(event_time) >= datetime(?)
            """,
            (since_iso,),
        ).fetchone()
        total_events = int(total_row["count"]) if total_row else 0

        stale_row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM source_health_events
            WHERE datetime(event_time) >= datetime(?) AND used_stale_cache = 1
            """,
            (since_iso,),
        ).fetchone()
        stale_events = int(stale_row["count"]) if stale_row else 0

        by_source_rows = conn.execute(
            """
            SELECT selected_source, COUNT(*) AS count
            FROM source_health_events
            WHERE datetime(event_time) >= datetime(?)
            GROUP BY selected_source
            ORDER BY count DESC
            """,
            (since_iso,),
        ).fetchall()

        by_granularity_rows = conn.execute(
            """
            SELECT granularity, COUNT(*) AS count
            FROM source_health_events
            WHERE datetime(event_time) >= datetime(?)
            GROUP BY granularity
            ORDER BY granularity ASC
            """,
            (since_iso,),
        ).fetchall()

        latest_rows = conn.execute(
            """
            SELECT event_time, symbol, granularity, selected_source, primary_status, secondary_status, used_stale_cache
            FROM source_health_events
            ORDER BY event_time DESC
            LIMIT 12
            """
        ).fetchall()

    stale_ratio = (stale_events / total_events) if total_events > 0 else 0.0
    live_events = max(total_events - stale_events, 0)
    live_source_ratio = (live_events / total_events) if total_events > 0 else 0.0
    status = "healthy"
    if total_events == 0:
        status = "no_recent_events"
    elif stale_ratio > 0.35:
        status = "degraded"
    elif stale_ratio > 0.15:
        status = "warning"

    return {
        "available": True,
        "status": status,
        "hours": horizon_hours,
        "since": since_iso,
        "total_events": total_events,
        "stale_cache_events": stale_events,
        "stale_cache_ratio": round(float(stale_ratio), 6),
        "live_source_events": live_events,
        "live_source_ratio": round(float(live_source_ratio), 6),
        "source_breakdown": [
            {"selected_source": str(row["selected_source"]), "count": int(row["count"])}
            for row in by_source_rows
        ],
        "granularity_breakdown": [
            {"granularity": str(row["granularity"]), "count": int(row["count"])}
            for row in by_granularity_rows
        ],
        "latest_events": [
            {
                "event_time": row["event_time"],
                "symbol": row["symbol"],
                "granularity": row["granularity"],
                "selected_source": row["selected_source"],
                "primary_status": row["primary_status"],
                "secondary_status": row["secondary_status"],
                "used_stale_cache": bool(row["used_stale_cache"]),
            }
            for row in latest_rows
        ],
    }
