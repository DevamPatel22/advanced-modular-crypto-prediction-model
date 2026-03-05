#!/usr/bin/env python3
"""Evaluate hard data/source SLAs before model training or promotion."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.data_readiness import latest_data_quality_report, source_health_summary


def _load_quality_payload(path: Path) -> dict[str, object] | None:
    """Internal helper to compute load quality payload."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    """Run the script entrypoint."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate data/source SLA gate")
    parser.add_argument("--quality-report", default="", help="Explicit data quality report path (optional)")
    parser.add_argument("--source-hours", type=int, default=24, help="Source health lookback window in hours")
    parser.add_argument(
        "--min-live-source-ratio",
        type=float,
        default=float(settings.sla_min_live_source_ratio),
        help="Minimum ratio of live-source selections (non-stale) required",
    )
    parser.add_argument(
        "--max-stale-cache-ratio",
        type=float,
        default=float(settings.sla_max_stale_cache_ratio),
        help="Maximum allowed stale-cache ratio",
    )
    parser.add_argument(
        "--require-quality-pass",
        action="store_true",
        help="Require data_quality_report gate_passed=true",
    )
    parser.add_argument("--output", default="reports/sla_gate_report.json", help="Output JSON path")
    args = parser.parse_args()

    if args.quality_report.strip():
        quality_path = Path(args.quality_report)
        if not quality_path.is_absolute():
            quality_path = PROJECT_ROOT / quality_path
        quality_payload = _load_quality_payload(quality_path)
        quality = {
            "available": quality_payload is not None,
            "path": str(quality_path),
            "generated_at": quality_payload.get("generated_at") if quality_payload else None,
            "gate_passed": bool(quality_payload.get("gate_passed")) if quality_payload else None,
            "failing_pair_count": quality_payload.get("failing_pair_count") if quality_payload else None,
            "pair_count": quality_payload.get("pair_count") if quality_payload else None,
        }
    else:
        quality = latest_data_quality_report()

    source = source_health_summary(hours=max(int(args.source_hours), 1))
    source_live_ratio = float(source.get("live_source_ratio", 0.0)) if bool(source.get("available")) else 0.0
    source_stale_ratio = float(source.get("stale_cache_ratio", 1.0)) if bool(source.get("available")) else 1.0

    quality_pass = bool(quality.get("gate_passed")) if bool(quality.get("available")) else False
    live_pass = bool(source.get("available")) and source_live_ratio >= float(args.min_live_source_ratio)
    stale_pass = bool(source.get("available")) and source_stale_ratio <= float(args.max_stale_cache_ratio)

    reasons: list[str] = []
    if args.require_quality_pass and not quality_pass:
        reasons.append("quality_gate_failed")
    if not live_pass:
        reasons.append("live_source_ratio_below_sla")
    if not stale_pass:
        reasons.append("stale_cache_ratio_above_sla")

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": "ok" if not reasons else "failed",
        "sla_passed": len(reasons) == 0,
        "reasons": reasons,
        "thresholds": {
            "require_quality_pass": bool(args.require_quality_pass),
            "min_live_source_ratio": float(args.min_live_source_ratio),
            "max_stale_cache_ratio": float(args.max_stale_cache_ratio),
            "source_hours": int(max(int(args.source_hours), 1)),
        },
        "quality_report": quality,
        "source_health": source,
        "checks": {
            "quality_pass": quality_pass,
            "live_source_ratio_pass": live_pass,
            "stale_cache_ratio_pass": stale_pass,
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if reasons:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
