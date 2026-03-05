#!/usr/bin/env python3
"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.data_readiness import latest_data_quality_report, source_health_summary
from app.services.experiment_tracker import log_experiment_event
from app.services.model_registry import ModelRegistry


def _find_rollback_target(registry_payload: dict[str, object], active_version: str) -> tuple[str | None, dict[str, dict[str, bool]] | None]:
    history = registry_payload.get("history", [])
    if not isinstance(history, list):
        return None, None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("to_active", "")) != active_version:
            continue
        previous = str(item.get("from_active", "")).strip()
        previous_promoted = item.get("from_promoted", {})
        if not previous:
            continue
        if isinstance(previous_promoted, dict):
            cleaned: dict[str, dict[str, bool]] = {}
            for symbol, horizons in previous_promoted.items():
                if not isinstance(horizons, dict):
                    continue
                cleaned[str(symbol)] = {str(h): bool(v) for h, v in horizons.items() if bool(v)}
            return previous, cleaned
        return previous, {}
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-rollback guard based on data quality and source health")
    parser.add_argument("--hours", type=int, default=24, help="Source-health lookback window")
    parser.add_argument("--max-stale-cache-ratio", type=float, default=0.35, help="Rollback trigger stale-cache ratio")
    parser.add_argument("--require-quality-pass", action="store_true", help="Rollback trigger if latest quality gate failed")
    parser.add_argument("--apply", action="store_true", help="Apply rollback (default is dry-run)")
    parser.add_argument("--output", default="reports/auto_rollback_guard.json", help="Output report path")
    args = parser.parse_args()

    registry = ModelRegistry()
    registry_payload = registry.read()
    active_version = registry.get_active_model_version()

    quality = latest_data_quality_report()
    source = source_health_summary(hours=max(int(args.hours), 1))
    stale_ratio = float(source.get("stale_cache_ratio", 0.0) or 0.0)
    quality_gate_passed = bool(quality.get("gate_passed")) if bool(quality.get("available")) else True

    reasons: list[str] = []
    if stale_ratio > float(args.max_stale_cache_ratio):
        reasons.append("stale_cache_ratio_exceeded")
    if args.require_quality_pass and not quality_gate_passed:
        reasons.append("quality_gate_failed")
    if str(source.get("status")) == "degraded":
        reasons.append("source_health_degraded")

    should_rollback = len(reasons) > 0
    target_version, target_promoted = _find_rollback_target(registry_payload, active_version=active_version)
    applied = False
    rollback_payload: dict[str, object] | None = None

    if should_rollback and args.apply and target_version and target_promoted is not None:
        rollback_payload = registry.rollback_to(
            target_version=target_version,
            promoted=target_promoted,
            reason=";".join(reasons),
        )
        applied = True

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "active_before": active_version,
        "active_after": registry.get_active_model_version(),
        "should_rollback": should_rollback,
        "applied": applied,
        "reasons": reasons,
        "rollback_target_version": target_version,
        "quality": quality,
        "source_health": source,
        "registry_snapshot": rollback_payload if rollback_payload is not None else registry_payload,
    }

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    event_path = log_experiment_event(
        "auto_rollback_guard",
        {
            "should_rollback": should_rollback,
            "applied": applied,
            "active_before": active_version,
            "active_after": registry.get_active_model_version(),
            "reasons": reasons,
            "output_report": str(output_path),
        },
    )
    report["experiment_events_path"] = str(event_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if should_rollback and args.apply and not applied:
        raise SystemExit("Rollback requested but no valid rollback target was found in registry history")


if __name__ == "__main__":
    main()
