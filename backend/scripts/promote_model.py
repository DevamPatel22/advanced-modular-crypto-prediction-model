#!/usr/bin/env python3
"""Promote a candidate model version into the active registry using phase policies."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_registry import ModelRegistry
from app.services.experiment_tracker import log_experiment_event
from app.config import get_settings


def _read_json(path: Path) -> dict[str, object] | None:
    """Internal helper to compute read JSON."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def _ci_gate_ok(path: Path, max_age_hours: int) -> tuple[bool, str]:
    """Internal helper to compute ci gate ok."""
    payload = _read_json(path)
    if payload is None:
        return False, "ci_gate_report_missing_or_invalid"
    if str(payload.get("status", "")).lower() != "ok":
        return False, "ci_gate_report_failed"
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str):
        return False, "ci_gate_generated_at_missing"
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except Exception:
        return False, "ci_gate_generated_at_invalid"
    min_allowed = datetime.now(tz=UTC) - timedelta(hours=max(int(max_age_hours), 1))
    if ts < min_allowed:
        return False, "ci_gate_report_too_old"
    return True, ""


def main() -> None:
    """Run the script entrypoint."""
    defaults = get_settings()
    parser = argparse.ArgumentParser(description="Promote candidate model version to active registry")
    parser.add_argument("--candidate", required=True, help="Candidate model version name")
    parser.add_argument("--active", default=None, help="Current active version (optional check)")
    parser.add_argument(
        "--output",
        default="reports/promotion_report.json",
        help="Promotion report output path",
    )
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3"],
        default="phase1",
        help="Activation phase policy",
    )
    parser.add_argument(
        "--phase2-batch-size",
        type=int,
        default=20,
        help="Batch size for phase2 activation",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly passed pairs with currently promoted registry entries",
    )
    parser.add_argument(
        "--require-ci-gate",
        action="store_true",
        help="Require a fresh passing CI gate report before promotion",
    )
    parser.add_argument(
        "--ci-gate-report",
        default="reports/ci_gate_latest.json",
        help="CI gate report path",
    )
    parser.add_argument(
        "--ci-gate-max-age-hours",
        type=int,
        default=int(defaults.ci_gate_max_age_hours),
        help="Maximum allowed CI gate report age",
    )
    args = parser.parse_args()
    settings = defaults
    bootstrap_horizons = {
        item.strip() for item in settings.bootstrap_phase1_horizons.split(",") if item.strip()
    }
    phase1_symbols = [item.strip().upper() for item in settings.phase1_focus_symbols.split(",") if item.strip()]

    registry = ModelRegistry()
    current_active = registry.get_active_model_version()
    if args.active and args.active != current_active:
        raise SystemExit(f"Active version mismatch: expected {args.active}, found {current_active}")

    if args.require_ci_gate:
        ci_gate_path = Path(args.ci_gate_report)
        if not ci_gate_path.is_absolute():
            ci_gate_path = PROJECT_ROOT / ci_gate_path
        ci_ok, ci_reason = _ci_gate_ok(ci_gate_path, int(args.ci_gate_max_age_hours))
        if not ci_ok:
            raise SystemExit(f"CI gate check failed: {ci_reason} ({ci_gate_path})")

    candidate_root = registry.models_root / args.candidate
    if not candidate_root.exists():
        raise SystemExit(f"Candidate version not found: {candidate_root}")

    passed_by_symbol: dict[str, dict[str, bool]] = {}
    failures: list[dict[str, str]] = []

    for symbol_dir in sorted(candidate_root.iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name.upper()
        symbol_map: dict[str, bool] = {}

        for metrics_file in sorted(symbol_dir.glob("metrics_*.json")):
            horizon = metrics_file.stem.replace("metrics_", "")
            # Phase-1 bootstrap intentionally limits early rollout to selected horizons.
            if args.phase == "phase1" and horizon not in bootstrap_horizons:
                continue
            payload = _read_json(metrics_file)
            if payload is None:
                failures.append({"symbol": symbol, "horizon": horizon, "reason": "invalid_metrics_json"})
                continue

            gate = payload.get("promotion_gate", {})
            passed = bool(gate.get("passed")) if isinstance(gate, dict) else False
            if passed:
                symbol_map[horizon] = True
            else:
                reason = "promotion_gate_failed"
                if isinstance(gate, dict):
                    failed = gate.get("failed_reasons")
                    if isinstance(failed, list) and failed:
                        reason = ",".join(str(item) for item in failed)
                failures.append({"symbol": symbol, "horizon": horizon, "reason": reason})

        if symbol_map:
            passed_by_symbol[symbol] = symbol_map

    promoted: dict[str, dict[str, bool]] = {}
    if args.phase == "phase1":
        for symbol in phase1_symbols:
            if symbol in passed_by_symbol:
                promoted[symbol] = passed_by_symbol[symbol]
    elif args.phase == "phase2":
        existing = registry.read().get("promoted", {})
        if isinstance(existing, dict):
            promoted.update({str(k): {str(h): bool(v) for h, v in dict(val).items()} for k, val in existing.items() if isinstance(val, dict)})
        remaining = [item for item in sorted(passed_by_symbol) if item not in promoted]
        for symbol in remaining[: max(args.phase2_batch_size, 1)]:
            promoted[symbol] = passed_by_symbol[symbol]
    else:
        promoted = passed_by_symbol

    if args.merge_existing:
        # Preserve already promoted pairs while adding newly qualified ones.
        existing_promoted = registry.read().get("promoted", {})
        merged: dict[str, dict[str, bool]] = {}
        if isinstance(existing_promoted, dict):
            for symbol, horizons in existing_promoted.items():
                if not isinstance(horizons, dict):
                    continue
                merged[str(symbol)] = {str(h): bool(v) for h, v in horizons.items() if bool(v)}
        for symbol, horizons in promoted.items():
            bucket = merged.setdefault(symbol, {})
            for horizon, passed in horizons.items():
                if bool(passed):
                    bucket[horizon] = True
        promoted = {symbol: horizons for symbol, horizons in merged.items() if horizons}

    promoted_pairs = sum(len(item) for item in promoted.values())
    promoted_symbols = len(promoted)
    kept_previous_active = promoted_pairs == 0
    if kept_previous_active:
        registry_payload = registry.read()
    else:
        registry_payload = registry.promote_candidate(candidate_version=args.candidate, promoted=promoted)

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "previous_active": current_active,
        "new_active": current_active if kept_previous_active else args.candidate,
        "phase": args.phase,
        "promoted_symbols": promoted_symbols,
        "promoted_pairs": promoted_pairs,
        "kept_previous_active": kept_previous_active,
        "failed_pairs": failures,
        "registry_path": str(registry.registry_path),
        "registry": registry_payload,
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    events_path = log_experiment_event(
        "promotion",
        {
            "candidate": args.candidate,
            "phase": args.phase,
            "promoted_symbols": promoted_symbols,
            "promoted_pairs": promoted_pairs,
            "kept_previous_active": kept_previous_active,
            "previous_active": current_active,
            "new_active": report["new_active"],
            "output_report": str(output_path),
        },
    )
    report["experiment_events_path"] = str(events_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
