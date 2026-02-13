#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_registry import ModelRegistry


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def main() -> None:
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
    args = parser.parse_args()

    registry = ModelRegistry()
    current_active = registry.get_active_model_version()
    if args.active and args.active != current_active:
        raise SystemExit(f"Active version mismatch: expected {args.active}, found {current_active}")

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

    phase1_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
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

    registry_payload = registry.promote_candidate(candidate_version=args.candidate, promoted=promoted)

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "previous_active": current_active,
        "new_active": args.candidate,
        "phase": args.phase,
        "promoted_symbols": len(promoted),
        "promoted_pairs": sum(len(item) for item in promoted.values()),
        "failed_pairs": failures,
        "registry_path": str(registry.registry_path),
        "registry": registry_payload,
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
