#!/usr/bin/env python3
"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.features import FEATURE_VERSION
from app.ml.training import all_horizon_specs, evaluate_symbol_horizon, list_symbols_with_any_candles
from app.services.markets import fetch_symbols_by_quote
from app.services.experiment_tracker import log_experiment_event
from app.config import get_settings


def _default_model_version() -> str:
    """Compute default model version. Internal helper."""
    return datetime.now(tz=UTC).strftime("daily-%Y%m%d-%H%M%S")


async def _resolve_universe(limit: int) -> list[str]:
    """Resolve universe. Internal helper."""
    market_symbols = await fetch_symbols_by_quote(quote="USD", limit=limit)
    by_market = [item.symbol for item in market_symbols]
    # Train only symbols that have at least minimal historical rows in local store.
    in_db = set(list_symbols_with_any_candles(min_rows=100))
    return [symbol for symbol in by_market if symbol in in_db]

def _parse_symbols(raw: str) -> list[str]:
    """Parse symbols. Internal helper."""
    symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return sorted(set(symbols))


def _sha256_file(path: Path) -> str:
    """Internal helper to compute sha256 file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Run the script entrypoint."""
    parser = argparse.ArgumentParser(description="Train candidate models for all US-tradable USD crypto symbols")
    parser.add_argument("--model-version", default=_default_model_version(), help="Model version folder name")
    parser.add_argument(
        "--output",
        default="reports/summary_report.json",
        help="Aggregate report output path",
    )
    parser.add_argument("--symbol-limit", type=int, default=2000, help="Universe fetch cap")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size used for progress sections")
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated explicit symbol set (example: BTC-USD,ETH-USD,SOL-USD)",
    )
    parser.add_argument(
        "--allow-existing-version",
        action="store_true",
        help="Allow training into an existing model version folder (disabled by default for immutability)",
    )
    args = parser.parse_args()
    settings = get_settings()

    discovered_universe = asyncio.run(_resolve_universe(args.symbol_limit))
    requested_symbols = _parse_symbols(args.symbols)
    if requested_symbols:
        allowed = set(discovered_universe)
        universe = [symbol for symbol in requested_symbols if symbol in allowed]
    else:
        universe = discovered_universe

    models_root = PROJECT_ROOT / "data" / "models"
    reports_root = PROJECT_ROOT / "reports"
    symbols_report_root = reports_root / "symbols"
    symbols_report_root.mkdir(parents=True, exist_ok=True)
    version_root = models_root / args.model_version
    if version_root.exists() and any(version_root.iterdir()) and not args.allow_existing_version:
        raise SystemExit(
            f"Model version '{args.model_version}' already exists and is non-empty. "
            "Choose a new version or pass --allow-existing-version."
        )

    horizon_specs = all_horizon_specs()

    aggregate: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": args.model_version,
        "feature_version": FEATURE_VERSION,
        "training_universe": {
            "count": len(universe),
            "symbols": universe,
        },
        "promotion_gate": {
            "classification": ["f1 > baseline.f1", "accuracy > baseline.accuracy"],
            "regression": ["rmse < baseline.rmse"],
            "stochastic": ["abs(residual_acf1) <= 0.10"],
        },
        "training_config": {
            "classification_label_mode": settings.classification_label_mode,
            "triple_barrier_sigma_mult": float(settings.triple_barrier_sigma_mult),
            "high_confidence_threshold": float(settings.high_confidence_threshold),
        },
        "symbols": {},
        "phase_activation": {
            "phase_1": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "phase_2_batch_size": args.batch_size,
        },
        "near_pass_candidates": [],
    }

    for idx, symbol in enumerate(universe, start=1):
        symbol_result: dict[str, object] = {
            "symbol": symbol,
            "model_version": args.model_version,
            "horizons": {},
            "summary": {
                "passed": 0,
                "failed": 0,
                "insufficient": 0,
            },
        }

        for spec in horizon_specs:
            # Each symbol+horizon is trained/evaluated independently with strict gate checks.
            entry = evaluate_symbol_horizon(
                symbol=symbol,
                spec=spec,
                model_version=args.model_version,
                models_root=models_root,
                write_artifacts=True,
            )
            symbol_result["horizons"][spec.label] = entry

            if entry.get("status") != "ok":
                symbol_result["summary"]["insufficient"] = int(symbol_result["summary"]["insufficient"]) + 1
                continue

            gate = entry.get("promotion_gate", {})
            passed = bool(gate.get("passed")) if isinstance(gate, dict) else False
            if passed:
                symbol_result["summary"]["passed"] = int(symbol_result["summary"]["passed"]) + 1
            else:
                symbol_result["summary"]["failed"] = int(symbol_result["summary"]["failed"]) + 1
                delta = entry.get("near_pass_delta", {})
                if isinstance(delta, dict):
                    aggregate["near_pass_candidates"].append(
                        {
                            "symbol": symbol,
                            "horizon": spec.label,
                            "delta_f1": float(delta.get("f1_vs_baseline", -999.0)),
                            "delta_accuracy": float(delta.get("accuracy_vs_baseline", -999.0)),
                            "delta_rmse_margin": float(delta.get("rmse_vs_baseline", -999.0)),
                            "failed_reasons": gate.get("failed_reasons", []),
                        }
                    )

        symbol_report_path = symbols_report_root / f"{symbol.lower().replace('-', '_')}.json"
        symbol_report_path.write_text(json.dumps(symbol_result, indent=2), encoding="utf-8")
        aggregate["symbols"][symbol] = symbol_result

        if idx % max(args.batch_size, 1) == 0:
            print(f"Completed {idx}/{len(universe)} symbols")

    version_manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": args.model_version,
        "feature_version": FEATURE_VERSION,
        "universe_count": len(universe),
        "symbols": universe,
    }
    version_manifest_path = version_root / "manifest.json"
    version_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    version_manifest_path.write_text(json.dumps(version_manifest, indent=2), encoding="utf-8")

    artifact_rows: list[dict[str, object]] = []
    # Capture immutable checksums for reproducibility/audit of generated artifacts.
    for artifact in sorted(version_root.rglob("*")):
        if not artifact.is_file():
            continue
        rel = artifact.relative_to(version_root)
        artifact_rows.append(
            {
                "path": str(rel),
                "size_bytes": int(artifact.stat().st_size),
                "sha256": _sha256_file(artifact),
            }
        )
    artifact_manifest_path = version_root / "artifact_manifest.json"
    artifact_manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": args.model_version,
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
    }
    artifact_manifest_path.write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
    aggregate["artifact_manifest"] = str(artifact_manifest_path)

    output_path = PROJECT_ROOT / args.output
    if isinstance(aggregate.get("near_pass_candidates"), list):
        aggregate["near_pass_candidates"] = sorted(
            aggregate["near_pass_candidates"],
            key=lambda item: (
                abs(float(item.get("delta_f1", -999.0))) + abs(float(item.get("delta_accuracy", -999.0))),
                abs(float(item.get("delta_rmse_margin", -999.0))),
            ),
        )[:50]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    event_path = log_experiment_event(
        "train_all_symbols",
        {
            "model_version": args.model_version,
            "universe_size": len(universe),
            "summary_report": str(output_path),
            "manifest": str(version_manifest_path),
            "artifact_manifest": str(artifact_manifest_path),
            "near_pass_candidates": len(aggregate.get("near_pass_candidates", [])) if isinstance(aggregate.get("near_pass_candidates"), list) else None,
        },
    )
    print(
        json.dumps(
            {
                "summary_report": str(output_path),
                "manifest": str(version_manifest_path),
                "artifact_manifest": str(artifact_manifest_path),
            },
            indent=2,
        )
    )
    print(json.dumps({"experiment_events_path": str(event_path)}, indent=2))


if __name__ == "__main__":
    main()
