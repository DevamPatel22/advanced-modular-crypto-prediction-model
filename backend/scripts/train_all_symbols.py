#!/usr/bin/env python3
"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.features import FEATURE_VERSION
from app.ml.training import all_horizon_specs, evaluate_symbol_horizon, list_symbols_with_any_candles, training_window_config_snapshot
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


def _parse_horizons(raw: str) -> list[str]:
    """Parse horizons. Internal helper."""
    horizons = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return sorted(set(horizons))


def _parse_symbol_horizon_pairs(raw: str) -> set[tuple[str, str]]:
    """Parse SYMBOL:HORIZON CSV into a normalized pair set."""
    out: set[tuple[str, str]] = set()
    for token in raw.split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        symbol_raw, horizon_raw = [part.strip() for part in item.split(":", 1)]
        if not symbol_raw or not horizon_raw:
            continue
        out.add((symbol_raw.upper(), horizon_raw.lower()))
    return out


def _parse_int_map(raw: str) -> dict[str, int]:
    """Parse granularity:int mapping text."""
    parsed: dict[str, int] = {}
    for token in raw.split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        key_raw, value_raw = item.split(":", 1)
        key = key_raw.strip()
        try:
            value = int(value_raw.strip())
        except ValueError:
            continue
        if key and value > 0:
            parsed[key] = value
    return parsed


def _query_dataset_snapshot(db_path: Path, symbols: list[str], granularities: list[str]) -> list[dict[str, object]]:
    """Query rows/min/max timestamp snapshot for dataset lineage."""
    if not db_path.exists() or not symbols or not granularities:
        return []
    placeholders_symbols = ",".join("?" for _ in symbols)
    placeholders_granularities = ",".join("?" for _ in granularities)
    query = f"""
        SELECT symbol, granularity, COUNT(*) AS row_count, MIN(start_time) AS min_start_time, MAX(start_time) AS max_start_time
        FROM candles
        WHERE symbol IN ({placeholders_symbols}) AND granularity IN ({placeholders_granularities})
        GROUP BY symbol, granularity
        ORDER BY symbol ASC, granularity ASC
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, symbols + granularities).fetchall()
    out: list[dict[str, object]] = []
    for symbol, granularity, row_count, min_start, max_start in rows:
        out.append(
            {
                "symbol": str(symbol),
                "granularity": str(granularity),
                "row_count": int(row_count),
                "min_start_time": int(min_start) if min_start is not None else None,
                "max_start_time": int(max_start) if max_start is not None else None,
            }
        )
    return out


def _compute_asof_cutoff_map(
    dataset_snapshot: list[dict[str, object]],
    symbols: list[str],
    granularities: list[str],
) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
    """Compute synchronized as-of cutoffs by granularity and coverage diagnostics."""
    by_granularity: dict[str, dict[str, int]] = {granularity: {} for granularity in granularities}
    for row in dataset_snapshot:
        symbol = str(row.get("symbol", "")).upper()
        granularity = str(row.get("granularity", ""))
        latest = row.get("max_start_time")
        if symbol and granularity in by_granularity and isinstance(latest, int):
            by_granularity[granularity][symbol] = latest

    cutoff_map: dict[str, int] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for granularity in granularities:
        latest_by_symbol = by_granularity.get(granularity, {})
        available_count = len(latest_by_symbol)
        expected_count = len(symbols)
        if available_count == 0:
            diagnostics[granularity] = {
                "available_symbols": 0,
                "expected_symbols": expected_count,
                "coverage_ratio": 0.0,
                "cutoff_start_time": None,
            }
            continue
        cutoff = min(latest_by_symbol.values())
        cutoff_map[granularity] = int(cutoff)
        diagnostics[granularity] = {
            "available_symbols": available_count,
            "expected_symbols": expected_count,
            "coverage_ratio": float(available_count / max(expected_count, 1)),
            "cutoff_start_time": int(cutoff),
        }
    return cutoff_map, diagnostics


def _compute_sync_row_depth_map(
    dataset_snapshot: list[dict[str, object]],
    symbols: list[str],
    granularities: list[str],
    min_coverage_ratio: float,
) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
    """Compute synchronized row-depth map by granularity for cross-symbol parity."""
    by_granularity: dict[str, dict[str, int]] = {granularity: {} for granularity in granularities}
    for row in dataset_snapshot:
        symbol = str(row.get("symbol", "")).upper()
        granularity = str(row.get("granularity", ""))
        row_count = row.get("row_count")
        if symbol and granularity in by_granularity and isinstance(row_count, int):
            by_granularity[granularity][symbol] = int(row_count)

    min_cov = float(min(max(min_coverage_ratio, 0.0), 1.0))
    row_depth_map: dict[str, int] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for granularity in granularities:
        counts = by_granularity.get(granularity, {})
        available = len(counts)
        expected = len(symbols)
        coverage_ratio = float(available / max(expected, 1))
        coverage_passed = coverage_ratio >= min_cov
        target_depth = int(min(counts.values())) if counts and coverage_passed else None
        if isinstance(target_depth, int) and target_depth > 0:
            row_depth_map[granularity] = target_depth
        diagnostics[granularity] = {
            "available_symbols": available,
            "expected_symbols": expected,
            "coverage_ratio": coverage_ratio,
            "min_coverage_ratio_required": min_cov,
            "coverage_passed": bool(coverage_passed),
            "target_row_depth": target_depth,
        }
    return row_depth_map, diagnostics


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
        "--horizons",
        default="",
        help="Optional comma-separated horizon subset (example: 6h,12h,1d)",
    )
    parser.add_argument(
        "--allow-existing-version",
        action="store_true",
        help="Allow training into an existing model version folder (disabled by default for immutability)",
    )
    parser.add_argument(
        "--exclude-symbol-horizons",
        default="",
        help="Optional comma-separated SYMBOL:HORIZON list to skip from training",
    )
    parser.add_argument(
        "--asof-cutoff-map",
        default="",
        help="Optional granularity:start_time map for synchronized training cutoff (example: 1m:1772559060,1h:1772557200)",
    )
    args = parser.parse_args()
    settings = get_settings()

    requested_symbols = _parse_symbols(args.symbols)
    if requested_symbols:
        allowed = set(list_symbols_with_any_candles(min_rows=100))
        universe = [symbol for symbol in requested_symbols if symbol in allowed]
    else:
        discovered_universe = asyncio.run(_resolve_universe(args.symbol_limit))
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
    requested_horizons = _parse_horizons(args.horizons)
    if requested_horizons:
        allowed = {spec.label for spec in horizon_specs}
        selected = [spec for spec in horizon_specs if spec.label in requested_horizons and spec.label in allowed]
        if not selected:
            raise SystemExit("No valid horizons selected; check --horizons values")
        horizon_specs = selected
    excluded_pairs = _parse_symbol_horizon_pairs(args.exclude_symbol_horizons)

    db_path = PROJECT_ROOT / settings.market_data_sqlite_path
    granularities_for_run = sorted({granularity for spec in horizon_specs for granularity, _steps in spec.candidates})
    dataset_snapshot = _query_dataset_snapshot(db_path=db_path, symbols=universe, granularities=granularities_for_run)
    auto_asof_map, asof_diagnostics = _compute_asof_cutoff_map(
        dataset_snapshot=dataset_snapshot,
        symbols=universe,
        granularities=granularities_for_run,
    )
    sync_row_depth_map, sync_row_depth_diagnostics = _compute_sync_row_depth_map(
        dataset_snapshot=dataset_snapshot,
        symbols=universe,
        granularities=granularities_for_run,
        min_coverage_ratio=float(settings.train_sync_row_depth_min_coverage_ratio),
    )
    if bool(settings.train_sync_row_depth_enabled) and bool(settings.train_sync_row_depth_require_all_granularities):
        missing = [granularity for granularity in granularities_for_run if granularity not in sync_row_depth_map]
        if missing:
            raise SystemExit(
                "Synchronized row-depth map missing required granularities: "
                + ",".join(sorted(missing))
            )
    explicit_asof_map = _parse_int_map(args.asof_cutoff_map)
    asof_cutoff_map = explicit_asof_map if explicit_asof_map else (auto_asof_map if settings.train_asof_sync_enabled else {})
    active_sync_row_depth_map = sync_row_depth_map if bool(settings.train_sync_row_depth_enabled) else {}

    aggregate: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": args.model_version,
        "feature_version": FEATURE_VERSION,
        "training_universe": {
            "count": len(universe),
            "symbols": universe,
        },
        "selected_horizons": [spec.label for spec in horizon_specs],
        "excluded_symbol_horizons": sorted([f"{symbol}:{horizon}" for symbol, horizon in excluded_pairs]),
        "asof_cutoff_map": asof_cutoff_map,
        "asof_sync_enabled": bool(settings.train_asof_sync_enabled),
        "asof_diagnostics": asof_diagnostics,
        "sync_row_depth_enabled": bool(settings.train_sync_row_depth_enabled),
        "sync_row_depth_map": active_sync_row_depth_map,
        "sync_row_depth_diagnostics": sync_row_depth_diagnostics,
        "dataset_snapshot": {
            "db_path": str(db_path),
            "rows": dataset_snapshot,
        },
        "promotion_gate": {
            "classification": ["f1 > baseline.f1", "accuracy > baseline.accuracy"],
            "regression": ["rmse < baseline.rmse"],
            "execution": ["net_mean_return > baseline.net_mean_return (if enabled)"],
            "risk": ["max_drawdown >= configured floor and baseline floor"],
            "stochastic": ["abs(residual_acf1) <= 0.10"],
        },
        "training_config": {
            "classification_label_mode": settings.classification_label_mode,
            "triple_barrier_sigma_mult": float(settings.triple_barrier_sigma_mult),
            "high_confidence_threshold": float(settings.high_confidence_threshold),
            "horizon_training_window": training_window_config_snapshot(),
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
            if (symbol, spec.label) in excluded_pairs:
                symbol_result["horizons"][spec.label] = {
                    "symbol": symbol,
                    "horizon": spec.label,
                    "status": "excluded_data_quality",
                    "reason": "excluded_by_retrain_controller",
                }
                symbol_result["summary"]["insufficient"] = int(symbol_result["summary"]["insufficient"]) + 1
                continue
            # Each symbol+horizon is trained/evaluated independently with strict gate checks.
            entry = evaluate_symbol_horizon(
                symbol=symbol,
                spec=spec,
                model_version=args.model_version,
                models_root=models_root,
                write_artifacts=True,
                as_of_cutoff_by_granularity=asof_cutoff_map if asof_cutoff_map else None,
                sync_row_depth_by_granularity=active_sync_row_depth_map if active_sync_row_depth_map else None,
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
        "selected_horizons": [spec.label for spec in horizon_specs],
        "excluded_symbol_horizons": sorted([f"{symbol}:{horizon}" for symbol, horizon in excluded_pairs]),
        "asof_cutoff_map": asof_cutoff_map,
        "sync_row_depth_map": active_sync_row_depth_map,
        "horizon_training_window": training_window_config_snapshot(),
        "dataset_snapshot_path": str((version_root / "dataset_manifest.json")),
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

    dataset_manifest_path = version_root / "dataset_manifest.json"
    dataset_manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": args.model_version,
        "db_path": str(db_path),
        "asof_cutoff_map": asof_cutoff_map,
        "asof_diagnostics": asof_diagnostics,
        "sync_row_depth_enabled": bool(settings.train_sync_row_depth_enabled),
        "sync_row_depth_map": active_sync_row_depth_map,
        "sync_row_depth_diagnostics": sync_row_depth_diagnostics,
        "granularities": granularities_for_run,
        "rows": dataset_snapshot,
    }
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    aggregate["dataset_manifest"] = str(dataset_manifest_path)

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
            "dataset_manifest": str(dataset_manifest_path),
            "near_pass_candidates": len(aggregate.get("near_pass_candidates", [])) if isinstance(aggregate.get("near_pass_candidates"), list) else None,
            "sync_row_depth_enabled": bool(settings.train_sync_row_depth_enabled),
        },
    )
    print(
        json.dumps(
            {
                "summary_report": str(output_path),
                "manifest": str(version_manifest_path),
                "artifact_manifest": str(artifact_manifest_path),
                "dataset_manifest": str(dataset_manifest_path),
            },
            indent=2,
        )
    )
    print(json.dumps({"experiment_events_path": str(event_path)}, indent=2))


if __name__ == "__main__":
    main()
