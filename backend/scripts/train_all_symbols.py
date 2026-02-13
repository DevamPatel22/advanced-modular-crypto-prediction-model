#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
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


def _default_model_version() -> str:
    return datetime.now(tz=UTC).strftime("daily-%Y%m%d-%H%M%S")


async def _resolve_universe(limit: int) -> list[str]:
    market_symbols = await fetch_symbols_by_quote(quote="USD", limit=limit)
    by_market = [item.symbol for item in market_symbols]
    in_db = set(list_symbols_with_any_candles(min_rows=100))
    return [symbol for symbol in by_market if symbol in in_db]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train candidate models for all US-tradable USD crypto symbols")
    parser.add_argument("--model-version", default=_default_model_version(), help="Model version folder name")
    parser.add_argument(
        "--output",
        default="reports/summary_report.json",
        help="Aggregate report output path",
    )
    parser.add_argument("--symbol-limit", type=int, default=2000, help="Universe fetch cap")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size used for progress sections")
    args = parser.parse_args()

    universe = asyncio.run(_resolve_universe(args.symbol_limit))

    models_root = PROJECT_ROOT / "data" / "models"
    reports_root = PROJECT_ROOT / "reports"
    symbols_report_root = reports_root / "symbols"
    symbols_report_root.mkdir(parents=True, exist_ok=True)

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
        "symbols": {},
        "phase_activation": {
            "phase_1": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "phase_2_batch_size": args.batch_size,
        },
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
    version_manifest_path = models_root / args.model_version / "manifest.json"
    version_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    version_manifest_path.write_text(json.dumps(version_manifest, indent=2), encoding="utf-8")

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"summary_report": str(output_path), "manifest": str(version_manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
