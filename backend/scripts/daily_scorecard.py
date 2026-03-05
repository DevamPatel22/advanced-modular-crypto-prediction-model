#!/usr/bin/env python3
"""Generate daily model scorecard for core and overall symbol/horizon coverage."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.data_readiness import source_health_summary
from app.services.model_registry import ModelRegistry


def _parse_csv(raw: str, upper: bool = False) -> list[str]:
    """Internal helper to compute parse csv."""
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if upper:
        items = [item.upper() for item in items]
    return sorted(set(items))


def _latest_report(glob_pattern: str) -> Path | None:
    """Internal helper to compute latest report."""
    reports = sorted((PROJECT_ROOT / "reports").glob(glob_pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _load_json(path: Path | None) -> dict[str, object] | None:
    """Internal helper to compute load json."""
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_summary_with_symbols(model_version: str) -> Path | None:
    """Internal helper to compute latest summary with symbols."""
    patterns = [f"summary_report_{model_version}*.json"] if model_version else ["summary_report*.json"]
    for pattern in patterns:
        for path in sorted((PROJECT_ROOT / "reports").glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
            payload = _load_json(path)
            if isinstance(payload, dict) and isinstance(payload.get("symbols"), dict):
                return path
    return None


def _float(value: object, default: float = 0.0) -> float:
    """Internal helper to compute float."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _summarize_pairs(pairs: list[dict[str, object]]) -> dict[str, object]:
    """Internal helper to compute summarize pairs."""
    if not pairs:
        return {
            "pair_count": 0,
            "gate_pass_rate": 0.0,
            "avg_net_mean_return": 0.0,
            "avg_baseline_net_mean_return": 0.0,
            "avg_net_return_edge": 0.0,
            "avg_max_drawdown": 0.0,
            "avg_baseline_max_drawdown": 0.0,
            "avg_abstain_rate": 0.0,
        }

    gate_flags = [bool(item.get("gate_passed")) for item in pairs]
    net_returns = [_float(item.get("net_mean_return"), 0.0) for item in pairs]
    baseline_net_returns = [_float(item.get("baseline_net_mean_return"), 0.0) for item in pairs]
    drawdowns = [_float(item.get("max_drawdown"), 0.0) for item in pairs]
    baseline_drawdowns = [_float(item.get("baseline_max_drawdown"), 0.0) for item in pairs]
    abstain_rates = [_float(item.get("abstain_rate"), 0.0) for item in pairs]

    return {
        "pair_count": len(pairs),
        "gate_pass_rate": round(float(mean(gate_flags)), 6),
        "avg_net_mean_return": round(float(mean(net_returns)), 8),
        "avg_baseline_net_mean_return": round(float(mean(baseline_net_returns)), 8),
        "avg_net_return_edge": round(float(mean([a - b for a, b in zip(net_returns, baseline_net_returns)])), 8),
        "avg_max_drawdown": round(float(mean(drawdowns)), 8),
        "avg_baseline_max_drawdown": round(float(mean(baseline_drawdowns)), 8),
        "avg_abstain_rate": round(float(mean(abstain_rates)), 6),
    }


def _pair_rows(summary_payload: dict[str, object]) -> list[dict[str, object]]:
    """Internal helper to compute pair rows."""
    symbol_tree = summary_payload.get("symbols", {})
    if not isinstance(symbol_tree, dict):
        return []

    rows: list[dict[str, object]] = []
    for symbol, symbol_payload in symbol_tree.items():
        if not isinstance(symbol_payload, dict):
            continue
        horizons = symbol_payload.get("horizons", {})
        if not isinstance(horizons, dict):
            continue
        for horizon, entry in horizons.items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("status", "")) != "ok":
                continue
            gate = entry.get("promotion_gate", {})
            execution = entry.get("execution_aware_metrics", {})
            baseline_execution = entry.get("baseline_execution_aware_metrics", {})
            paper = entry.get("paper_trading_metrics", {})
            baseline_paper = entry.get("baseline_paper_trading_metrics", {})
            meta = entry.get("meta_labeling", {})
            take_rate = _float(meta.get("test_take_rate"), 1.0) if isinstance(meta, dict) else 1.0
            rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "horizon": str(horizon),
                    "gate_passed": bool(gate.get("passed")) if isinstance(gate, dict) else False,
                    "failed_reasons": gate.get("failed_reasons", []) if isinstance(gate, dict) else [],
                    "net_mean_return": _float(execution.get("net_mean_return"), 0.0) if isinstance(execution, dict) else 0.0,
                    "baseline_net_mean_return": _float(baseline_execution.get("net_mean_return"), 0.0)
                    if isinstance(baseline_execution, dict)
                    else 0.0,
                    "max_drawdown": _float(paper.get("max_drawdown"), 0.0) if isinstance(paper, dict) else 0.0,
                    "baseline_max_drawdown": _float(baseline_paper.get("max_drawdown"), 0.0)
                    if isinstance(baseline_paper, dict)
                    else 0.0,
                    "abstain_rate": float(max(0.0, min(1.0, 1.0 - take_rate))),
                }
            )
    return rows


def main() -> None:
    """Run the script entrypoint."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate model scorecard report")
    parser.add_argument("--model-version", default="", help="Model version hint for report discovery")
    parser.add_argument("--summary-report", default="", help="Path to summary report JSON (optional)")
    parser.add_argument("--promotion-report", default="", help="Path to promotion report JSON (optional)")
    parser.add_argument("--source-hours", type=int, default=24, help="Source-health lookback window")
    parser.add_argument(
        "--symbols",
        default=settings.phase1_focus_symbols,
        help="Core symbols for phase-1 scorecard",
    )
    parser.add_argument(
        "--horizons",
        default=settings.phase1_focus_horizons,
        help="Core horizons for phase-1 scorecard",
    )
    parser.add_argument("--output", default="reports/scorecard_latest.json", help="Output path")
    args = parser.parse_args()

    model_version = args.model_version.strip()
    summary_path = Path(args.summary_report) if args.summary_report.strip() else None
    promotion_path = Path(args.promotion_report) if args.promotion_report.strip() else None
    if summary_path and not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    if promotion_path and not promotion_path.is_absolute():
        promotion_path = PROJECT_ROOT / promotion_path

    if summary_path is None:
        summary_path = _latest_summary_with_symbols(model_version)
    if promotion_path is None:
        if model_version:
            promotion_path = _latest_report(f"promotion_report_{model_version}*.json")
        if promotion_path is None:
            promotion_path = _latest_report("promotion_report*.json")

    summary_payload = _load_json(summary_path)
    promotion_payload = _load_json(promotion_path)
    if summary_payload is None:
        raise SystemExit("Unable to resolve a valid summary report for scorecard generation")

    if not model_version:
        model_version = str(summary_payload.get("model_version", "unknown"))
    core_symbols = set(_parse_csv(args.symbols, upper=True))
    core_horizons = set(_parse_csv(args.horizons, upper=False))

    rows = _pair_rows(summary_payload)
    core_rows = [item for item in rows if item["symbol"] in core_symbols and str(item["horizon"]) in core_horizons]
    failed_reason_counts: dict[str, int] = {}
    for item in rows:
        for reason in item.get("failed_reasons", []):
            key = str(reason)
            failed_reason_counts[key] = failed_reason_counts.get(key, 0) + 1

    registry = ModelRegistry().read()
    source = source_health_summary(hours=max(int(args.source_hours), 1))
    scorecard = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": "ok",
        "model_version": model_version,
        "inputs": {
            "summary_report": str(summary_path) if summary_path else None,
            "promotion_report": str(promotion_path) if promotion_path else None,
        },
        "promotion_snapshot": {
            "promoted_pairs_in_report": int(promotion_payload.get("promoted_pairs", 0)) if isinstance(promotion_payload, dict) else 0,
            "promoted_symbols_in_report": int(promotion_payload.get("promoted_symbols", 0)) if isinstance(promotion_payload, dict) else 0,
            "active_model_version": registry.get("active_model_version"),
        },
        "core_focus": {
            "symbols": sorted(core_symbols),
            "horizons": sorted(core_horizons),
        },
        "core_summary": _summarize_pairs(core_rows),
        "overall_summary": _summarize_pairs(rows),
        "top_failed_reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(failed_reason_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        ],
        "source_health": source,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
