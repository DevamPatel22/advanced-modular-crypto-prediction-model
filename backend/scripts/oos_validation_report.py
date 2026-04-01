#!/usr/bin/env python3
"""Generate out-of-sample validation summary for promoted pairs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _float(value: object, default: float = 0.0) -> float:
    """Cast float safely."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path) -> dict[str, object]:
    """Load JSON document as dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in {path}")
    return payload


def _promoted_pairs(promote_payload: dict[str, object]) -> list[tuple[str, str]]:
    """Extract promoted symbol+horizon pairs from promote report."""
    promoted = promote_payload.get("promoted", {})
    if not isinstance(promoted, dict) or not promoted:
        registry = promote_payload.get("registry", {})
        if isinstance(registry, dict):
            promoted = registry.get("promoted", {})
    if not isinstance(promoted, dict):
        return []
    pairs: list[tuple[str, str]] = []
    for symbol, horizons in promoted.items():
        if not isinstance(horizons, dict):
            continue
        for horizon, enabled in horizons.items():
            if bool(enabled):
                pairs.append((str(symbol).upper(), str(horizon).lower()))
    return sorted(set(pairs))


def main() -> None:
    """Run script entrypoint."""
    parser = argparse.ArgumentParser(description="Generate OOS validation snapshot for promoted pairs")
    parser.add_argument("--summary-report", required=True, help="Path to training summary report")
    parser.add_argument("--promotion-report", required=True, help="Path to promotion report")
    parser.add_argument("--output", default="reports/oos_validation_report.json", help="Output path")
    parser.add_argument("--min-net-mean-return", type=float, default=0.0, help="Minimum acceptable net mean return")
    parser.add_argument("--min-sharpe-net", type=float, default=0.0, help="Minimum acceptable net Sharpe")
    args = parser.parse_args()

    summary_path = Path(args.summary_report)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    promotion_path = Path(args.promotion_report)
    if not promotion_path.is_absolute():
        promotion_path = PROJECT_ROOT / promotion_path
    if not summary_path.exists():
        raise SystemExit(f"Summary report not found: {summary_path}")
    if not promotion_path.exists():
        raise SystemExit(f"Promotion report not found: {promotion_path}")

    summary_payload = _load_json(summary_path)
    promote_payload = _load_json(promotion_path)
    symbols = summary_payload.get("symbols", {})
    if not isinstance(symbols, dict):
        symbols = {}
    pairs = _promoted_pairs(promote_payload)

    pair_rows: list[dict[str, object]] = []
    for symbol, horizon in pairs:
        symbol_payload = symbols.get(symbol, {})
        if not isinstance(symbol_payload, dict):
            continue
        horizons = symbol_payload.get("horizons", {})
        if not isinstance(horizons, dict):
            continue
        entry = horizons.get(horizon, {})
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        execution = entry.get("execution_aware_metrics", {})
        paper = entry.get("paper_trading_metrics", {})
        if not isinstance(execution, dict):
            execution = {}
        if not isinstance(paper, dict):
            paper = {}

        net_mean_return = _float(execution.get("net_mean_return"), 0.0)
        sharpe_net = _float(execution.get("sharpe_net"), 0.0)
        total_return = _float(paper.get("total_return"), 0.0)
        max_drawdown = _float(paper.get("max_drawdown"), 0.0)
        checks = {
            "net_mean_return_pass": net_mean_return >= float(args.min_net_mean_return),
            "sharpe_net_pass": sharpe_net >= float(args.min_sharpe_net),
        }
        pair_rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "net_mean_return": net_mean_return,
                "sharpe_net": sharpe_net,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "checks": checks,
                "passed": bool(checks["net_mean_return_pass"] and checks["sharpe_net_pass"]),
            }
        )

    evaluated = len(pair_rows)
    passed = sum(1 for row in pair_rows if bool(row.get("passed")))
    avg_net = sum(_float(row.get("net_mean_return"), 0.0) for row in pair_rows) / max(evaluated, 1)
    avg_sharpe = sum(_float(row.get("sharpe_net"), 0.0) for row in pair_rows) / max(evaluated, 1)
    avg_total = sum(_float(row.get("total_return"), 0.0) for row in pair_rows) / max(evaluated, 1)

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "summary_report": str(summary_path),
        "promotion_report": str(promotion_path),
        "model_version": summary_payload.get("model_version"),
        "promotion_phase": promote_payload.get("phase"),
        "promoted_pairs_reported": int(promote_payload.get("promoted_pairs", 0) or 0),
        "pairs_evaluated": evaluated,
        "pairs_passing_oos_rules": passed,
        "rules": {
            "min_net_mean_return": float(args.min_net_mean_return),
            "min_sharpe_net": float(args.min_sharpe_net),
        },
        "aggregate": {
            "avg_net_mean_return": avg_net,
            "avg_sharpe_net": avg_sharpe,
            "avg_total_return": avg_total,
            "pair_pass_rate": float(passed / max(evaluated, 1)),
        },
        "pairs": pair_rows,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
