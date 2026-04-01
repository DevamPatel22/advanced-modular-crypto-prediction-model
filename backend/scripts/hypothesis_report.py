#!/usr/bin/env python3
"""Generate explicit alpha-hypothesis pass/fail diagnostics from a summary report."""

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

DEFAULT_HORIZON_RULES = {
    "6h": {
        "min_ic": 0.005,
        "min_decile_spread_bps": 0.5,
        "min_sign_alignment": 0.51,
        "min_net_mean_return": 0.0,
        "min_sharpe_net": 0.0,
        "max_drawdown_limit": -0.45,
    },
    "12h": {
        "min_ic": 0.005,
        "min_decile_spread_bps": 0.5,
        "min_sign_alignment": 0.51,
        "min_net_mean_return": 0.0,
        "min_sharpe_net": 0.0,
        "max_drawdown_limit": -0.45,
    },
    "1d": {
        "min_ic": 0.004,
        "min_decile_spread_bps": 0.4,
        "min_sign_alignment": 0.51,
        "min_net_mean_return": 0.0,
        "min_sharpe_net": 0.0,
        "max_drawdown_limit": -0.45,
    },
}


def _float(value: object, default: float = 0.0) -> float:
    """Cast float with safe fallback."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_csv(raw: str) -> list[str]:
    """Parse comma-separated values."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _horizon_rule(horizon: str, settings_defaults: dict[str, float]) -> dict[str, float]:
    """Resolve per-horizon rule with explicit fallback defaults."""
    if horizon in DEFAULT_HORIZON_RULES:
        return dict(DEFAULT_HORIZON_RULES[horizon])
    return dict(settings_defaults)


def _evaluate_signal(signal_entry: dict[str, object], horizon: str, rule: dict[str, float]) -> dict[str, object]:
    """Evaluate one signal against explicit directional hypothesis rules."""
    expected_direction = str(signal_entry.get("expected_direction", "positive")).strip().lower()
    expected_sign = -1.0 if expected_direction == "negative" else 1.0
    ic = _float(signal_entry.get("information_coefficient"), 0.0)
    spread_bps = _float(signal_entry.get("decile_spread_bps"), 0.0)
    sign_alignment = _float(signal_entry.get("sign_alignment"), 0.0)
    net_mean_return = _float(signal_entry.get("net_mean_return"), float("nan"))
    sharpe_net = _float(signal_entry.get("sharpe_net"), float("nan"))
    max_drawdown = _float(signal_entry.get("max_drawdown"), float("nan"))
    total_return = _float(signal_entry.get("total_return"), float("nan"))

    directional_ic = expected_sign * ic
    directional_spread = expected_sign * spread_bps

    ic_pass = directional_ic >= float(rule["min_ic"])
    spread_pass = directional_spread >= float(rule["min_decile_spread_bps"])
    align_pass = sign_alignment >= float(rule["min_sign_alignment"])
    net_pass = net_mean_return >= float(rule["min_net_mean_return"])
    sharpe_pass = sharpe_net >= float(rule["min_sharpe_net"])
    drawdown_pass = max_drawdown >= float(rule["max_drawdown_limit"])
    passed = bool(ic_pass and spread_pass and align_pass and net_pass and sharpe_pass and drawdown_pass)

    failed_reasons: list[str] = []
    if not ic_pass:
        failed_reasons.append("ic_below_threshold")
    if not spread_pass:
        failed_reasons.append("decile_spread_below_threshold")
    if not align_pass:
        failed_reasons.append("sign_alignment_below_threshold")
    if not net_pass:
        failed_reasons.append("net_mean_return_below_threshold")
    if not sharpe_pass:
        failed_reasons.append("sharpe_net_below_threshold")
    if not drawdown_pass:
        failed_reasons.append("max_drawdown_below_limit")

    return {
        "signal": str(signal_entry.get("signal", "")),
        "horizon": horizon,
        "expected_effect": "positive" if expected_sign > 0 else "negative",
        "expected_direction": expected_direction,
        "expected_sign": int(expected_sign),
        "rationale": str(signal_entry.get("rationale", "")),
        "sample_count": int(signal_entry.get("sample_count", 0) or 0),
        "information_coefficient": ic,
        "decile_spread_bps": spread_bps,
        "sign_alignment": sign_alignment,
        "net_mean_return": net_mean_return,
        "sharpe_net": sharpe_net,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "directional_information_coefficient": directional_ic,
        "directional_decile_spread_bps": directional_spread,
        "rule": {
            "min_ic": float(rule["min_ic"]),
            "min_decile_spread_bps": float(rule["min_decile_spread_bps"]),
            "min_sign_alignment": float(rule["min_sign_alignment"]),
            "min_net_mean_return": float(rule["min_net_mean_return"]),
            "min_sharpe_net": float(rule["min_sharpe_net"]),
            "max_drawdown_limit": float(rule["max_drawdown_limit"]),
        },
        "checks": {
            "ic_pass": bool(ic_pass),
            "decile_spread_pass": bool(spread_pass),
            "sign_alignment_pass": bool(align_pass),
            "net_mean_return_pass": bool(net_pass),
            "sharpe_net_pass": bool(sharpe_pass),
            "max_drawdown_pass": bool(drawdown_pass),
        },
        "passed": passed,
        "kill_switch_recommended": bool(not passed),
        "failed_reasons": failed_reasons,
    }


def main() -> None:
    """Run script entrypoint."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate alpha-hypothesis diagnostics report")
    parser.add_argument("--summary-report", required=True, help="Summary report path")
    parser.add_argument("--output", default="reports/hypothesis_report.json", help="Output JSON path")
    parser.add_argument("--horizons", default="", help="Optional comma-separated horizon filter")
    args = parser.parse_args()

    summary_path = Path(args.summary_report)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    if not summary_path.exists():
        raise SystemExit(f"Summary report not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    symbols = payload.get("symbols", {})
    if not isinstance(symbols, dict):
        raise SystemExit("Summary report has invalid symbol payload")

    selected_horizons = set(item.lower() for item in _parse_csv(args.horizons))
    default_rule = {
        "min_ic": float(settings.alpha_kill_min_ic),
        "min_decile_spread_bps": float(settings.alpha_kill_min_decile_spread_bps),
        "min_sign_alignment": float(settings.alpha_kill_min_sign_alignment),
        "min_net_mean_return": float(settings.alpha_kill_min_signal_net_mean_return),
        "min_sharpe_net": float(settings.alpha_kill_min_signal_sharpe_net),
        "max_drawdown_limit": float(settings.alpha_kill_max_signal_drawdown_limit),
    }

    evaluated_pairs: list[dict[str, object]] = []
    per_signal_summary: dict[str, dict[str, object]] = {}
    total_signals = 0
    passed_signals = 0

    for symbol, symbol_payload in symbols.items():
        if not isinstance(symbol_payload, dict):
            continue
        horizons = symbol_payload.get("horizons", {})
        if not isinstance(horizons, dict):
            continue
        for horizon, entry in horizons.items():
            horizon_key = str(horizon).lower()
            if selected_horizons and horizon_key not in selected_horizons:
                continue
            if not isinstance(entry, dict) or entry.get("status") != "ok":
                continue
            alpha = entry.get("alpha_signal_diagnostics", {})
            if not isinstance(alpha, dict) or not bool(alpha.get("available")):
                continue
            signals = alpha.get("signals", [])
            if not isinstance(signals, list) or not signals:
                continue

            rule = _horizon_rule(horizon_key, default_rule)
            signal_rows: list[dict[str, object]] = []
            for signal_entry in signals:
                if not isinstance(signal_entry, dict):
                    continue
                row = _evaluate_signal(signal_entry, horizon_key, rule)
                signal_rows.append(row)
                total_signals += 1
                if bool(row.get("passed")):
                    passed_signals += 1
                name = str(row.get("signal", ""))
                agg = per_signal_summary.setdefault(name, {"total": 0, "passed": 0, "failed": 0})
                agg["total"] = int(agg["total"]) + 1
                if bool(row.get("passed")):
                    agg["passed"] = int(agg["passed"]) + 1
                else:
                    agg["failed"] = int(agg["failed"]) + 1

            evaluated_pairs.append(
                {
                    "symbol": str(symbol).upper(),
                    "horizon": horizon_key,
                    "signals_evaluated": len(signal_rows),
                    "signals_passed": sum(1 for row in signal_rows if bool(row.get("passed"))),
                    "signals_failed": sum(1 for row in signal_rows if not bool(row.get("passed"))),
                    "signals": signal_rows,
                }
            )

    for name, stats in per_signal_summary.items():
        total = max(int(stats.get("total", 0)), 1)
        stats["pass_rate"] = round(float(int(stats.get("passed", 0)) / total), 6)

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "summary_report": str(summary_path),
        "model_version": payload.get("model_version"),
        "feature_version": payload.get("feature_version"),
        "horizon_filter": sorted(selected_horizons),
        "rules_default": default_rule,
        "rules_by_horizon": DEFAULT_HORIZON_RULES,
        "pairs_evaluated": len(evaluated_pairs),
        "signals_total": total_signals,
        "signals_passed": passed_signals,
        "signals_failed": max(total_signals - passed_signals, 0),
        "signal_pass_rate": round(float(passed_signals / max(total_signals, 1)), 6),
        "per_signal_summary": per_signal_summary,
        "pairs": evaluated_pairs,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
