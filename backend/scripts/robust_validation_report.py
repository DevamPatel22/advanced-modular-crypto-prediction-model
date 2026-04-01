#!/usr/bin/env python3
"""Build a robust-validation bundle for the active or specified model version."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.model_registry import ModelRegistry
from app.services.shadow_book import shadow_report_path


def _safe_load(path: Path) -> dict[str, object] | None:
    """Read JSON safely."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _float(value: object, default: float = math.nan) -> float:
    """Cast a numeric field safely."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _json_safe(value: object) -> object:
    """Recursively replace non-finite floats with JSON-safe nulls."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _promoted_map_for_version(model_version: str, registry_payload: dict[str, object]) -> dict[str, dict[str, bool]]:
    """Resolve the promoted pair map for a model version from registry lineage."""
    if str(registry_payload.get("active_model_version", "")) == model_version:
        promoted = registry_payload.get("promoted", {})
        return promoted if isinstance(promoted, dict) else {}
    history = registry_payload.get("history", [])
    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            if str(item.get("to_active", "")) != model_version:
                continue
            promoted = item.get("to_promoted", {})
            return promoted if isinstance(promoted, dict) else {}
    return {}


def _pair_row(symbol: str, horizon: str, metrics: dict[str, object], drawdown_floor: float) -> dict[str, object]:
    """Normalize one promoted pair row from metrics artifacts."""
    execution = metrics.get("execution_aware_metrics", {})
    paper = metrics.get("paper_trading_metrics", {})
    leakage = metrics.get("data_leakage_checks", {})
    walk_forward = metrics.get("walk_forward_gate", {})
    regime_breakdown = metrics.get("regime_breakdown", {})
    stress = metrics.get("execution_stress_metrics", {})

    if not isinstance(execution, dict):
        execution = {}
    if not isinstance(paper, dict):
        paper = {}
    if not isinstance(leakage, dict):
        leakage = {}
    if not isinstance(walk_forward, dict):
        walk_forward = {}
    if not isinstance(regime_breakdown, dict):
        regime_breakdown = {}
    if not isinstance(stress, dict):
        stress = {}

    net_mean_return = _float(execution.get("net_mean_return"))
    sharpe_net = _float(execution.get("sharpe_net"))
    total_return = _float(paper.get("total_return"))
    max_drawdown = _float(paper.get("max_drawdown"))
    pair_pass = bool(
        net_mean_return > 0.0
        and sharpe_net > 0.0
        and total_return > 0.0
        and max_drawdown >= drawdown_floor
        and bool(metrics.get("walk_forward_enforced"))
        and bool(walk_forward.get("strict_pass_all_folds"))
        and bool(leakage.get("pass"))
    )
    return {
        "symbol": symbol,
        "horizon": horizon,
        "model_version": metrics.get("model_version"),
        "promotion_gate_passed": bool(metrics.get("promotion_gate", {}).get("passed")) if isinstance(metrics.get("promotion_gate"), dict) else False,
        "walk_forward_enforced": bool(metrics.get("walk_forward_enforced")),
        "walk_forward_passed": bool(walk_forward.get("strict_pass_all_folds")),
        "leakage_passed": bool(leakage.get("pass")),
        "net_mean_return": net_mean_return,
        "sharpe_net": sharpe_net,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "probabilistic_sharpe_gt_zero": _float(execution.get("probabilistic_sharpe_gt_zero")),
        "sample_count": _float(execution.get("sample_count"), 0.0),
        "regime_breakdown": regime_breakdown,
        "execution_stress_metrics": stress,
        "passes_robust_pair_rules": pair_pass,
    }


def _aggregate_oos(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate OOS-like financial metrics across promoted pairs."""
    if not rows:
        return {"pair_count": 0, "available": False}
    return {
        "available": True,
        "pair_count": len(rows),
        "pair_pass_count": sum(bool(row.get("passes_robust_pair_rules")) for row in rows),
        "pair_pass_rate": float(sum(bool(row.get("passes_robust_pair_rules")) for row in rows) / max(len(rows), 1)),
        "positive_net_count": sum(_float(row.get("net_mean_return"), 0.0) > 0.0 for row in rows),
        "positive_total_return_count": sum(_float(row.get("total_return"), 0.0) > 0.0 for row in rows),
        "positive_sharpe_count": sum(_float(row.get("sharpe_net"), 0.0) > 0.0 for row in rows),
        "walk_forward_enforced_count": sum(bool(row.get("walk_forward_enforced")) for row in rows),
        "walk_forward_pass_count": sum(bool(row.get("walk_forward_passed")) for row in rows),
        "leakage_pass_count": sum(bool(row.get("leakage_passed")) for row in rows),
        "avg_net_mean_return": float(sum(_float(row.get("net_mean_return"), 0.0) for row in rows) / len(rows)),
        "avg_sharpe_net": float(sum(_float(row.get("sharpe_net"), 0.0) for row in rows) / len(rows)),
        "avg_total_return": float(sum(_float(row.get("total_return"), 0.0) for row in rows) / len(rows)),
        "avg_max_drawdown": float(sum(_float(row.get("max_drawdown"), 0.0) for row in rows) / len(rows)),
        "avg_probabilistic_sharpe_gt_zero": float(
            sum(
                _float(row.get("probabilistic_sharpe_gt_zero"), 0.0)
                for row in rows
                if math.isfinite(_float(row.get("probabilistic_sharpe_gt_zero")))
            )
            / max(
                sum(math.isfinite(_float(row.get("probabilistic_sharpe_gt_zero"))) for row in rows),
                1,
            )
        ),
    }


def _aggregate_regimes(rows: list[dict[str, object]], drawdown_floor: float) -> dict[str, dict[str, object]]:
    """Aggregate regime-wise diagnostics across promoted pairs."""
    buckets: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        regime_breakdown = row.get("regime_breakdown", {})
        if not isinstance(regime_breakdown, dict):
            continue
        for regime, payload in regime_breakdown.items():
            if not isinstance(payload, dict):
                continue
            buckets.setdefault(str(regime), []).append(
                {
                    "count": _float(payload.get("count"), 0.0),
                    "accuracy": _float(payload.get("accuracy")),
                    "f1": _float(payload.get("f1")),
                    "net_mean_return": _float(payload.get("net_mean_return")),
                    "sharpe_net": _float(payload.get("sharpe_net")),
                    "total_return": _float(payload.get("total_return")),
                    "max_drawdown": _float(payload.get("max_drawdown")),
                }
            )

    summary: dict[str, dict[str, object]] = {}
    for regime, items in buckets.items():
        pair_count = len(items)
        robust_pass_count = sum(
            item["net_mean_return"] > 0.0 and item["sharpe_net"] > 0.0 and item["max_drawdown"] >= drawdown_floor
            for item in items
            if math.isfinite(item["net_mean_return"]) and math.isfinite(item["sharpe_net"]) and math.isfinite(item["max_drawdown"])
        )
        summary[regime] = {
            "pair_count": pair_count,
            "robust_pass_count": robust_pass_count,
            "robust_pass_rate": float(robust_pass_count / max(pair_count, 1)),
            "avg_accuracy": float(sum(item["accuracy"] for item in items if math.isfinite(item["accuracy"])) / max(sum(math.isfinite(item["accuracy"]) for item in items), 1)),
            "avg_f1": float(sum(item["f1"] for item in items if math.isfinite(item["f1"])) / max(sum(math.isfinite(item["f1"]) for item in items), 1)),
            "avg_net_mean_return": float(sum(item["net_mean_return"] for item in items if math.isfinite(item["net_mean_return"])) / max(sum(math.isfinite(item["net_mean_return"]) for item in items), 1)),
            "avg_sharpe_net": float(sum(item["sharpe_net"] for item in items if math.isfinite(item["sharpe_net"])) / max(sum(math.isfinite(item["sharpe_net"]) for item in items), 1)),
            "avg_total_return": float(sum(item["total_return"] for item in items if math.isfinite(item["total_return"])) / max(sum(math.isfinite(item["total_return"]) for item in items), 1)),
            "avg_max_drawdown": float(sum(item["max_drawdown"] for item in items if math.isfinite(item["max_drawdown"])) / max(sum(math.isfinite(item["max_drawdown"]) for item in items), 1)),
        }
    return summary


def _aggregate_stress(rows: list[dict[str, object]], drawdown_floor: float) -> dict[str, dict[str, object]]:
    """Aggregate execution-stress scenario metrics across promoted pairs."""
    scenario_buckets: dict[str, list[dict[str, float | bool]]] = {}
    for row in rows:
        stress = row.get("execution_stress_metrics", {})
        if not isinstance(stress, dict):
            continue
        for scenario_name, payload in stress.items():
            if not isinstance(payload, dict):
                continue
            execution = payload.get("execution", {})
            paper = payload.get("paper", {})
            if not isinstance(execution, dict):
                execution = {}
            if not isinstance(paper, dict):
                paper = {}
            scenario_buckets.setdefault(str(scenario_name), []).append(
                {
                    "net_mean_return": _float(execution.get("net_mean_return")),
                    "sharpe_net": _float(execution.get("sharpe_net")),
                    "total_return": _float(paper.get("total_return")),
                    "max_drawdown": _float(paper.get("max_drawdown")),
                    "survives_basic_gate": bool(payload.get("survives_basic_gate")),
                }
            )

    out: dict[str, dict[str, object]] = {}
    for scenario_name, items in scenario_buckets.items():
        pair_count = len(items)
        survives = sum(bool(item.get("survives_basic_gate")) for item in items)
        out[scenario_name] = {
            "pair_count": pair_count,
            "survival_count": survives,
            "survival_rate": float(survives / max(pair_count, 1)),
            "avg_net_mean_return": float(sum(_float(item.get("net_mean_return"), 0.0) for item in items) / max(pair_count, 1)),
            "avg_sharpe_net": float(sum(_float(item.get("sharpe_net"), 0.0) for item in items) / max(pair_count, 1)),
            "avg_total_return": float(sum(_float(item.get("total_return"), 0.0) for item in items) / max(pair_count, 1)),
            "avg_max_drawdown": float(sum(_float(item.get("max_drawdown"), 0.0) for item in items) / max(pair_count, 1)),
            "drawdown_floor": drawdown_floor,
        }
    return out


def _shadow_summary_for_model(
    shadow_payload: dict[str, object] | None,
    *,
    model_version: str,
    promoted_pair_count: int,
) -> dict[str, object]:
    """Resolve the latest shadow-validation payload for one model version."""
    if shadow_payload is None:
        return {
            "available": False,
            "matches_model_version": False,
            "shadow_gate_passed": False,
            "message": "No shadow report found yet",
        }

    models_payload = shadow_payload.get("models", {})
    model_payload: dict[str, object] | None = None
    if isinstance(models_payload, dict):
        candidate = models_payload.get(model_version)
        model_payload = candidate if isinstance(candidate, dict) else None
    if model_payload is None and str(shadow_payload.get("model_version", "")) == model_version:
        model_payload = shadow_payload
    if model_payload is None:
        return {
            "available": True,
            "path": str(shadow_report_path()),
            "generated_at": shadow_payload.get("generated_at"),
            "matches_model_version": False,
            "model_versions": shadow_payload.get("model_versions", []),
            "shadow_gate_passed": False,
            "message": f"Latest shadow report does not contain results for model version: {model_version}",
        }

    overall = model_payload.get("overall", {})
    pairs = model_payload.get("pairs", [])
    if not isinstance(overall, dict):
        overall = {}
    if not isinstance(pairs, list):
        pairs = []

    signal_count = int(_float(overall.get("signal_count"), 0.0))
    pair_count = int(_float(overall.get("pair_count"), 0.0))
    passing_pairs = int(_float(overall.get("passing_pairs"), 0.0))
    required_signal_count = max(promoted_pair_count, 1) * 5
    shadow_gate_passed = bool(
        signal_count >= required_signal_count
        and pair_count >= promoted_pair_count
        and passing_pairs >= promoted_pair_count
    )
    return {
        "available": True,
        "path": str(shadow_report_path()),
        "generated_at": shadow_payload.get("generated_at"),
        "matches_model_version": True,
        "model_version": model_version,
        "overall": overall,
        "pairs": pairs,
        "signal_count": signal_count,
        "pair_count": pair_count,
        "passing_pairs": passing_pairs,
        "required_signal_count": required_signal_count,
        "shadow_gate_passed": shadow_gate_passed,
    }


def main() -> None:
    """Run the script entrypoint."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate robust validation report for a model version")
    parser.add_argument("--model-version", default="", help="Model version to audit (defaults to active)")
    parser.add_argument("--output", default="reports/robust_validation_latest.json", help="Output JSON path")
    args = parser.parse_args()

    registry = ModelRegistry()
    registry_payload = registry.read()
    model_version = args.model_version.strip() or registry.get_active_model_version()
    promoted = _promoted_map_for_version(model_version, registry_payload)
    if not promoted:
        raise SystemExit(f"No promoted pairs found for model version: {model_version}")

    drawdown_floor = float(settings.promotion_max_drawdown_limit)
    rows: list[dict[str, object]] = []
    missing_metrics: list[str] = []
    for symbol, horizons in promoted.items():
        if not isinstance(horizons, dict):
            continue
        for horizon, enabled in horizons.items():
            if not bool(enabled):
                continue
            metrics_path = Path(settings.model_artifacts_root) / model_version / str(symbol).upper() / f"metrics_{horizon}.json"
            payload = _safe_load(metrics_path)
            if payload is None:
                missing_metrics.append(f"{symbol}:{horizon}")
                continue
            rows.append(_pair_row(str(symbol).upper(), str(horizon).lower(), payload, drawdown_floor))

    oos_summary = _aggregate_oos(rows)
    regime_summary = _aggregate_regimes(rows, drawdown_floor)
    stress_summary = _aggregate_stress(rows, drawdown_floor)
    shadow_summary = _shadow_summary_for_model(
        _safe_load(shadow_report_path()),
        model_version=model_version,
        promoted_pair_count=len(rows),
    )
    regime_gate_pass = bool(regime_summary) and all(
        float(item.get("robust_pass_rate", 0.0)) >= 0.50 for item in regime_summary.values()
    )
    stress_gate_pass = bool(stress_summary) and all(
        float(item.get("survival_rate", 0.0)) >= 0.50 for item in stress_summary.values()
    )
    robust_gate_pass = bool(
        oos_summary.get("available")
        and float(oos_summary.get("pair_pass_rate", 0.0)) >= 0.95
        and int(oos_summary.get("walk_forward_enforced_count", 0)) == len(rows)
        and int(oos_summary.get("walk_forward_pass_count", 0)) == len(rows)
        and regime_gate_pass
        and stress_gate_pass
        and bool(shadow_summary.get("shadow_gate_passed"))
    )

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": model_version,
        "active_model_version": registry_payload.get("active_model_version"),
        "promoted_pair_count": len(rows),
        "missing_metrics_pairs": missing_metrics,
        "rules": {
            "pair_rule": {
                "net_mean_return": "> 0",
                "sharpe_net": "> 0",
                "total_return": "> 0",
                "max_drawdown": f">= {drawdown_floor}",
                "walk_forward_enforced": True,
                "walk_forward_passed": True,
                "leakage_passed": True,
            },
            "robust_gate": {
                "pair_pass_rate_min": 0.95,
                "walk_forward_enforced_all_pairs": True,
                "walk_forward_pass_all_pairs": True,
                "regime_robust_pass_rate_min": 0.50,
                "stress_survival_rate_min": 0.50,
                "shadow_pairs_match_promoted": True,
                "shadow_signal_count_per_pair_min": 5,
                "shadow_passing_pairs_match_promoted": True,
            },
        },
        "oos_summary": oos_summary,
        "regime_summary": regime_summary,
        "stress_summary": stress_summary,
        "shadow_validation": shadow_summary,
        "regime_gate_passed": regime_gate_pass,
        "stress_gate_passed": stress_gate_pass,
        "robust_alpha_gate_passed": robust_gate_pass,
        "pairs": rows,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_report = _json_safe(report)
    output_path.write_text(json.dumps(safe_report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(safe_report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
