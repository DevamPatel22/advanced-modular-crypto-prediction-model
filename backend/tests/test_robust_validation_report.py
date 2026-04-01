"""Unit tests for robust validation aggregation."""

from scripts.robust_validation_report import (
    _aggregate_oos,
    _aggregate_regimes,
    _aggregate_stress,
    _json_safe,
    _pair_row,
    _shadow_summary_for_model,
)


def test_pair_row_and_aggregates_capture_robust_passes() -> None:
    """Robust report helpers should reflect strong pairs as passing."""
    metrics = {
        "model_version": "candidate-x",
        "promotion_gate": {"passed": True, "failed_reasons": []},
        "walk_forward_enforced": True,
        "walk_forward_gate": {"strict_pass_all_folds": True},
        "data_leakage_checks": {"pass": True},
        "execution_aware_metrics": {
            "net_mean_return": 0.001,
            "sharpe_net": 1.5,
            "probabilistic_sharpe_gt_zero": 0.97,
        },
        "paper_trading_metrics": {
            "total_return": 0.12,
            "max_drawdown": -0.10,
        },
        "regime_breakdown": {
            "up": {
                "count": 100,
                "accuracy": 0.60,
                "f1": 0.58,
                "net_mean_return": 0.0012,
                "sharpe_net": 1.8,
                "total_return": 0.10,
                "max_drawdown": -0.08,
            }
        },
        "execution_stress_metrics": {
            "cost_x2": {
                "execution": {"net_mean_return": 0.0004, "sharpe_net": 0.9},
                "paper": {"total_return": 0.04, "max_drawdown": -0.12},
                "survives_basic_gate": True,
            }
        },
    }

    row = _pair_row("BTC-USD", "6h", metrics, drawdown_floor=-0.45)
    assert row["passes_robust_pair_rules"] is True

    oos = _aggregate_oos([row])
    assert oos["pair_pass_rate"] == 1.0
    assert oos["walk_forward_enforced_count"] == 1
    assert oos["walk_forward_pass_count"] == 1

    regimes = _aggregate_regimes([row], drawdown_floor=-0.45)
    assert regimes["up"]["robust_pass_rate"] == 1.0
    assert regimes["up"]["avg_sharpe_net"] == 1.8

    stress = _aggregate_stress([row], drawdown_floor=-0.45)
    assert stress["cost_x2"]["survival_rate"] == 1.0
    assert stress["cost_x2"]["avg_total_return"] == 0.04


def test_shadow_summary_requires_matching_model_and_enough_signals() -> None:
    """Robust shadow validation should reject mismatched or undersized evidence."""
    shadow_payload = {
        "generated_at": "2026-04-01T00:00:00+00:00",
        "model_versions": ["shadow-v1", "shadow-v2"],
        "models": {
            "shadow-v1": {
                "overall": {
                    "signal_count": 10,
                    "pair_count": 2,
                    "passing_pairs": 2,
                },
                "pairs": [],
            }
        },
    }

    missing = _shadow_summary_for_model(shadow_payload, model_version="shadow-v2", promoted_pair_count=2)
    assert missing["matches_model_version"] is False
    assert missing["shadow_gate_passed"] is False

    matched = _shadow_summary_for_model(shadow_payload, model_version="shadow-v1", promoted_pair_count=2)
    assert matched["matches_model_version"] is True
    assert matched["required_signal_count"] == 10
    assert matched["shadow_gate_passed"] is True

    undersized = _shadow_summary_for_model(shadow_payload, model_version="shadow-v1", promoted_pair_count=3)
    assert undersized["shadow_gate_passed"] is False


def test_json_safe_replaces_nan_with_null_compatible_values() -> None:
    """Robust validation output should not emit NaN tokens."""
    payload = {"value": float("nan"), "nested": [1.0, float("inf"), {"x": float("-inf")}]}

    safe = _json_safe(payload)
    assert safe == {"value": None, "nested": [1.0, None, {"x": None}]}
