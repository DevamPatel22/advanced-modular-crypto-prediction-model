"""Unit tests for scorecard pair extraction and ablation summaries."""

import pytest

from scripts.daily_scorecard import _pair_rows, _summarize_pairs


def test_pair_rows_include_meta_ablation_deltas() -> None:
    """Scorecard row extraction should carry ablation deltas for diagnostics."""
    summary_payload = {
        "symbols": {
            "BTC-USD": {
                "horizons": {
                    "6h": {
                        "status": "ok",
                        "promotion_gate": {"passed": True, "failed_reasons": []},
                        "execution_aware_metrics": {"net_mean_return": 0.01},
                        "baseline_execution_aware_metrics": {"net_mean_return": 0.002},
                        "paper_trading_metrics": {"max_drawdown": -0.08, "total_return": 0.12},
                        "baseline_paper_trading_metrics": {"max_drawdown": -0.11, "total_return": 0.05},
                        "meta_labeling": {"test_take_rate": 0.80},
                        "ablation_metrics": {
                            "edge_filter_take_rate": 0.65,
                            "with_meta_abstention": {
                                "execution": {"net_mean_return": 0.01},
                                "paper": {"total_return": 0.12},
                            },
                            "without_meta_abstention": {
                                "execution": {"net_mean_return": 0.006},
                                "paper": {"total_return": 0.08},
                            },
                        },
                    }
                }
            }
        }
    }

    rows = _pair_rows(summary_payload)
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BTC-USD"
    assert row["horizon"] == "6h"
    assert row["edge_filter_take_rate"] == 0.65
    assert row["meta_ablation_execution_delta"] == pytest.approx(0.004)
    assert row["meta_ablation_total_return_delta"] == pytest.approx(0.04)

    summary = _summarize_pairs(rows)
    assert summary["pair_count"] == 1
    assert summary["avg_edge_filter_take_rate"] == 0.65
    assert summary["avg_meta_ablation_execution_delta"] == pytest.approx(0.004)
    assert summary["avg_meta_ablation_total_return_delta"] == pytest.approx(0.04)
