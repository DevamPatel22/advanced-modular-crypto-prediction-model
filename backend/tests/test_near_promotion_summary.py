"""Unit tests for near-promotion summary/report payload generation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.near_promotion_retrain import _build_symbol_tree


def test_build_symbol_tree_merges_retrained_and_carried_forward_pairs(tmp_path: Path) -> None:
    """Near-loop summaries should expose a full symbol tree for downstream reports."""
    models_root = tmp_path / "models"
    metrics_path = models_root / "candidate-1" / "ETH-USD" / "metrics_1d.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "symbol": "ETH-USD",
                "horizon": "1d",
                "status": "ok",
                "promotion_gate": {"passed": True, "failed_reasons": []},
                "execution_aware_metrics": {"net_mean_return": 0.001},
                "baseline_execution_aware_metrics": {"net_mean_return": 0.0},
                "paper_trading_metrics": {"max_drawdown": -0.10, "total_return": 0.12},
                "baseline_paper_trading_metrics": {"max_drawdown": -0.30, "total_return": 0.01},
            }
        ),
        encoding="utf-8",
    )

    symbols = _build_symbol_tree(
        results=[
            {
                "symbol": "BTC-USD",
                "horizon": "6h",
                "status": "ok",
                "promotion_gate": {"passed": False, "failed_reasons": ["execution_sharpe_not_above_baseline"]},
            }
        ],
        carry_forward_pairs=[
            {
                "symbol": "ETH-USD",
                "horizon": "1d",
                "status": "copied_from_active",
            }
        ],
        models_root=models_root,
        model_version="candidate-1",
    )

    assert sorted(symbols) == ["BTC-USD", "ETH-USD"]
    assert symbols["BTC-USD"]["summary"] == {"passed": 0, "failed": 1, "insufficient": 0}
    assert symbols["ETH-USD"]["summary"] == {"passed": 1, "failed": 0, "insufficient": 0}
    assert symbols["ETH-USD"]["horizons"]["1d"]["execution_aware_metrics"]["net_mean_return"] == 0.001
