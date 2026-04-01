"""Unit tests for shadow trading report aggregation."""

from scripts.shadow_trading import build_shadow_report_payload


def test_shadow_report_payload_summarizes_settled_rows() -> None:
    """Settled rows should aggregate into pair and overall shadow stats."""
    rows = []
    for idx in range(5):
        rows.append(
            {
                "status": "settled",
                "model_version": "shadow-v1",
                "symbol": "BTC-USD",
                "horizon": "6h",
                "direction_hit": True,
                "absolute_error_usd": 10.0 + idx,
                "absolute_error_pct": 0.01,
                "net_return_after_costs": 0.002,
            }
        )

    report = build_shadow_report_payload(rows)
    assert report["overall"]["signal_count"] == 5
    assert report["overall"]["pair_count"] == 1
    assert report["overall"]["direction_accuracy"] == 1.0
    assert report["pairs"][0]["passes_shadow_gate"] is True
    assert report["pairs"][0]["avg_net_return_after_costs"] == 0.002
    assert report["model_versions"] == ["shadow-v1"]
    assert report["models"]["shadow-v1"]["overall"]["passing_pairs"] == 1


def test_shadow_report_payload_groups_multiple_model_versions() -> None:
    """Mixed-version shadow rows should be preserved per model instead of mislabeled as one run."""
    rows = [
        {
            "status": "settled",
            "model_version": "shadow-v1",
            "symbol": "BTC-USD",
            "horizon": "6h",
            "direction_hit": True,
            "absolute_error_usd": 5.0,
            "absolute_error_pct": 0.01,
            "net_return_after_costs": 0.002,
        },
        {
            "status": "settled",
            "model_version": "shadow-v2",
            "symbol": "ETH-USD",
            "horizon": "1d",
            "direction_hit": False,
            "absolute_error_usd": 15.0,
            "absolute_error_pct": 0.03,
            "net_return_after_costs": -0.001,
        },
    ]

    report = build_shadow_report_payload(rows)
    assert report["model_version"] is None
    assert report["mixed_model_versions"] is True
    assert report["model_versions"] == ["shadow-v1", "shadow-v2"]
    assert report["models"]["shadow-v1"]["overall"]["pair_count"] == 1
    assert report["models"]["shadow-v2"]["overall"]["pair_count"] == 1
