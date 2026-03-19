"""Unit tests for quality-gate symbol+horizon exclusion mapping."""

from scripts.daily_retrain import _excluded_pairs_from_quality_report


def test_excluded_pairs_map_from_failing_pairs_and_cross_symbol_freshness() -> None:
    """Failing quality rows should map into horizon exclusions for selected scope."""
    payload = {
        "failing_pairs": [
            {"symbol": "BTC-USD", "granularity": "1h"},
            {"symbol": "ETH-USD", "granularity": "1h"},
            {"symbol": "IGNORED-USD", "granularity": "1h"},
        ],
        "failing_cross_symbol_freshness": [
            {"granularity": "1d"},
        ],
    }
    symbols = ["BTC-USD", "ETH-USD"]
    horizons = ["6h", "1d"]

    excluded = _excluded_pairs_from_quality_report(
        quality_payload=payload,
        symbols=symbols,
        horizons=horizons,
    )

    # 1h failures should map to both selected horizons where 1h is a candidate granularity.
    assert ("BTC-USD", "6h") in excluded
    assert ("BTC-USD", "1d") in excluded
    assert ("ETH-USD", "6h") in excluded
    assert ("ETH-USD", "1d") in excluded

    # Unknown symbol rows must be ignored.
    assert ("IGNORED-USD", "6h") not in excluded
