"""Unit tests for synchronized training window controls."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.training import _apply_horizon_training_window
from scripts.train_all_symbols import _compute_sync_row_depth_map


def test_sync_row_depth_uses_min_row_count_with_full_coverage() -> None:
    """Row-depth sync should pick the shared minimum when all symbols are covered."""
    snapshot = [
        {"symbol": "BTC-USD", "granularity": "1h", "row_count": 10000},
        {"symbol": "ETH-USD", "granularity": "1h", "row_count": 8200},
        {"symbol": "SOL-USD", "granularity": "1h", "row_count": 9100},
        {"symbol": "BTC-USD", "granularity": "6h", "row_count": 7000},
        {"symbol": "ETH-USD", "granularity": "6h", "row_count": 6800},
        {"symbol": "SOL-USD", "granularity": "6h", "row_count": 6900},
    ]
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    granularities = ["1h", "6h"]

    sync_map, diagnostics = _compute_sync_row_depth_map(
        dataset_snapshot=snapshot,
        symbols=symbols,
        granularities=granularities,
        min_coverage_ratio=1.0,
    )

    assert sync_map == {"1h": 8200, "6h": 6800}
    assert diagnostics["1h"]["coverage_passed"] is True
    assert diagnostics["6h"]["coverage_passed"] is True


def test_sync_row_depth_blocks_granularity_when_coverage_below_threshold() -> None:
    """Granularity should be excluded from sync map when coverage is below required ratio."""
    snapshot = [
        {"symbol": "BTC-USD", "granularity": "1h", "row_count": 10000},
        {"symbol": "ETH-USD", "granularity": "1h", "row_count": 8200},
        {"symbol": "SOL-USD", "granularity": "1h", "row_count": 9100},
        {"symbol": "BTC-USD", "granularity": "6h", "row_count": 7000},
        {"symbol": "ETH-USD", "granularity": "6h", "row_count": 6800},
    ]
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    granularities = ["1h", "6h"]

    sync_map, diagnostics = _compute_sync_row_depth_map(
        dataset_snapshot=snapshot,
        symbols=symbols,
        granularities=granularities,
        min_coverage_ratio=1.0,
    )

    assert sync_map == {"1h": 8200}
    assert diagnostics["6h"]["coverage_ratio"] < 1.0
    assert diagnostics["6h"]["coverage_passed"] is False


def test_horizon_window_keeps_recent_tail_for_shorter_horizon() -> None:
    """Short/medium horizons should train on a recent tail instead of the full archive."""
    frame = pd.DataFrame({"start_time": list(range(15000)), "close": [1.0] * 15000})

    trimmed, meta = _apply_horizon_training_window(frame, horizon="6h", required_rows=400)

    assert len(trimmed) == 9000
    assert int(trimmed["start_time"].iloc[0]) == 6000
    assert int(trimmed["start_time"].iloc[-1]) == 14999
    assert meta["trim_applied"] is True


def test_horizon_window_never_cuts_below_required_training_floor() -> None:
    """Configured caps should not reduce the frame below the minimum safe training depth."""
    frame = pd.DataFrame({"start_time": list(range(12000)), "close": [1.0] * 12000})

    trimmed, meta = _apply_horizon_training_window(frame, horizon="1d", required_rows=8000)

    assert len(trimmed) == 8000
    assert int(trimmed["start_time"].iloc[0]) == 4000
    assert meta["trim_applied"] is True
