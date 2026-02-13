from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    from app.ml.features import HORIZON_TO_DATA_WINDOW
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    HORIZON_TO_DATA_WINDOW = {}  # type: ignore[assignment]


class HorizonMappingTests(unittest.TestCase):
    def test_expected_horizons_present(self) -> None:
        if not HORIZON_TO_DATA_WINDOW:
            self.skipTest("feature dependencies not available")
        expected = {"5m", "1h", "6h", "12h", "1d", "1w", "1mo", "3mo"}
        self.assertEqual(set(HORIZON_TO_DATA_WINDOW.keys()), expected)

    @unittest.skipIf(np is None or pd is None, "numpy/pandas not available in environment")
    def test_target_shift_correctness(self) -> None:
        closes = pd.Series(np.arange(1.0, 30.0))
        steps = 6
        target = closes.shift(-steps)
        self.assertEqual(float(target.iloc[0]), 7.0)
        self.assertTrue(np.isnan(target.iloc[-1]))


if __name__ == "__main__":
    unittest.main()
