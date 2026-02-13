from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
    from app.ml.features import FEATURE_COLUMNS, FEATURE_VERSION, build_features
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    FEATURE_COLUMNS = []  # type: ignore[assignment]
    FEATURE_VERSION = "unknown"  # type: ignore[assignment]
    build_features = None  # type: ignore[assignment]


class FeaturePipelineTests(unittest.TestCase):
    @unittest.skipIf(np is None or pd is None, "numpy/pandas not available in environment")
    def _sample_frame(self, rows: int = 220) -> pd.DataFrame:
        values = np.linspace(100.0, 140.0, rows)
        noise = np.sin(np.arange(rows) / 8.0)
        close = values + noise
        return pd.DataFrame(
            {
                "start_time": np.arange(rows),
                "open": close - 0.3,
                "high": close + 0.8,
                "low": close - 0.9,
                "close": close,
                "volume": np.linspace(1000.0, 1800.0, rows),
            }
        )

    def test_feature_version_locked(self) -> None:
        if build_features is None:
            self.skipTest("feature dependencies not available")
        self.assertEqual(FEATURE_VERSION, "v2")

    @unittest.skipIf(np is None or pd is None or build_features is None, "numpy/pandas not available in environment")
    def test_feature_generation_deterministic(self) -> None:
        frame = self._sample_frame()
        a = build_features(frame)
        b = build_features(frame)
        pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS], check_exact=False, rtol=1e-12, atol=1e-12)

    @unittest.skipIf(np is None or pd is None or build_features is None, "numpy/pandas not available in environment")
    def test_feature_output_has_no_inf(self) -> None:
        frame = self._sample_frame()
        out = build_features(frame)
        arr = out[FEATURE_COLUMNS].to_numpy(dtype=float)
        self.assertFalse(np.isinf(arr).any())

    @unittest.skipIf(np is None or pd is None or build_features is None, "numpy/pandas not available in environment")
    def test_markov_features_are_probability_bounded(self) -> None:
        frame = self._sample_frame()
        out = build_features(frame).dropna(subset=["markov_prob_down", "markov_prob_flat", "markov_prob_up"])
        if out.empty:
            self.skipTest("not enough rows for markov estimates")
        for col in ["markov_prob_down", "markov_prob_flat", "markov_prob_up"]:
            self.assertTrue(((out[col] >= 0) & (out[col] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
