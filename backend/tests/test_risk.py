from __future__ import annotations

import unittest

try:
    from app.services.predictor import _prediction_range_and_risk
except Exception:  # pragma: no cover
    _prediction_range_and_risk = None  # type: ignore[assignment]


class RiskRangeTests(unittest.TestCase):
    @unittest.skipIf(_prediction_range_and_risk is None, "predictor dependencies not available")
    def test_bounds_are_monotonic(self) -> None:
        min_pct, max_pct, risk_score, risk_level = _prediction_range_and_risk("up", 0.7, 0.02)
        self.assertLessEqual(min_pct, max_pct)
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 100.0)
        self.assertIn(risk_level, {"low", "medium", "high"})

    @unittest.skipIf(_prediction_range_and_risk is None, "predictor dependencies not available")
    def test_higher_volatility_increases_risk_score(self) -> None:
        _min_a, _max_a, low_score, _ = _prediction_range_and_risk("up", 0.7, 0.01)
        _min_b, _max_b, high_score, _ = _prediction_range_and_risk("up", 0.7, 0.04)
        self.assertGreater(high_score, low_score)


if __name__ == "__main__":
    unittest.main()
