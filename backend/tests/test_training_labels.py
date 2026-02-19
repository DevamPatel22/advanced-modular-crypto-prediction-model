from __future__ import annotations

import unittest

try:
    import numpy as np
    from app.ml.training import _triple_barrier_labels
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _triple_barrier_labels = None  # type: ignore[assignment]


class TripleBarrierLabelTests(unittest.TestCase):
    @unittest.skipIf(np is None or _triple_barrier_labels is None, "training dependencies not available")
    def test_labels_detect_upward_touch(self) -> None:
        close = np.array([100, 101, 102, 103, 104, 105], dtype=float)
        high = np.array([100.2, 102.5, 103.0, 103.4, 104.4, 105.5], dtype=float)
        low = np.array([99.8, 100.5, 101.2, 102.1, 103.5, 104.8], dtype=float)
        sigma = np.full(close.shape[0], 0.004, dtype=float)
        labels = _triple_barrier_labels(close, high, low, sigma, steps_ahead=2, sigma_mult=1.0)
        self.assertEqual(int(labels[0]), 1)

    @unittest.skipIf(np is None or _triple_barrier_labels is None, "training dependencies not available")
    def test_labels_detect_downward_touch(self) -> None:
        close = np.array([100, 99.8, 99.5, 99.2, 99.0, 98.8], dtype=float)
        high = np.array([100.1, 100.0, 99.9, 99.5, 99.2, 99.0], dtype=float)
        low = np.array([99.9, 99.2, 98.8, 98.7, 98.5, 98.3], dtype=float)
        sigma = np.full(close.shape[0], 0.004, dtype=float)
        labels = _triple_barrier_labels(close, high, low, sigma, steps_ahead=2, sigma_mult=1.0)
        self.assertEqual(int(labels[0]), 0)


if __name__ == "__main__":
    unittest.main()
