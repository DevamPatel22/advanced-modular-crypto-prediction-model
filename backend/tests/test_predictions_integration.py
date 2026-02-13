from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

try:
    from app.config import get_settings
    from app.schemas.prediction import PredictionRequest
    from app.services.predictor import generate_prediction
except Exception:  # pragma: no cover
    get_settings = None  # type: ignore[assignment]
    PredictionRequest = None  # type: ignore[assignment]
    generate_prediction = None  # type: ignore[assignment]


class PredictionFallbackIntegrationTests(unittest.TestCase):
    @unittest.skipIf(generate_prediction is None or PredictionRequest is None or get_settings is None, "predictor dependencies not available")
    def test_prediction_falls_back_without_promoted_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "market_data.db"
            models_root = Path(tmp) / "models"
            registry = models_root / "registry.json"

            os.environ["MARKET_DATA_SQLITE_PATH"] = str(db_path)
            os.environ["MODEL_ARTIFACTS_ROOT"] = str(models_root)
            os.environ["MODEL_REGISTRY_PATH"] = str(registry)
            get_settings.cache_clear()

            payload = PredictionRequest(symbol="BTC-USD", horizon="1h", include_debug=True)
            response = generate_prediction(payload)

            self.assertEqual(response.symbol, "BTC-USD")
            self.assertIn(response.direction, {"up", "down"})
            self.assertIsNotNone(response.debug)
            self.assertEqual(response.debug.get("inference_mode"), "fallback")

            get_settings.cache_clear()


if __name__ == "__main__":
    unittest.main()
