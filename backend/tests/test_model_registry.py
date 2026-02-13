from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.model_registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def test_registry_defaults_and_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            registry = ModelRegistry(models_root=root, registry_path=root / "registry.json")

            active = registry.get_active_model_version()
            self.assertIsInstance(active, str)
            self.assertTrue(len(active) > 0)

            promoted = {"BTC-USD": {"1h": True}}
            payload = registry.promote_candidate("daily-20260213", promoted)
            self.assertEqual(payload["active_model_version"], "daily-20260213")
            self.assertTrue(registry.is_promoted("BTC-USD", "1h"))
            self.assertFalse(registry.is_promoted("ETH-USD", "1h"))

    def test_resolve_artifacts_fallbacks_when_files_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            registry = ModelRegistry(models_root=root, registry_path=root / "registry.json")
            registry.promote_candidate("daily-20260213", {"BTC-USD": {"1h": True}})
            self.assertIsNone(registry.resolve_artifacts("BTC-USD", "1h"))


if __name__ == "__main__":
    unittest.main()
