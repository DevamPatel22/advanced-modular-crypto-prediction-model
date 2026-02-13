from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings


class ModelRegistry:
    def __init__(self, models_root: Path | None = None, registry_path: Path | None = None) -> None:
        settings = get_settings()
        self.models_root = models_root or Path(settings.model_artifacts_root)
        self.registry_path = registry_path or Path(settings.model_registry_path)

    def _default_registry(self) -> dict[str, object]:
        settings = get_settings()
        return {
            "active_model_version": settings.default_model_version,
            "promoted": {},
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }

    def read(self) -> dict[str, object]:
        if not self.registry_path.exists():
            return self._default_registry()
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._default_registry()
            return payload
        except Exception:
            return self._default_registry()

    def write(self, payload: dict[str, object]) -> None:
        payload["updated_at"] = datetime.now(tz=UTC).isoformat()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_active_model_version(self) -> str:
        settings = get_settings()
        payload = self.read()
        return str(payload.get("active_model_version") or settings.default_model_version)

    def is_promoted(self, symbol: str, horizon: str) -> bool:
        payload = self.read()
        promoted = payload.get("promoted", {})
        if not isinstance(promoted, dict):
            return False
        symbol_map = promoted.get(symbol.upper(), {})
        if not isinstance(symbol_map, dict):
            return False
        return bool(symbol_map.get(horizon))

    def resolve_artifacts(self, symbol: str, horizon: str) -> dict[str, Path] | None:
        if not self.is_promoted(symbol, horizon):
            return None

        version = self.get_active_model_version()
        symbol_dir = self.models_root / version / symbol.upper()
        paths = {
            "classification": symbol_dir / f"cls_{horizon}.joblib",
            "regression": symbol_dir / f"reg_{horizon}.joblib",
            "calibration": symbol_dir / f"calibration_{horizon}.json",
            "metrics": symbol_dir / f"metrics_{horizon}.json",
        }
        if all(path.exists() for path in paths.values()):
            return paths
        return None

    def promote_candidate(self, candidate_version: str, promoted: dict[str, dict[str, bool]]) -> dict[str, object]:
        payload = self.read()
        payload["active_model_version"] = candidate_version
        payload["promoted"] = promoted
        self.write(payload)
        return payload
