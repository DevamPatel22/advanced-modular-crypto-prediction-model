#!/usr/bin/env python3
"""Freeze the current active model version for live shadow trading."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_registry import ModelRegistry
from app.services.shadow_book import write_shadow_champion


def main() -> None:
    """Run the script entrypoint."""
    parser = argparse.ArgumentParser(description="Freeze a champion model for live shadow trading")
    parser.add_argument("--model-version", default="", help="Optional explicit model version (defaults to active)")
    parser.add_argument("--note", default="", help="Optional operator note")
    args = parser.parse_args()

    registry = ModelRegistry()
    registry_payload = registry.read()
    model_version = args.model_version.strip() or registry.get_active_model_version()
    promoted = registry_payload.get("promoted", {})
    if not isinstance(promoted, dict) or not promoted:
        raise SystemExit("No promoted pairs available to freeze for shadow trading")

    payload = {
        "frozen_at": datetime.now(tz=UTC).isoformat(),
        "model_version": model_version,
        "promoted": promoted,
        "note": args.note.strip() or None,
    }
    path = write_shadow_champion(payload)
    print(json.dumps({"status": "ok", "shadow_champion_path": str(path), "model_version": model_version}, indent=2))


if __name__ == "__main__":
    main()
