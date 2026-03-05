#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_registry import ModelRegistry


@dataclass(frozen=True)
class Profile:
    name: str
    env_overrides: dict[str, str]


PROFILE_LIBRARY: dict[str, Profile] = {
    "default": Profile(name="default", env_overrides={}),
    "triple_barrier_low_sigma": Profile(
        name="triple_barrier_low_sigma",
        env_overrides={
            "CLASSIFICATION_LABEL_MODE": "triple_barrier",
            "TRIPLE_BARRIER_SIGMA_MULT": "0.85",
            "REGIME_MODELS_ENABLED": "true",
        },
    ),
    "triple_barrier_high_sigma": Profile(
        name="triple_barrier_high_sigma",
        env_overrides={
            "CLASSIFICATION_LABEL_MODE": "triple_barrier",
            "TRIPLE_BARRIER_SIGMA_MULT": "1.20",
            "REGIME_MODELS_ENABLED": "true",
        },
    ),
    "directional_regime": Profile(
        name="directional_regime",
        env_overrides={
            "CLASSIFICATION_LABEL_MODE": "directional",
            "REGIME_MODELS_ENABLED": "true",
        },
    ),
    "directional_no_regime": Profile(
        name="directional_no_regime",
        env_overrides={
            "CLASSIFICATION_LABEL_MODE": "directional",
            "REGIME_MODELS_ENABLED": "false",
        },
    ),
    "triple_no_regime": Profile(
        name="triple_no_regime",
        env_overrides={
            "CLASSIFICATION_LABEL_MODE": "triple_barrier",
            "TRIPLE_BARRIER_SIGMA_MULT": "1.00",
            "REGIME_MODELS_ENABLED": "false",
        },
    ),
}


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in raw.split(","):
        text = token.strip()
        if not text:
            continue
        if ":" not in text:
            raise SystemExit(f"Invalid pair '{text}'. Expected SYMBOL:HORIZON.")
        symbol_raw, horizon_raw = [part.strip() for part in text.split(":", 1)]
        pair = (symbol_raw.upper(), horizon_raw.lower())
        if not pair[0] or not pair[1] or pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def _parse_profiles(raw: str) -> list[Profile]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        names = ["default"]
    profiles: list[Profile] = []
    for name in names:
        profile = PROFILE_LIBRARY.get(name)
        if profile is None:
            available = ", ".join(sorted(PROFILE_LIBRARY))
            raise SystemExit(f"Unknown profile '{name}'. Available: {available}")
        profiles.append(profile)
    return profiles


def _promoted_set(payload: dict[str, object]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    promoted = payload.get("promoted", {})
    if not isinstance(promoted, dict):
        return out
    for symbol, horizons in promoted.items():
        if not isinstance(horizons, dict):
            continue
        for horizon, value in horizons.items():
            if bool(value):
                out.add((str(symbol).upper(), str(horizon)))
    return out


def _extract_json_lines(stdout: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _default_output_path() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "reports" / f"strict_promotion_loop_{stamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict-only sequential promotion attempts for target symbol/horizon pairs")
    parser.add_argument("--pairs", required=True, help="Comma-separated SYMBOL:HORIZON list")
    parser.add_argument("--symbols", default="BTC-USD,ETH-USD,SOL-USD", help="Symbol universe forwarded to near_promotion_retrain")
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3"], default="phase3")
    parser.add_argument("--max-attempts-per-pair", type=int, default=8)
    parser.add_argument("--attempt-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--profiles",
        default="default,triple_barrier_low_sigma,triple_barrier_high_sigma,directional_regime,directional_no_regime,triple_no_regime",
        help="Comma-separated profile names to cycle across attempts",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Forward --skip-ingest to near_promotion_retrain")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    pairs = _parse_pairs(args.pairs)
    if not pairs:
        raise SystemExit("No valid pairs supplied")
    profiles = _parse_profiles(args.profiles)
    max_attempts = max(1, int(args.max_attempts_per_pair))
    attempt_timeout_seconds = max(120, int(args.attempt_timeout_seconds))

    registry = ModelRegistry()
    registry_before = registry.read()
    promoted_before = _promoted_set(registry_before)
    active_before = registry_before.get("active_model_version")

    pair_reports: list[dict[str, object]] = []
    total_attempts = 0

    for symbol, horizon in pairs:
        if (symbol, horizon) in promoted_before:
            pair_reports.append(
                {
                    "pair": f"{symbol}:{horizon}",
                    "status": "already_promoted",
                    "attempts": [],
                }
            )
            continue

        attempts: list[dict[str, object]] = []
        promoted = False
        for attempt_index in range(1, max_attempts + 1):
            profile = profiles[(attempt_index - 1) % len(profiles)]
            total_attempts += 1
            model_version = f"strictseq-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}-{symbol.lower().replace('-', '')}-{horizon}-a{attempt_index}"
            print(
                json.dumps(
                    {
                        "pair": f"{symbol}:{horizon}",
                        "attempt": attempt_index,
                        "profile": profile.name,
                        "model_version": model_version,
                    }
                ),
                flush=True,
            )

            command = [
                sys.executable,
                "scripts/near_promotion_retrain.py",
                "--phase",
                args.phase,
                "--symbols",
                args.symbols,
                "--max-pairs",
                "1",
                "--target-pairs",
                f"{symbol}:{horizon}",
                "--model-version",
                model_version,
            ]
            if args.skip_ingest:
                command.append("--skip-ingest")

            env = os.environ.copy()
            env.update(profile.env_overrides)
            try:
                proc = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=attempt_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "profile": profile.name,
                        "env_overrides": profile.env_overrides,
                        "model_version": model_version,
                        "command_return_code": -1,
                        "pair_evaluation": {},
                        "status_payload": {},
                        "promoted_after_attempt": False,
                        "stderr_tail": (exc.stderr or "")[-1200:],
                        "timeout_seconds": attempt_timeout_seconds,
                    }
                )
                continue

            json_lines = _extract_json_lines(proc.stdout)
            pair_line = next((item for item in json_lines if "pair" in item), {})
            status_line = next((item for item in reversed(json_lines) if "status" in item and "model_version" in item), {})

            registry_now = ModelRegistry().read()
            promoted_now = _promoted_set(registry_now)
            promoted = (symbol, horizon) in promoted_now

            attempts.append(
                {
                    "attempt": attempt_index,
                    "profile": profile.name,
                    "env_overrides": profile.env_overrides,
                    "model_version": model_version,
                    "command_return_code": int(proc.returncode),
                    "pair_evaluation": pair_line,
                    "status_payload": status_line,
                    "promoted_after_attempt": promoted,
                    "stderr_tail": proc.stderr[-1200:],
                }
            )

            if proc.returncode != 0:
                break
            if promoted:
                break

        pair_reports.append(
            {
                "pair": f"{symbol}:{horizon}",
                "status": "promoted" if promoted else "not_promoted",
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
        )

    registry_after = ModelRegistry().read()
    promoted_after = _promoted_set(registry_after)
    output_path = Path(args.output).resolve() if args.output.strip() else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "active_before": active_before,
        "active_after": registry_after.get("active_model_version"),
        "pair_count": len(pairs),
        "total_attempts": total_attempts,
        "promoted_before_count": len(promoted_before),
        "promoted_after_count": len(promoted_after),
        "profiles": [profile.name for profile in profiles],
        "pair_reports": pair_reports,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
