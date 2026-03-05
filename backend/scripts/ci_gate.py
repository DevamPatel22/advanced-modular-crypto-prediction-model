#!/usr/bin/env python3
"""Run CI-style gates and emit a machine-readable readiness report."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Internal helper to compute run."""
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False, env=merged_env)


def _is_fresh_pass(path: Path, max_age_hours: int) -> tuple[bool, dict[str, object] | None]:
    """Internal helper to compute is fresh pass."""
    if not path.exists():
        return False, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    if not isinstance(payload, dict):
        return False, None
    if str(payload.get("status", "")).lower() != "ok":
        return False, payload
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str):
        return False, payload
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except Exception:
        return False, payload
    min_allowed = datetime.now(tz=UTC) - timedelta(hours=max(max_age_hours, 1))
    return ts >= min_allowed, payload


def main() -> None:
    """Run the script entrypoint."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run CI gate checks and produce JSON status report")
    parser.add_argument("--output", default="reports/ci_gate_latest.json", help="Output JSON path")
    parser.add_argument("--max-age-hours", type=int, default=int(settings.ci_gate_max_age_hours))
    parser.add_argument("--reuse-if-fresh", action="store_true", help="Reuse existing passing report if fresh")
    parser.add_argument("--require-ruff", action="store_true", help="Fail gate when ruff is unavailable")
    parser.add_argument("--skip-repro-smoke", action="store_true", help="Skip reproducibility smoke run")
    parser.add_argument("--repro-phase", choices=["phase1", "phase2", "phase3"], default="phase1")
    parser.add_argument("--repro-symbols", default=settings.phase1_focus_symbols)
    parser.add_argument("--repro-horizons", default=settings.phase1_focus_horizons)
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_if_fresh:
        fresh, payload = _is_fresh_pass(output_path, int(args.max_age_hours))
        if fresh and payload is not None:
            payload = dict(payload)
            payload["reused"] = True
            print(json.dumps(payload, indent=2))
            return

    steps: list[dict[str, object]] = []

    compile_proc = _run([sys.executable, "-m", "compileall", "-q", "app", "scripts"])
    steps.append(
        {
            "name": "compile",
            "returncode": compile_proc.returncode,
            "status": "ok" if compile_proc.returncode == 0 else "failed",
            "stdout_tail": compile_proc.stdout[-4000:],
            "stderr_tail": compile_proc.stderr[-4000:],
        }
    )

    test_proc = _run([sys.executable, "-m", "pytest", "-q"], env={"PYTHONPATH": "."})
    steps.append(
        {
            "name": "pytest",
            "returncode": test_proc.returncode,
            "status": "ok" if test_proc.returncode == 0 else "failed",
            "stdout_tail": test_proc.stdout[-4000:],
            "stderr_tail": test_proc.stderr[-4000:],
        }
    )

    has_ruff = importlib.util.find_spec("ruff") is not None
    if has_ruff:
        lint_proc = _run([sys.executable, "-m", "ruff", "check", "app", "scripts", "tests"])
        lint_step = {
            "name": "ruff_lint",
            "returncode": lint_proc.returncode,
            "status": "ok" if lint_proc.returncode == 0 else "failed",
            "stdout_tail": lint_proc.stdout[-4000:],
            "stderr_tail": lint_proc.stderr[-4000:],
        }
    else:
        lint_step = {
            "name": "ruff_lint",
            "returncode": None,
            "status": "failed" if args.require_ruff else "skipped",
            "stdout_tail": "",
            "stderr_tail": "ruff_not_installed",
        }
    steps.append(lint_step)

    repro_step: dict[str, object] = {
        "name": "repro_smoke",
        "returncode": None,
        "status": "skipped" if args.skip_repro_smoke else "pending",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if not args.skip_repro_smoke:
        smoke_version = datetime.now(tz=UTC).strftime("ci-smoke-%Y%m%d-%H%M%S")
        repro_cmd = [
            sys.executable,
            "scripts/repro_pipeline.py",
            "--model-version",
            smoke_version,
            "--phase",
            args.repro_phase,
            "--symbols",
            args.repro_symbols,
            "--horizons",
            args.repro_horizons,
            "--skip-ingest",
            "--skip-data-quality-gate",
            "--output",
            f"reports/repro_bundle_{smoke_version}.json",
        ]
        repro_proc = _run(repro_cmd)
        repro_step = {
            "name": "repro_smoke",
            "command": repro_cmd,
            "returncode": repro_proc.returncode,
            "status": "ok" if repro_proc.returncode == 0 else "failed",
            "stdout_tail": repro_proc.stdout[-4000:],
            "stderr_tail": repro_proc.stderr[-4000:],
        }
    steps.append(repro_step)

    failing = [step for step in steps if str(step.get("status")) == "failed"]
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": "ok" if not failing else "failed",
        "checks_passed": len(failing) == 0,
        "checks": steps,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
