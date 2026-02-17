#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.ingestion import run_ingestion_cycle
from app.services.model_registry import ModelRegistry


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)


def _today_version() -> str:
    return datetime.now(tz=UTC).strftime("daily-%Y%m%d-%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily ingest -> train -> promote pipeline")
    parser.add_argument("--model-version", default=_today_version(), help="Candidate model version")
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3"], default="phase3", help="Promotion phase")
    parser.add_argument("--phase2-batch-size", type=int, default=20, help="Batch size when phase2 is selected")
    parser.add_argument("--symbol-limit", type=int, default=2000, help="Universe fetch cap for training")
    parser.add_argument("--batch-size", type=int, default=20, help="Training progress batch size")
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated explicit symbol set for training",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Skip pre-training ingestion cycle")
    parser.add_argument("--output-prefix", default="reports", help="Output folder for reports")
    args = parser.parse_args()

    settings = get_settings()
    output_root = PROJECT_ROOT / args.output_prefix
    output_root.mkdir(parents=True, exist_ok=True)

    run_report: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": args.model_version,
        "phase": args.phase,
        "steps": [],
    }

    if not args.skip_ingest:
        try:
            asyncio.run(run_ingestion_cycle())
            run_report["steps"].append({"step": "ingestion", "status": "ok"})
        except Exception as exc:
            run_report["steps"].append({"step": "ingestion", "status": "failed", "error": str(exc)})

    train_output = f"{args.output_prefix}/summary_report_{args.model_version}.json"
    train_cmd = [
        sys.executable,
        "scripts/train_all_symbols.py",
        "--model-version",
        args.model_version,
        "--output",
        train_output,
        "--symbol-limit",
        str(args.symbol_limit),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.symbols.strip():
        train_cmd.extend(["--symbols", args.symbols.strip()])
    train_proc = _run_command(train_cmd)
    run_report["steps"].append(
        {
            "step": "train_all_symbols",
            "status": "ok" if train_proc.returncode == 0 else "failed",
            "returncode": train_proc.returncode,
            "stdout_tail": train_proc.stdout[-4000:],
            "stderr_tail": train_proc.stderr[-4000:],
        }
    )

    if train_proc.returncode != 0:
        output_file = output_root / f"daily_retrain_{args.model_version}.json"
        output_file.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
        print(json.dumps({"status": "failed", "report": str(output_file)}, indent=2))
        raise SystemExit(1)

    promote_output = f"{args.output_prefix}/promotion_report_{args.model_version}.json"
    promote_cmd = [
        sys.executable,
        "scripts/promote_model.py",
        "--candidate",
        args.model_version,
        "--phase",
        args.phase,
        "--output",
        promote_output,
    ]

    if args.phase == "phase2":
        promote_cmd.extend(["--phase2-batch-size", str(args.phase2_batch_size)])

    registry = ModelRegistry()
    active_before = registry.get_active_model_version()
    if active_before:
        promote_cmd.extend(["--active", active_before])

    promote_proc = _run_command(promote_cmd)
    run_report["steps"].append(
        {
            "step": "promote_model",
            "status": "ok" if promote_proc.returncode == 0 else "failed",
            "returncode": promote_proc.returncode,
            "stdout_tail": promote_proc.stdout[-4000:],
            "stderr_tail": promote_proc.stderr[-4000:],
        }
    )

    output_file = output_root / f"daily_retrain_{args.model_version}.json"
    output_file.write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok" if promote_proc.returncode == 0 else "failed",
                "model_version": args.model_version,
                "active_before": active_before,
                "active_after": registry.get_active_model_version(),
                "run_report": str(output_file),
                "summary_report": str(PROJECT_ROOT / train_output),
                "promotion_report": str(PROJECT_ROOT / promote_output),
                "ingestion_enabled": settings.ingestion_enabled,
            },
            indent=2,
        )
    )

    if promote_proc.returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
