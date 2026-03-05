#!/usr/bin/env python3
"""One-command reproducible pipeline from ingestion to promotion reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.services.experiment_tracker import log_experiment_event


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=BACKEND_ROOT, text=True, capture_output=True, check=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _default_model_version() -> str:
    return datetime.now(tz=UTC).strftime("repro-%Y%m%d-%H%M%S")


def _as_report_file(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible ingest/train/promote/guard pipeline")
    parser.add_argument("--model-version", default=_default_model_version(), help="Candidate model version")
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3"], default="phase3")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol subset")
    parser.add_argument("--symbol-limit", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--phase2-batch-size", type=int, default=20)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument(
        "--enforce-data-quality",
        dest="enforce_data_quality",
        action="store_true",
        default=True,
        help="Enforce data quality gate before retrain (default: enabled)",
    )
    parser.add_argument(
        "--skip-data-quality-gate",
        dest="enforce_data_quality",
        action="store_false",
        help="Skip data quality gate in reproducibility run",
    )
    parser.add_argument("--quality-granularities", default="1m,5m,15m,1h,6h,1d")
    parser.add_argument("--quality-max-gap-ratio", type=float, default=0.10)
    parser.add_argument("--quality-min-rows-map", default="")
    parser.add_argument("--rollback-guard-hours", type=int, default=24)
    parser.add_argument("--rollback-guard-max-stale", type=float, default=0.35)
    parser.add_argument("--output", default="reports/repro_bundle.json", help="Bundle report output path")
    args = parser.parse_args()

    settings = get_settings()
    generated_at = datetime.now(tz=UTC).isoformat()

    retrain_cmd = [
        sys.executable,
        "scripts/daily_retrain.py",
        "--model-version",
        args.model_version,
        "--phase",
        args.phase,
        "--phase2-batch-size",
        str(args.phase2_batch_size),
        "--symbol-limit",
        str(args.symbol_limit),
        "--batch-size",
        str(args.batch_size),
        "--output-prefix",
        "reports",
        "--quality-granularities",
        args.quality_granularities,
        "--quality-max-gap-ratio",
        str(args.quality_max_gap_ratio),
    ]
    if args.enforce_data_quality:
        retrain_cmd.append("--enforce-data-quality")
    if args.quality_min_rows_map.strip():
        retrain_cmd.extend(["--quality-min-rows-map", args.quality_min_rows_map.strip()])
    if args.skip_ingest:
        retrain_cmd.append("--skip-ingest")
    if args.symbols.strip():
        retrain_cmd.extend(["--symbols", args.symbols.strip()])

    retrain_proc = _run_command(retrain_cmd)

    retrain_json: dict[str, object] = {}
    try:
        retrain_json = json.loads(retrain_proc.stdout)
    except Exception:
        retrain_json = {}

    rollback_cmd = [
        sys.executable,
        "scripts/auto_rollback_guard.py",
        "--hours",
        str(args.rollback_guard_hours),
        "--max-stale-cache-ratio",
        str(args.rollback_guard_max_stale),
        "--output",
        f"reports/auto_rollback_guard_{args.model_version}.json",
    ]
    rollback_proc = _run_command(rollback_cmd)
    rollback_json: dict[str, object] = {}
    try:
        rollback_json = json.loads(rollback_proc.stdout)
    except Exception:
        rollback_json = {}

    report_paths = {
        "daily_retrain_report": _as_report_file(retrain_json.get("run_report")),
        "summary_report": _as_report_file(retrain_json.get("summary_report")),
        "promotion_report": _as_report_file(retrain_json.get("promotion_report")),
        "quality_report": _as_report_file(retrain_json.get("quality_report")),
        "rollback_guard_report": _as_report_file(rollback_json.get("report")),
    }
    checksums: dict[str, dict[str, object]] = {}
    for name, path in report_paths.items():
        if path is None or not path.exists() or not path.is_file():
            continue
        checksums[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }

    bundle = {
        "generated_at": generated_at,
        "pipeline": "repro_pipeline",
        "model_version": args.model_version,
        "git_commit": _git_commit(),
        "settings_snapshot": {
            "ingestion_enabled": bool(settings.ingestion_enabled),
            "ingestion_symbol_limit": int(settings.ingestion_symbol_limit),
            "ingestion_limit_per_symbol": int(settings.ingestion_limit_per_symbol),
            "classification_label_mode": settings.classification_label_mode,
            "triple_barrier_sigma_mult": float(settings.triple_barrier_sigma_mult),
            "walk_forward_gate_mode": settings.walk_forward_gate_mode,
            "walk_forward_gate_folds": int(settings.walk_forward_gate_folds),
            "execution_fee_bps": float(settings.execution_fee_bps),
            "execution_slippage_bps": float(settings.execution_slippage_bps),
            "execution_max_turnover_per_step": float(settings.execution_max_turnover_per_step),
            "metric_ci_bootstrap_samples": int(settings.metric_ci_bootstrap_samples),
            "metric_ci_level": float(settings.metric_ci_level),
        },
        "commands": {
            "daily_retrain": retrain_cmd,
            "rollback_guard": rollback_cmd,
        },
        "steps": [
            {
                "name": "daily_retrain",
                "returncode": retrain_proc.returncode,
                "status": "ok" if retrain_proc.returncode == 0 else "failed",
                "stdout_tail": retrain_proc.stdout[-4000:],
                "stderr_tail": retrain_proc.stderr[-4000:],
                "parsed": retrain_json,
            },
            {
                "name": "rollback_guard",
                "returncode": rollback_proc.returncode,
                "status": "ok" if rollback_proc.returncode == 0 else "failed",
                "stdout_tail": rollback_proc.stdout[-4000:],
                "stderr_tail": rollback_proc.stderr[-4000:],
                "parsed": rollback_json,
            },
        ],
        "report_checksums": checksums,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = BACKEND_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    event_path = log_experiment_event(
        "repro_pipeline",
        {
            "model_version": args.model_version,
            "status": "ok" if retrain_proc.returncode == 0 else "failed",
            "bundle_report": str(output_path),
            "daily_retrain_status": bundle["steps"][0]["status"],
            "rollback_guard_status": bundle["steps"][1]["status"],
        },
    )

    print(
        json.dumps(
            {
                "status": "ok" if retrain_proc.returncode == 0 else "failed",
                "model_version": args.model_version,
                "bundle_report": str(output_path),
                "experiment_events_path": str(event_path),
            },
            indent=2,
        )
    )

    if retrain_proc.returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
