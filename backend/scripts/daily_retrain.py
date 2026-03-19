#!/usr/bin/env python3
"""Run ingestion, quality gate, training, and promotion as one retrain pipeline."""

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
from app.services.experiment_tracker import log_experiment_event
from app.services.model_registry import ModelRegistry
from app.ml.training import all_horizon_specs


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    # Helper to keep all child script executions consistent and reportable.
    """Run command. Internal helper."""
    return subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)


def _today_version() -> str:
    """Internal helper to compute today version."""
    return datetime.now(tz=UTC).strftime("daily-%Y%m%d-%H%M%S")


def _parse_csv(raw: str) -> list[str]:
    """Parse comma-separated values."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _excluded_pairs_from_quality_report(
    *,
    quality_payload: dict[str, object],
    symbols: list[str],
    horizons: list[str],
) -> set[tuple[str, str]]:
    """Map failing quality checks into symbol+horizon exclusions."""
    pair_set: set[tuple[str, str]] = set()
    all_specs = {spec.label: spec for spec in all_horizon_specs()}
    selected_specs = [all_specs[horizon] for horizon in horizons if horizon in all_specs]
    if not selected_specs:
        return pair_set

    def _horizons_for_granularity(granularity: str) -> list[str]:
        hits: list[str] = []
        for spec in selected_specs:
            if any(candidate_granularity == granularity for candidate_granularity, _steps in spec.candidates):
                hits.append(spec.label)
        return hits

    failing_pairs = quality_payload.get("failing_pairs", [])
    if isinstance(failing_pairs, list):
        for item in failing_pairs:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper()
            granularity = str(item.get("granularity", ""))
            if not symbol or symbol not in symbols:
                continue
            for horizon in _horizons_for_granularity(granularity):
                pair_set.add((symbol, horizon))

    failing_cross_symbol = quality_payload.get("failing_cross_symbol_freshness", [])
    if isinstance(failing_cross_symbol, list):
        for item in failing_cross_symbol:
            if not isinstance(item, dict):
                continue
            granularity = str(item.get("granularity", ""))
            for symbol in symbols:
                for horizon in _horizons_for_granularity(granularity):
                    pair_set.add((symbol, horizon))
    return pair_set


def main() -> None:
    """Run the script entrypoint."""
    defaults = get_settings()
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
    parser.add_argument(
        "--horizons",
        default="",
        help="Optional comma-separated explicit horizon set for training",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Skip pre-training ingestion cycle")
    parser.add_argument("--output-prefix", default="reports", help="Output folder for reports")
    parser.add_argument(
        "--enforce-data-quality",
        action="store_true",
        help="Run data quality preflight and abort retrain when gate fails",
    )
    parser.add_argument(
        "--quality-granularities",
        default="1m,5m,15m,1h,6h,1d",
        help="Granularities for data quality check",
    )
    parser.add_argument(
        "--quality-max-gap-ratio",
        type=float,
        default=0.10,
        help="Maximum gap ratio allowed in quality preflight",
    )
    parser.add_argument(
        "--quality-min-rows-map",
        default="",
        help="Optional per-granularity row floors for quality preflight, format 1m:20000,1h:4000",
    )
    parser.add_argument(
        "--quality-min-coverage-ratio-map",
        default="",
        help="Optional per-granularity coverage floors, format 1m:0.8,1h:0.9",
    )
    parser.add_argument(
        "--quality-max-cross-symbol-lag-steps-map",
        default="",
        help="Optional cross-symbol freshness spread ceilings by granularity, format 1m:480,1h:48",
    )
    parser.add_argument(
        "--auto-remediate-data-quality",
        dest="auto_remediate_data_quality",
        action="store_true",
        help="Attempt backfill remediation and pair-level exclusion when quality preflight fails",
    )
    parser.add_argument(
        "--no-auto-remediate-data-quality",
        dest="auto_remediate_data_quality",
        action="store_false",
        help="Disable automatic remediation fallback when quality preflight fails",
    )
    parser.add_argument(
        "--remediation-max-passes",
        type=int,
        default=2,
        help="Max backfill passes during automatic quality remediation",
    )
    parser.add_argument(
        "--remediation-sleep-ms",
        type=int,
        default=25,
        help="Delay between backfill API calls during remediation",
    )
    parser.add_argument(
        "--enforce-sla",
        action="store_true",
        help="Run SLA gate (data quality + source uptime) and abort retrain on failure",
    )
    parser.add_argument(
        "--sla-min-live-source-ratio",
        type=float,
        default=float(defaults.sla_min_live_source_ratio),
        help="Minimum acceptable live-source ratio for SLA gate",
    )
    parser.add_argument(
        "--sla-max-stale-cache-ratio",
        type=float,
        default=float(defaults.sla_max_stale_cache_ratio),
        help="Maximum acceptable stale-cache ratio for SLA gate",
    )
    parser.add_argument(
        "--sla-source-hours",
        type=int,
        default=24,
        help="Source health lookback window for SLA gate",
    )
    parser.add_argument(
        "--enforce-ci-gate",
        action="store_true",
        help="Run CI gate checks before promotion and require pass",
    )
    parser.add_argument(
        "--ci-gate-max-age-hours",
        type=int,
        default=int(defaults.ci_gate_max_age_hours),
        help="Maximum age for reusable passing CI gate report",
    )
    parser.add_argument(
        "--skip-scorecard",
        action="store_true",
        help="Skip scorecard/model-card generation after promotion",
    )
    parser.set_defaults(auto_remediate_data_quality=True)
    args = parser.parse_args()

    settings = get_settings()
    effective_symbols = args.symbols.strip()
    effective_horizons = args.horizons.strip()
    if args.phase == "phase1":
        if not effective_symbols:
            effective_symbols = settings.phase1_focus_symbols
        if not effective_horizons:
            effective_horizons = settings.phase1_focus_horizons
    output_root = PROJECT_ROOT / args.output_prefix
    output_root.mkdir(parents=True, exist_ok=True)

    run_report: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": args.model_version,
        "phase": args.phase,
        "steps": [],
    }
    quality_report_path = f"{args.output_prefix}/data_quality_{args.model_version}.json"
    sla_report_path = f"{args.output_prefix}/sla_gate_{args.model_version}.json"
    ci_gate_report_path = f"{args.output_prefix}/ci_gate_{args.model_version}.json"
    scorecard_output_path = f"{args.output_prefix}/scorecard_{args.model_version}.json"
    model_card_output_path = f"../docs/model-cards/{args.model_version}.md"

    if not args.skip_ingest:
        try:
            asyncio.run(run_ingestion_cycle())
            run_report["steps"].append({"step": "ingestion", "status": "ok"})
        except Exception as exc:
            run_report["steps"].append({"step": "ingestion", "status": "failed", "error": str(exc)})

    excluded_symbol_horizons: set[tuple[str, str]] = set()
    if args.enforce_data_quality:
        # Gate retraining when source data quality does not satisfy minimum standards.
        symbols_for_quality = effective_symbols if effective_symbols else settings.supported_symbols
        selected_symbols = [item.strip().upper() for item in symbols_for_quality.split(",") if item.strip()]
        selected_horizons = [
            item.strip().lower()
            for item in (effective_horizons if effective_horizons else settings.supported_horizons).split(",")
            if item.strip()
        ]

        def _quality_cmd() -> list[str]:
            cmd = [
                sys.executable,
                "scripts/data_quality_report.py",
                "--granularities",
                args.quality_granularities,
                "--max-gap-ratio",
                str(args.quality_max_gap_ratio),
                "--output",
                quality_report_path,
                "--symbols",
                symbols_for_quality,
            ]
            if args.quality_min_rows_map.strip():
                cmd.extend(["--min-rows-map", args.quality_min_rows_map.strip()])
            if args.quality_min_coverage_ratio_map.strip():
                cmd.extend(["--min-coverage-ratio-map", args.quality_min_coverage_ratio_map.strip()])
            if args.quality_max_cross_symbol_lag_steps_map.strip():
                cmd.extend(["--max-cross-symbol-lag-steps-map", args.quality_max_cross_symbol_lag_steps_map.strip()])
            return cmd

        quality_proc = _run_command(_quality_cmd())
        quality_payload: dict[str, object] = {}
        quality_gate_passed = False
        quality_file = PROJECT_ROOT / quality_report_path
        if quality_proc.returncode == 0 and quality_file.exists():
            try:
                quality_payload = json.loads(quality_file.read_text(encoding="utf-8"))
            except Exception:
                quality_payload = {}
            quality_gate_passed = bool(quality_payload.get("gate_passed"))

        run_report["steps"].append(
            {
                "step": "data_quality_preflight",
                "status": "ok" if (quality_proc.returncode == 0 and quality_gate_passed) else "failed",
                "returncode": quality_proc.returncode,
                "gate_passed": quality_gate_passed,
                "failing_pair_count": quality_payload.get("failing_pair_count"),
                "failing_cross_symbol_freshness_count": quality_payload.get("failing_cross_symbol_freshness_count"),
                "stdout_tail": quality_proc.stdout[-4000:],
                "stderr_tail": quality_proc.stderr[-4000:],
                "quality_report": str(PROJECT_ROOT / quality_report_path),
            }
        )

        if (quality_proc.returncode != 0 or not quality_gate_passed) and args.auto_remediate_data_quality:
            remediation_symbols: set[str] = set()
            remediation_granularities: set[str] = set()
            failing_pairs = quality_payload.get("failing_pairs", [])
            if isinstance(failing_pairs, list):
                for item in failing_pairs:
                    if not isinstance(item, dict):
                        continue
                    symbol = str(item.get("symbol", "")).upper()
                    granularity = str(item.get("granularity", ""))
                    if symbol:
                        remediation_symbols.add(symbol)
                    if granularity:
                        remediation_granularities.add(granularity)
            failing_cross = quality_payload.get("failing_cross_symbol_freshness", [])
            if isinstance(failing_cross, list):
                for item in failing_cross:
                    if not isinstance(item, dict):
                        continue
                    granularity = str(item.get("granularity", ""))
                    if granularity:
                        remediation_granularities.add(granularity)
                remediation_symbols.update(selected_symbols)

            if remediation_symbols and remediation_granularities:
                remediation_output = f"{args.output_prefix}/backfill_remediation_{args.model_version}.json"
                remediation_cmd = [
                    sys.executable,
                    "scripts/backfill_market_data.py",
                    "--symbols",
                    ",".join(sorted(remediation_symbols)),
                    "--granularities",
                    ",".join(sorted(remediation_granularities)),
                    "--max-passes",
                    str(max(int(args.remediation_max_passes), 1)),
                    "--sleep-ms",
                    str(max(int(args.remediation_sleep_ms), 0)),
                    "--output",
                    remediation_output,
                ]
                remediation_proc = _run_command(remediation_cmd)
                run_report["steps"].append(
                    {
                        "step": "data_quality_remediation_backfill",
                        "status": "ok" if remediation_proc.returncode == 0 else "failed",
                        "returncode": remediation_proc.returncode,
                        "symbols": sorted(remediation_symbols),
                        "granularities": sorted(remediation_granularities),
                        "stdout_tail": remediation_proc.stdout[-4000:],
                        "stderr_tail": remediation_proc.stderr[-4000:],
                        "report": str(PROJECT_ROOT / remediation_output),
                    }
                )

                quality_proc = _run_command(_quality_cmd())
                quality_payload = {}
                quality_gate_passed = False
                if quality_proc.returncode == 0 and quality_file.exists():
                    try:
                        quality_payload = json.loads(quality_file.read_text(encoding="utf-8"))
                    except Exception:
                        quality_payload = {}
                    quality_gate_passed = bool(quality_payload.get("gate_passed"))
                run_report["steps"].append(
                    {
                        "step": "data_quality_post_remediation",
                        "status": "ok" if (quality_proc.returncode == 0 and quality_gate_passed) else "failed",
                        "returncode": quality_proc.returncode,
                        "gate_passed": quality_gate_passed,
                        "failing_pair_count": quality_payload.get("failing_pair_count"),
                        "failing_cross_symbol_freshness_count": quality_payload.get("failing_cross_symbol_freshness_count"),
                        "stdout_tail": quality_proc.stdout[-4000:],
                        "stderr_tail": quality_proc.stderr[-4000:],
                        "quality_report": str(PROJECT_ROOT / quality_report_path),
                    }
                )

        if quality_proc.returncode != 0 or not quality_gate_passed:
            excluded_symbol_horizons = _excluded_pairs_from_quality_report(
                quality_payload=quality_payload,
                symbols=selected_symbols,
                horizons=selected_horizons,
            )
            run_report["steps"].append(
                {
                    "step": "data_quality_exclusions",
                    "status": "ok" if excluded_symbol_horizons else "failed",
                    "excluded_pairs": sorted([f"{symbol}:{horizon}" for symbol, horizon in excluded_symbol_horizons]),
                }
            )
            if not excluded_symbol_horizons:
                output_file = output_root / f"daily_retrain_{args.model_version}.json"
                output_file.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "reason": "data_quality_gate_failed",
                            "report": str(output_file),
                            "quality_report": str(PROJECT_ROOT / quality_report_path),
                        },
                        indent=2,
                    )
                )
                raise SystemExit(1)

    if args.enforce_sla:
        sla_cmd = [
            sys.executable,
            "scripts/sla_gate.py",
            "--source-hours",
            str(max(int(args.sla_source_hours), 1)),
            "--min-live-source-ratio",
            str(float(args.sla_min_live_source_ratio)),
            "--max-stale-cache-ratio",
            str(float(args.sla_max_stale_cache_ratio)),
            "--output",
            sla_report_path,
            "--require-quality-pass",
        ]
        quality_file = PROJECT_ROOT / quality_report_path
        if quality_file.exists():
            sla_cmd.extend(["--quality-report", str(quality_file)])
        sla_proc = _run_command(sla_cmd)
        run_report["steps"].append(
            {
                "step": "sla_gate",
                "status": "ok" if sla_proc.returncode == 0 else "failed",
                "returncode": sla_proc.returncode,
                "stdout_tail": sla_proc.stdout[-4000:],
                "stderr_tail": sla_proc.stderr[-4000:],
                "sla_report": str(PROJECT_ROOT / sla_report_path),
            }
        )
        if sla_proc.returncode != 0:
            output_file = output_root / f"daily_retrain_{args.model_version}.json"
            output_file.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "reason": "sla_gate_failed",
                        "report": str(output_file),
                        "sla_report": str(PROJECT_ROOT / sla_report_path),
                    },
                    indent=2,
                )
            )
            raise SystemExit(1)

    train_output = f"{args.output_prefix}/summary_report_{args.model_version}.json"
    # Train a candidate version first; promotion is handled separately below.
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
    if effective_symbols:
        train_cmd.extend(["--symbols", effective_symbols])
    if effective_horizons:
        train_cmd.extend(["--horizons", effective_horizons])
    if excluded_symbol_horizons:
        excluded_csv = ",".join(
            sorted([f"{symbol}:{horizon}" for symbol, horizon in excluded_symbol_horizons])
        )
        train_cmd.extend(["--exclude-symbol-horizons", excluded_csv])
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

    if args.enforce_ci_gate:
        ci_cmd = [
            sys.executable,
            "scripts/ci_gate.py",
            "--output",
            ci_gate_report_path,
            "--reuse-if-fresh",
            "--max-age-hours",
            str(max(int(args.ci_gate_max_age_hours), 1)),
            "--repro-phase",
            "phase1",
            "--repro-symbols",
            (effective_symbols if effective_symbols else settings.phase1_focus_symbols),
            "--repro-horizons",
            (effective_horizons if effective_horizons else settings.phase1_focus_horizons),
        ]
        ci_proc = _run_command(ci_cmd)
        run_report["steps"].append(
            {
                "step": "ci_gate",
                "status": "ok" if ci_proc.returncode == 0 else "failed",
                "returncode": ci_proc.returncode,
                "stdout_tail": ci_proc.stdout[-4000:],
                "stderr_tail": ci_proc.stderr[-4000:],
                "ci_gate_report": str(PROJECT_ROOT / ci_gate_report_path),
            }
        )
        if ci_proc.returncode != 0:
            output_file = output_root / f"daily_retrain_{args.model_version}.json"
            output_file.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "reason": "ci_gate_failed",
                        "report": str(output_file),
                        "ci_gate_report": str(PROJECT_ROOT / ci_gate_report_path),
                    },
                    indent=2,
                )
            )
            raise SystemExit(1)

    promote_output = f"{args.output_prefix}/promotion_report_{args.model_version}.json"
    # Promotion applies phase rules and only activates pairs that pass strict gates.
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
    if args.enforce_ci_gate:
        promote_cmd.extend(
            [
                "--require-ci-gate",
                "--ci-gate-report",
                ci_gate_report_path,
                "--ci-gate-max-age-hours",
                str(max(int(args.ci_gate_max_age_hours), 1)),
            ]
        )

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

    if not args.skip_scorecard:
        scorecard_cmd = [
            sys.executable,
            "scripts/daily_scorecard.py",
            "--model-version",
            args.model_version,
            "--summary-report",
            train_output,
            "--promotion-report",
            promote_output,
            "--symbols",
            (effective_symbols if effective_symbols else settings.phase1_focus_symbols),
            "--horizons",
            (effective_horizons if effective_horizons else settings.phase1_focus_horizons),
            "--source-hours",
            str(max(int(args.sla_source_hours), 1)),
            "--output",
            scorecard_output_path,
        ]
        scorecard_proc = _run_command(scorecard_cmd)
        run_report["steps"].append(
            {
                "step": "daily_scorecard",
                "status": "ok" if scorecard_proc.returncode == 0 else "failed",
                "returncode": scorecard_proc.returncode,
                "stdout_tail": scorecard_proc.stdout[-4000:],
                "stderr_tail": scorecard_proc.stderr[-4000:],
                "scorecard_report": str(PROJECT_ROOT / scorecard_output_path),
            }
        )

        model_card_cmd = [
            sys.executable,
            "scripts/generate_model_card.py",
            "--model-version",
            args.model_version,
            "--summary-report",
            train_output,
            "--promotion-report",
            promote_output,
            "--scorecard-report",
            scorecard_output_path,
            "--output",
            model_card_output_path,
        ]
        model_card_proc = _run_command(model_card_cmd)
        run_report["steps"].append(
            {
                "step": "generate_model_card",
                "status": "ok" if model_card_proc.returncode == 0 else "failed",
                "returncode": model_card_proc.returncode,
                "stdout_tail": model_card_proc.stdout[-4000:],
                "stderr_tail": model_card_proc.stderr[-4000:],
                "model_card": str(PROJECT_ROOT / model_card_output_path),
            }
        )

    output_file = output_root / f"daily_retrain_{args.model_version}.json"
    output_file.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    event_path = log_experiment_event(
        "daily_retrain",
        {
            "model_version": args.model_version,
            "phase": args.phase,
            "status": "ok" if promote_proc.returncode == 0 else "failed",
            "report": str(output_file),
            "summary_report": str(PROJECT_ROOT / train_output),
            "promotion_report": str(PROJECT_ROOT / promote_output),
            "quality_report": str(PROJECT_ROOT / quality_report_path) if args.enforce_data_quality else None,
            "sla_report": str(PROJECT_ROOT / sla_report_path) if args.enforce_sla else None,
            "ci_gate_report": str(PROJECT_ROOT / ci_gate_report_path) if args.enforce_ci_gate else None,
            "scorecard_report": str(PROJECT_ROOT / scorecard_output_path) if not args.skip_scorecard else None,
            "model_card": str(PROJECT_ROOT / model_card_output_path) if not args.skip_scorecard else None,
            "excluded_symbol_horizons": sorted([f"{symbol}:{horizon}" for symbol, horizon in excluded_symbol_horizons]),
        },
    )

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
                "quality_report": str(PROJECT_ROOT / quality_report_path) if args.enforce_data_quality else None,
                "sla_report": str(PROJECT_ROOT / sla_report_path) if args.enforce_sla else None,
                "ci_gate_report": str(PROJECT_ROOT / ci_gate_report_path) if args.enforce_ci_gate else None,
                "scorecard_report": str(PROJECT_ROOT / scorecard_output_path) if not args.skip_scorecard else None,
                "model_card": str(PROJECT_ROOT / model_card_output_path) if not args.skip_scorecard else None,
                "experiment_events_path": str(event_path),
                "ingestion_enabled": settings.ingestion_enabled,
                "effective_symbols": effective_symbols if effective_symbols else None,
                "effective_horizons": effective_horizons if effective_horizons else None,
                "excluded_symbol_horizons": sorted([f"{symbol}:{horizon}" for symbol, horizon in excluded_symbol_horizons]),
            },
            indent=2,
        )
    )

    if promote_proc.returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
