#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.ml.features import HORIZON_SPECS
from app.ml.training import evaluate_symbol_horizon
from app.services.ingestion import run_ingestion_cycle
from app.services.model_registry import ModelRegistry


@dataclass(frozen=True)
class PairCandidate:
    symbol: str
    horizon: str
    failed_count: int
    deficit_score: float
    failed_reasons: list[str]
    source_report: str


def _candidate_payload(candidate: PairCandidate) -> dict[str, object]:
    deficit = float(candidate.deficit_score)
    if not math.isfinite(deficit):
        deficit_payload: float | None = None
    else:
        deficit_payload = deficit
    return {
        "symbol": candidate.symbol,
        "horizon": candidate.horizon,
        "failed_count": int(candidate.failed_count),
        "deficit_score": deficit_payload,
        "failed_reasons": list(candidate.failed_reasons),
        "source_report": candidate.source_report,
    }


def _default_model_version() -> str:
    return datetime.now(tz=UTC).strftime("nearloop-%Y%m%d-%H%M%S")


def _parse_symbols(raw: str) -> list[str]:
    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return sorted(set(values))


def _parse_target_pairs(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        if ":" not in text:
            raise SystemExit(f"Invalid target pair '{item}'. Expected SYMBOL:HORIZON format.")
        symbol_raw, horizon_raw = [part.strip() for part in text.split(":", 1)]
        symbol = symbol_raw.upper()
        horizon = horizon_raw.lower()
        pair = (symbol, horizon)
        if not symbol or not horizon or pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def _promoted_pair_set(payload: dict[str, object]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    promoted = payload.get("promoted", {})
    if not isinstance(promoted, dict):
        return out
    for symbol, horizons in promoted.items():
        if not isinstance(horizons, dict):
            continue
        for horizon, passed in horizons.items():
            if bool(passed):
                out.add((str(symbol).upper(), str(horizon)))
    return out


def _ranked_summary_reports(reports_root: Path) -> list[Path]:
    paths = list(reports_root.glob("summary_report_*.json"))
    if not paths:
        return []

    def _score(path: Path) -> tuple[int, int, float, int, int]:
        mtime = float(path.stat().st_mtime)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # Keep unreadable reports at the bottom while still allowing fallback.
            return (0, 0, 0, 0, mtime)

        if not isinstance(payload, dict):
            return (0, 0, 0, 0, mtime)

        results = payload.get("results", [])
        near_candidates = payload.get("near_pass_candidates", [])
        symbols = payload.get("symbols", {})
        results_count = len(results) if isinstance(results, list) else 0
        near_count = len(near_candidates) if isinstance(near_candidates, list) else 0
        symbols_count = len(symbols) if isinstance(symbols, dict) else 0
        coverage = max(results_count, near_count, symbols_count)

        model_version = str(payload.get("model_version", ""))
        is_non_nearloop = 0 if model_version.startswith("nearloop-") else 1
        is_usable_coverage = 1 if coverage >= 6 else 0
        return (is_usable_coverage, is_non_nearloop, mtime, coverage, near_count)

    return sorted(paths, key=_score, reverse=True)


def _to_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _deficit_score(delta_f1: float, delta_accuracy: float, delta_rmse: float) -> float:
    return abs(min(delta_f1, 0.0)) + abs(min(delta_accuracy, 0.0)) + abs(min(delta_rmse, 0.0))


def _extract_candidates_from_results(payload: dict[str, object], report_name: str) -> list[PairCandidate]:
    out: list[PairCandidate] = []
    results = payload.get("results", [])
    if not isinstance(results, list):
        return out

    for item in results:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper()
        horizon = str(item.get("horizon", ""))
        gate = item.get("promotion_gate", {})
        if not isinstance(gate, dict):
            continue
        if bool(gate.get("passed")):
            continue
        failed_reasons = [str(reason) for reason in gate.get("failed_reasons", []) if str(reason)]
        if not symbol or not horizon:
            continue
        delta = item.get("near_pass_delta", {})
        if not isinstance(delta, dict):
            delta = {}
        d_f1 = _to_float(delta.get("f1_vs_baseline"), fallback=0.0)
        d_acc = _to_float(delta.get("accuracy_vs_baseline"), fallback=0.0)
        d_rmse = _to_float(delta.get("rmse_vs_baseline"), fallback=0.0)
        out.append(
            PairCandidate(
                symbol=symbol,
                horizon=horizon,
                failed_count=max(len(failed_reasons), 1),
                deficit_score=_deficit_score(d_f1, d_acc, d_rmse),
                failed_reasons=failed_reasons,
                source_report=report_name,
            )
        )
    return out


def _extract_candidates_from_near_pass(payload: dict[str, object], report_name: str) -> list[PairCandidate]:
    out: list[PairCandidate] = []
    items = payload.get("near_pass_candidates", [])
    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper()
        horizon = str(item.get("horizon", ""))
        failed_reasons = [str(reason) for reason in item.get("failed_reasons", []) if str(reason)]
        if not symbol or not horizon:
            continue
        d_f1 = _to_float(item.get("delta_f1"), fallback=0.0)
        d_acc = _to_float(item.get("delta_accuracy"), fallback=0.0)
        d_rmse = _to_float(item.get("delta_rmse_margin"), fallback=0.0)
        out.append(
            PairCandidate(
                symbol=symbol,
                horizon=horizon,
                failed_count=max(len(failed_reasons), 1),
                deficit_score=_deficit_score(d_f1, d_acc, d_rmse),
                failed_reasons=failed_reasons,
                source_report=report_name,
            )
        )
    return out


def _extract_candidates_from_symbol_tree(payload: dict[str, object], report_name: str) -> list[PairCandidate]:
    out: list[PairCandidate] = []
    symbols = payload.get("symbols", {})
    if not isinstance(symbols, dict):
        return out

    for symbol_key, symbol_payload in symbols.items():
        symbol_name = str(symbol_key).upper()
        if not isinstance(symbol_payload, dict):
            continue
        horizons = symbol_payload.get("horizons", {})
        if not isinstance(horizons, dict):
            continue
        for horizon_key, entry in horizons.items():
            if not isinstance(entry, dict):
                continue
            gate = entry.get("promotion_gate", {})
            if not isinstance(gate, dict):
                continue
            if bool(gate.get("passed")):
                continue
            failed_reasons = [str(reason) for reason in gate.get("failed_reasons", []) if str(reason)]
            delta = entry.get("near_pass_delta", {})
            if not isinstance(delta, dict):
                delta = {}
            d_f1 = _to_float(delta.get("f1_vs_baseline"), fallback=0.0)
            d_acc = _to_float(delta.get("accuracy_vs_baseline"), fallback=0.0)
            d_rmse = _to_float(delta.get("rmse_vs_baseline"), fallback=0.0)
            out.append(
                PairCandidate(
                    symbol=symbol_name,
                    horizon=str(horizon_key),
                    failed_count=max(len(failed_reasons), 1),
                    deficit_score=_deficit_score(d_f1, d_acc, d_rmse),
                    failed_reasons=failed_reasons,
                    source_report=report_name,
                )
            )
    return out


def _rank_candidates(
    payload: dict[str, object],
    report_name: str,
    allowed_symbols: set[str],
    promoted_pairs: set[tuple[str, str]],
) -> list[PairCandidate]:
    candidates = (
        _extract_candidates_from_results(payload, report_name)
        + _extract_candidates_from_near_pass(payload, report_name)
        + _extract_candidates_from_symbol_tree(payload, report_name)
    )
    dedup: dict[tuple[str, str], PairCandidate] = {}

    for candidate in candidates:
        key = (candidate.symbol, candidate.horizon)
        if candidate.symbol not in allowed_symbols:
            continue
        if key in promoted_pairs:
            continue
        previous = dedup.get(key)
        if previous is None or (candidate.failed_count, candidate.deficit_score) < (previous.failed_count, previous.deficit_score):
            dedup[key] = candidate

    ranked = sorted(dedup.values(), key=lambda item: (item.failed_count, item.deficit_score, item.symbol, item.horizon))
    return ranked


def _fallback_pairs(
    symbols: list[str],
    promoted_pairs: set[tuple[str, str]],
    existing: set[tuple[str, str]],
) -> list[PairCandidate]:
    horizon_priority = ["12h", "6h", "3h", "1h", "5m"]
    out: list[PairCandidate] = []
    for symbol in symbols:
        for horizon in horizon_priority:
            key = (symbol, horizon)
            if key in promoted_pairs or key in existing:
                continue
            out.append(
                PairCandidate(
                    symbol=symbol,
                    horizon=horizon,
                    failed_count=99,
                    deficit_score=math.inf,
                    failed_reasons=["fallback_pair"],
                    source_report="fallback_priority",
                )
            )
    return out


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)


def _artifact_bundle_paths(models_root: Path, model_version: str, symbol: str, horizon: str) -> dict[str, Path]:
    symbol_dir = models_root / model_version / symbol
    return {
        "classification": symbol_dir / f"cls_{horizon}.joblib",
        "regression": symbol_dir / f"reg_{horizon}.joblib",
        "calibration": symbol_dir / f"calibration_{horizon}.json",
        "metrics": symbol_dir / f"metrics_{horizon}.json",
    }


def _artifact_bundle_exists(models_root: Path, model_version: str, symbol: str, horizon: str) -> bool:
    paths = _artifact_bundle_paths(models_root, model_version, symbol, horizon)
    return all(path.exists() for path in paths.values())


def _copy_artifact_bundle(
    models_root: Path,
    source_model_version: str,
    target_model_version: str,
    symbol: str,
    horizon: str,
) -> bool:
    source_paths = _artifact_bundle_paths(models_root, source_model_version, symbol, horizon)
    target_paths = _artifact_bundle_paths(models_root, target_model_version, symbol, horizon)
    if not all(path.exists() for path in source_paths.values()):
        return False
    target_dir = models_root / target_model_version / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    for key in source_paths:
        shutil.copy2(source_paths[key], target_paths[key])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously retrain near-promotion pairs and promote merged winners")
    parser.add_argument("--model-version", default=_default_model_version(), help="Candidate model version")
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3"], default="phase3", help="Promotion phase")
    parser.add_argument("--phase2-batch-size", type=int, default=20, help="Batch size when phase2 is selected")
    parser.add_argument("--max-pairs", type=int, default=6, help="Max near-promotion pairs to retrain per cycle")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol whitelist")
    parser.add_argument(
        "--target-pairs",
        default="",
        help="Optional comma-separated forced pair list in SYMBOL:HORIZON format. Overrides candidate ranking when set.",
    )
    parser.add_argument("--summary-report", default="", help="Optional explicit summary report path to source candidates")
    parser.add_argument("--output-prefix", default="reports", help="Output folder for reports")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip pre-training ingestion cycle")
    parser.add_argument(
        "--soft-promote-f1-delta-min",
        type=float,
        default=None,
        help="If set, near-pass pairs with f1 delta >= value can be soft-promoted after strict promotion step.",
    )
    parser.add_argument(
        "--soft-promote-accuracy-delta-min",
        type=float,
        default=None,
        help="If set, near-pass pairs with accuracy delta >= value can be soft-promoted after strict promotion step.",
    )
    parser.add_argument(
        "--soft-promote-rmse-delta-min",
        type=float,
        default=None,
        help="If set, near-pass pairs with rmse delta >= value can be soft-promoted after strict promotion step.",
    )
    args = parser.parse_args()

    settings = get_settings()
    symbols = _parse_symbols(args.symbols) if args.symbols.strip() else _parse_symbols(settings.supported_symbols)
    target_pairs = _parse_target_pairs(args.target_pairs) if args.target_pairs.strip() else []
    if target_pairs:
        target_symbols = {pair[0] for pair in target_pairs}
        symbols = sorted(set(symbols).union(target_symbols))
    if not symbols:
        raise SystemExit("No symbols configured for near-promotion retrain")

    output_root = PROJECT_ROOT / args.output_prefix
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root = PROJECT_ROOT / "reports"
    models_root = PROJECT_ROOT / "data" / "models"
    registry = ModelRegistry()
    registry_before = registry.read()
    promoted_pairs = _promoted_pair_set(registry_before)
    active_before = registry.get_active_model_version()

    if not args.skip_ingest:
        asyncio.run(run_ingestion_cycle())

    ranked: list[PairCandidate] = []
    payload: dict[str, object] = {}
    summary_path: Path | None = None
    if args.summary_report.strip():
        summary_path = (PROJECT_ROOT / args.summary_report).resolve()
        if not summary_path.exists():
            raise SystemExit(f"Summary report not found: {summary_path}")
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
        ranked = _rank_candidates(
            payload=payload,
            report_name=summary_path.name,
            allowed_symbols=set(symbols),
            promoted_pairs=promoted_pairs,
        )
    else:
        ranked_reports = _ranked_summary_reports(reports_root)
        if not ranked_reports:
            raise SystemExit("No summary_report_*.json found")
        for path in ranked_reports:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                continue
            attempt = _rank_candidates(
                payload=loaded,
                report_name=path.name,
                allowed_symbols=set(symbols),
                promoted_pairs=promoted_pairs,
            )
            summary_path = path
            payload = loaded
            ranked = attempt
            if attempt:
                break
    if summary_path is None:
        raise SystemExit("Unable to resolve a summary report for near-promotion retraining")
    print(
        json.dumps(
            {
                "near_promotion_source_report": str(summary_path),
                "candidate_pool_size": len(ranked),
                "already_promoted_pairs": len(promoted_pairs),
            }
        )
    )

    if target_pairs:
        selected = [
            PairCandidate(
                symbol=symbol,
                horizon=horizon,
                failed_count=0,
                deficit_score=0.0,
                failed_reasons=["target_pair"],
                source_report="manual_target_pairs",
            )
            for symbol, horizon in target_pairs
        ]
    else:
        selected = ranked[: max(args.max_pairs, 1)]
        selected_keys = {(item.symbol, item.horizon) for item in selected}
        if len(selected) < max(args.max_pairs, 1):
            for item in _fallback_pairs(symbols=symbols, promoted_pairs=promoted_pairs, existing=selected_keys):
                selected.append(item)
                selected_keys.add((item.symbol, item.horizon))
                if len(selected) >= max(args.max_pairs, 1):
                    break

    horizon_specs = {item.label: item for item in HORIZON_SPECS}
    results: list[dict[str, object]] = []
    for index, item in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] evaluating {item.symbol}:{item.horizon} "
            f"(failed_count={item.failed_count}, source={item.source_report})"
        )
        spec = horizon_specs.get(item.horizon)
        if spec is None:
            results.append(
                {
                    "symbol": item.symbol,
                    "horizon": item.horizon,
                    "status": "skipped",
                    "reason": "unsupported_horizon",
                }
            )
            continue
        entry = evaluate_symbol_horizon(
            symbol=item.symbol,
            spec=spec,
            model_version=args.model_version,
            models_root=models_root,
            write_artifacts=True,
        )
        if entry.get("status") == "ok":
            gate = entry.get("promotion_gate", {})
            delta = entry.get("near_pass_delta", {})
            print(
                json.dumps(
                    {
                        "pair": f"{item.symbol}:{item.horizon}",
                        "gate_passed": bool(gate.get("passed")) if isinstance(gate, dict) else False,
                        "failed_reasons": gate.get("failed_reasons", []) if isinstance(gate, dict) else [],
                        "near_pass_delta": delta if isinstance(delta, dict) else {},
                    }
                )
            )
        results.append(entry)

    carry_forward_pairs: list[dict[str, object]] = []
    for symbol, horizon in sorted(promoted_pairs):
        if _artifact_bundle_exists(models_root, args.model_version, symbol, horizon):
            carry_forward_pairs.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "status": "already_present",
                }
            )
            continue
        copied = False
        if active_before:
            copied = _copy_artifact_bundle(
                models_root=models_root,
                source_model_version=active_before,
                target_model_version=args.model_version,
                symbol=symbol,
                horizon=horizon,
            )
        if copied:
            carry_forward_pairs.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "status": "copied_from_active",
                    "source_model_version": active_before,
                }
            )
            continue

        spec = horizon_specs.get(horizon)
        if spec is None:
            carry_forward_pairs.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "status": "missing",
                    "reason": "unsupported_horizon",
                }
            )
            continue

        entry = evaluate_symbol_horizon(
            symbol=symbol,
            spec=spec,
            model_version=args.model_version,
            models_root=models_root,
            write_artifacts=True,
        )
        results.append(entry)
        carry_forward_pairs.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "status": "retrained" if entry.get("status") == "ok" else "retrain_failed",
                "reason": entry.get("reason"),
            }
        )
        print(
            json.dumps(
                {
                    "carry_forward_pair": f"{symbol}:{horizon}",
                    "status": carry_forward_pairs[-1]["status"],
                }
            )
        )

    summary_output = output_root / f"summary_report_{args.model_version}.json"
    summary_output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "model_version": args.model_version,
                "source_summary_report": str(summary_path),
                "selected_pairs": [_candidate_payload(item) for item in selected],
                "carry_forward_pairs": carry_forward_pairs,
                "results": results,
                "phase": args.phase,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    promote_output = output_root / f"promotion_report_{args.model_version}.json"
    promote_cmd = [
        sys.executable,
        "scripts/promote_model.py",
        "--candidate",
        args.model_version,
        "--phase",
        args.phase,
        "--merge-existing",
        "--output",
        str(promote_output.relative_to(PROJECT_ROOT)),
    ]
    if active_before:
        promote_cmd.extend(["--active", active_before])
    if args.phase == "phase2":
        promote_cmd.extend(["--phase2-batch-size", str(args.phase2_batch_size)])

    promote_proc = _run_command(promote_cmd)
    if promote_proc.returncode != 0:
        raise SystemExit(promote_proc.stderr[-4000:])

    soft_promoted_pairs: list[str] = []
    soft_mode_enabled = (
        args.soft_promote_f1_delta_min is not None
        or args.soft_promote_accuracy_delta_min is not None
        or args.soft_promote_rmse_delta_min is not None
    )
    if soft_mode_enabled:
        registry_after = ModelRegistry()
        payload_after = registry_after.read()
        promoted_after = payload_after.get("promoted", {})
        if not isinstance(promoted_after, dict):
            promoted_after = {}

        results_map: dict[tuple[str, str], dict[str, object]] = {}
        for entry in results:
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != "ok":
                continue
            symbol = str(entry.get("symbol", "")).upper()
            horizon = str(entry.get("horizon", ""))
            if symbol and horizon:
                results_map[(symbol, horizon)] = entry

        for candidate in selected:
            pair_key = (candidate.symbol, candidate.horizon)
            if pair_key in promoted_pairs:
                continue
            entry = results_map.get(pair_key)
            if entry is None:
                continue
            if not _artifact_bundle_exists(models_root, args.model_version, candidate.symbol, candidate.horizon):
                continue
            gate = entry.get("promotion_gate", {})
            if isinstance(gate, dict) and bool(gate.get("passed")):
                continue
            delta = entry.get("near_pass_delta", {})
            if not isinstance(delta, dict):
                continue
            f1_delta = _to_float(delta.get("f1_vs_baseline"), fallback=-math.inf)
            acc_delta = _to_float(delta.get("accuracy_vs_baseline"), fallback=-math.inf)
            rmse_delta = _to_float(delta.get("rmse_vs_baseline"), fallback=-math.inf)

            if args.soft_promote_f1_delta_min is not None and f1_delta < float(args.soft_promote_f1_delta_min):
                continue
            if args.soft_promote_accuracy_delta_min is not None and acc_delta < float(args.soft_promote_accuracy_delta_min):
                continue
            if args.soft_promote_rmse_delta_min is not None and rmse_delta < float(args.soft_promote_rmse_delta_min):
                continue

            symbol_map = promoted_after.get(candidate.symbol, {})
            if not isinstance(symbol_map, dict):
                symbol_map = {}
            symbol_map[candidate.horizon] = True
            promoted_after[candidate.symbol] = symbol_map
            soft_promoted_pairs.append(f"{candidate.symbol}:{candidate.horizon}")

        if soft_promoted_pairs:
            payload_after["active_model_version"] = args.model_version
            payload_after["promoted"] = promoted_after
            registry_after.write(payload_after)
            print(json.dumps({"soft_promoted_pairs": soft_promoted_pairs}))

    run_output = output_root / f"near_promotion_{args.model_version}.json"
    run_output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "model_version": args.model_version,
                "active_before": active_before,
                "active_after": ModelRegistry().get_active_model_version(),
                "source_summary_report": str(summary_path),
                "selected_pairs_count": len(selected),
                "selected_pairs": [_candidate_payload(item) for item in selected],
                "carry_forward_pairs": carry_forward_pairs,
                "soft_promoted_pairs": soft_promoted_pairs,
                "summary_report": str(summary_output),
                "promotion_report": str(promote_output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "model_version": args.model_version,
                "active_before": active_before,
                "active_after": ModelRegistry().get_active_model_version(),
                "selected_pairs": [f"{item.symbol}:{item.horizon}" for item in selected],
                "carry_forward_pairs": [f"{item['symbol']}:{item['horizon']}:{item['status']}" for item in carry_forward_pairs],
                "soft_promoted_pairs": soft_promoted_pairs,
                "summary_report": str(summary_output),
                "promotion_report": str(promote_output),
                "run_report": str(run_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
