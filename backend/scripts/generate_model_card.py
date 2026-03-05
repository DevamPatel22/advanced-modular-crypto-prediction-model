#!/usr/bin/env python3
"""Generate a model card summarizing strengths, weaknesses, and risk controls."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json(path: Path) -> dict[str, object] | None:
    """Internal helper to compute load json."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest(pattern: str) -> Path | None:
    """Internal helper to compute latest."""
    paths = sorted((PROJECT_ROOT / "reports").glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _latest_summary_with_symbols(model_version: str) -> Path | None:
    """Internal helper to compute latest summary with symbols."""
    pattern = f"summary_report_{model_version}*.json" if model_version else "summary_report*.json"
    paths = sorted((PROJECT_ROOT / "reports").glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        payload = _load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("symbols"), dict):
            return path
    return None


def main() -> None:
    """Run the script entrypoint."""
    parser = argparse.ArgumentParser(description="Generate model card markdown")
    parser.add_argument("--model-version", default="", help="Model version hint for report discovery")
    parser.add_argument("--summary-report", default="", help="Summary report path")
    parser.add_argument("--promotion-report", default="", help="Promotion report path")
    parser.add_argument("--scorecard-report", default="", help="Scorecard report path")
    parser.add_argument("--output", default="", help="Output markdown path")
    args = parser.parse_args()

    model_version = args.model_version.strip()
    summary_path = Path(args.summary_report) if args.summary_report.strip() else None
    promotion_path = Path(args.promotion_report) if args.promotion_report.strip() else None
    scorecard_path = Path(args.scorecard_report) if args.scorecard_report.strip() else None
    if summary_path is not None and not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    if promotion_path is not None and not promotion_path.is_absolute():
        promotion_path = PROJECT_ROOT / promotion_path
    if scorecard_path is not None and not scorecard_path.is_absolute():
        scorecard_path = PROJECT_ROOT / scorecard_path

    if summary_path is None:
        summary_path = _latest_summary_with_symbols(model_version)
    if promotion_path is None:
        promotion_path = _latest(f"promotion_report_{model_version}*.json") if model_version else _latest("promotion_report*.json")
    if scorecard_path is None:
        scorecard_path = _latest(f"scorecard_{model_version}*.json") if model_version else _latest("scorecard*.json")

    summary = _load_json(summary_path) if summary_path else None
    promotion = _load_json(promotion_path) if promotion_path else None
    scorecard = _load_json(scorecard_path) if scorecard_path else None
    if summary is None:
        raise SystemExit("Unable to load summary report for model card generation")

    if not model_version:
        model_version = str(summary.get("model_version", "unknown"))
    if not args.output.strip():
        output_path = REPO_ROOT / "docs" / "model-cards" / f"{model_version}.md"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    promoted_pairs = int(promotion.get("promoted_pairs", 0)) if isinstance(promotion, dict) else 0
    promoted_symbols = int(promotion.get("promoted_symbols", 0)) if isinstance(promotion, dict) else 0
    core_summary = scorecard.get("core_summary", {}) if isinstance(scorecard, dict) else {}
    overall_summary = scorecard.get("overall_summary", {}) if isinstance(scorecard, dict) else {}
    failed_reasons = scorecard.get("top_failed_reasons", []) if isinstance(scorecard, dict) else []

    lines: list[str] = []
    lines.append(f"# Model Card: {model_version}")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(tz=UTC).isoformat()}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Asset class: US-tradable USD crypto pairs")
    lines.append("- Model family: ensemble classification + ensemble regression with optional regime routing")
    lines.append("- Decision controls: confidence calibration, meta-labeling abstention, conformal intervals")
    lines.append("")
    lines.append("## What Works")
    lines.append("")
    lines.append(f"- Promoted symbols: {promoted_symbols}")
    lines.append(f"- Promoted pairs: {promoted_pairs}")
    lines.append(
        f"- Core gate pass rate: {float(core_summary.get('gate_pass_rate', 0.0)):.4f} "
        f"({int(core_summary.get('pair_count', 0))} evaluated core pairs)"
    )
    lines.append(
        f"- Overall gate pass rate: {float(overall_summary.get('gate_pass_rate', 0.0)):.4f} "
        f"({int(overall_summary.get('pair_count', 0))} evaluated pairs)"
    )
    lines.append(
        f"- Core avg net-return edge vs baseline: {float(core_summary.get('avg_net_return_edge', 0.0)):.8f}"
    )
    lines.append("")
    lines.append("## What Does Not Work Yet")
    lines.append("")
    if isinstance(failed_reasons, list) and failed_reasons:
        for item in failed_reasons[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('reason', 'unknown')}: {int(item.get('count', 0))} pairs")
    else:
        lines.append("- No failure breakdown available in scorecard.")
    lines.append("")
    lines.append("## Risk and Reliability")
    lines.append("")
    lines.append(f"- Core avg max drawdown: {float(core_summary.get('avg_max_drawdown', 0.0)):.6f}")
    lines.append(f"- Overall avg max drawdown: {float(overall_summary.get('avg_max_drawdown', 0.0)):.6f}")
    lines.append(f"- Core avg abstain rate: {float(core_summary.get('avg_abstain_rate', 0.0)):.4f}")
    lines.append("- Promotion gate requires baseline outperformance, leakage checks, and execution/risk constraints.")
    lines.append("")
    lines.append("## Failure Modes")
    lines.append("")
    lines.append("- Thin or stale short-horizon data can block promotions.")
    lines.append("- Regime shifts can reduce directional edge and increase false positives.")
    lines.append("- Transaction costs can erase raw predictive edge if turnover is too high.")
    lines.append("")
    lines.append("## Mitigations")
    lines.append("")
    lines.append("- Enforce SLA gates for freshness, coverage, and source uptime before retraining.")
    lines.append("- Use meta-labeling abstention to reduce low-edge trades.")
    lines.append("- Use conformal intervals and drawdown constraints to keep risk bounded.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Summary report: `{summary_path}`")
    lines.append(f"- Promotion report: `{promotion_path}`")
    lines.append(f"- Scorecard report: `{scorecard_path}`")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "model_version": model_version,
                "model_card": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
