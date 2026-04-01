#!/usr/bin/env python3
"""Capture, settle, and report live shadow predictions for a frozen champion."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.ml.features import HORIZON_SPECS, HorizonSpec
from app.ml.training import load_candles
from app.schemas.prediction import PredictionRequest
from app.services.predictor import generate_prediction_for_model
from app.services.shadow_book import (
    append_shadow_prediction,
    read_shadow_champion,
    read_shadow_predictions,
    shadow_report_path,
    write_shadow_predictions,
)


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse `SYMBOL:HORIZON` CSV text."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"Invalid pair '{item}', expected SYMBOL:HORIZON")
        symbol, horizon = [part.strip() for part in item.split(":", 1)]
        pair = (symbol.upper(), horizon.lower())
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def _pair_list_from_promoted(promoted: dict[str, object]) -> list[tuple[str, str]]:
    """Flatten promoted map into symbol/horizon tuples."""
    pairs: list[tuple[str, str]] = []
    for symbol, horizons in promoted.items():
        if not isinstance(horizons, dict):
            continue
        for horizon, enabled in horizons.items():
            if bool(enabled):
                pairs.append((str(symbol).upper(), str(horizon).lower()))
    return sorted(set(pairs))


def _horizon_spec(horizon: str) -> HorizonSpec:
    """Resolve a horizon spec or fail fast."""
    for spec in HORIZON_SPECS:
        if spec.label == horizon:
            return spec
    raise SystemExit(f"Unsupported horizon: {horizon}")


def _granularity_seconds(granularity: str) -> int:
    """Map candle granularity label to seconds."""
    value = granularity.strip().lower()
    if value.endswith("mo") and value[:-2].isdigit():
        return int(value[:-2]) * 30 * 24 * 60 * 60
    unit = value[-1:] if value else ""
    qty = int(value[:-1]) if value[:-1].isdigit() else 1
    mapping = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
    return max(qty, 1) * mapping.get(unit, 3600)


def _realized_close(symbol: str, horizon: str, target_time: datetime) -> tuple[float | None, str | None]:
    """Resolve the nearest available realized close around the target timestamp."""
    spec = _horizon_spec(horizon)
    target_ts = int(target_time.timestamp())
    candidates = sorted(spec.candidates, key=lambda item: _granularity_seconds(item[0]))
    for granularity, _steps in candidates:
        frame = load_candles(symbol, granularity)
        if frame.empty:
            continue
        eligible = frame[frame["start_time"] <= target_ts]
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        return float(row["close"]), granularity
    return None, None


def _capture(args: argparse.Namespace) -> None:
    """Capture one shadow prediction per selected pair."""
    champion = read_shadow_champion()
    model_version = args.model_version.strip() or str(champion.get("model_version", ""))
    if not model_version:
        raise SystemExit("No frozen shadow champion found; run freeze_shadow_champion.py first or pass --model-version")
    promoted = champion.get("promoted", {})
    pair_source = _pair_list_from_promoted(promoted if isinstance(promoted, dict) else {})
    pairs = _parse_pairs(args.pairs) if args.pairs.strip() else pair_source
    if not pairs:
        raise SystemExit("No pairs selected for shadow capture")

    captured: list[dict[str, object]] = []
    now = datetime.now(tz=UTC)
    for symbol, horizon in pairs:
        payload = PredictionRequest(symbol=symbol, horizon=horizon, include_debug=False)
        response = generate_prediction_for_model(payload, model_version=model_version)
        row = {
            "id": str(uuid.uuid4()),
            "status": "pending",
            "captured_at": now.isoformat(),
            "model_version": model_version,
            "symbol": symbol,
            "horizon": horizon,
            "current_price": float(response.current_price),
            "predicted_close": float(response.predicted_close),
            "direction": response.direction,
            "confidence": float(response.confidence),
            "predicted_low_usd": float(response.predicted_low_usd),
            "predicted_high_usd": float(response.predicted_high_usd),
            "conformal_low_usd": float(response.conformal_low_usd),
            "conformal_high_usd": float(response.conformal_high_usd),
            "horizon_end_at": response.horizon_end_at.astimezone(UTC).isoformat(),
        }
        append_shadow_prediction(row)
        captured.append(row)

    print(json.dumps({"status": "ok", "captured_count": len(captured), "model_version": model_version, "pairs": [f"{s}:{h}" for s, h in pairs]}, indent=2))


def _settle(args: argparse.Namespace) -> None:
    """Settle matured shadow rows with realized market outcomes."""
    settings = get_settings()
    grace_seconds = max(int(args.grace_seconds if args.grace_seconds is not None else settings.shadow_settlement_grace_seconds), 0)
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=grace_seconds)
    fee_plus_slippage = (float(settings.execution_fee_bps) + float(settings.execution_slippage_bps)) / 10000.0

    rows = read_shadow_predictions()
    settled = 0
    for row in rows:
        if str(row.get("status", "")) != "pending":
            continue
        try:
            horizon_end_at = datetime.fromisoformat(str(row["horizon_end_at"]).replace("Z", "+00:00"))
        except Exception:
            continue
        if horizon_end_at > cutoff:
            continue
        realized_close, realized_granularity = _realized_close(str(row["symbol"]), str(row["horizon"]), horizon_end_at)
        if realized_close is None:
            continue
        current_price = float(row["current_price"])
        predicted_close = float(row["predicted_close"])
        raw_return = (realized_close / max(current_price, 1e-12)) - 1.0
        direction = str(row["direction"])
        signed_return = raw_return if direction == "up" else -raw_return
        net_return = signed_return - fee_plus_slippage
        realized_direction = "up" if realized_close >= current_price else "down"
        row.update(
            {
                "status": "settled",
                "settled_at": datetime.now(tz=UTC).isoformat(),
                "realized_close": float(realized_close),
                "realized_granularity": realized_granularity,
                "realized_direction": realized_direction,
                "direction_hit": bool(realized_direction == direction),
                "absolute_error_usd": abs(realized_close - predicted_close),
                "absolute_error_pct": abs((realized_close / max(predicted_close, 1e-12)) - 1.0),
                "signed_return": float(signed_return),
                "net_return_after_costs": float(net_return),
            }
        )
        settled += 1

    write_shadow_predictions(rows)
    print(json.dumps({"status": "ok", "settled_count": settled, "total_rows": len(rows)}, indent=2))


def _max_drawdown(returns: list[float]) -> float:
    """Compute max drawdown from sequential compounded returns."""
    if not returns:
        return 0.0
    equity = []
    running = 1.0
    for value in returns:
        running *= max(1.0 + float(value), 1e-6)
        equity.append(running)
    peak = 0.0
    drawdowns: list[float] = []
    for value in equity:
        peak = max(peak, value)
        drawdowns.append((value / max(peak, 1e-12)) - 1.0)
    return float(min(drawdowns)) if drawdowns else 0.0


def _summarize_shadow_rows(settled_rows: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Aggregate settled shadow rows into overall and per-pair summaries."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in settled_rows:
        key = (str(row.get("model_version", "")), str(row.get("symbol", "")), str(row.get("horizon", "")))
        grouped.setdefault(key, []).append(row)

    pair_rows: list[dict[str, object]] = []
    all_net_returns: list[float] = []
    all_hits = 0
    for (model_version, symbol, horizon), items in sorted(grouped.items()):
        hits = sum(bool(item.get("direction_hit")) for item in items)
        net_returns = [float(item.get("net_return_after_costs", 0.0)) for item in items]
        all_net_returns.extend(net_returns)
        all_hits += hits
        total_return = 1.0
        for value in net_returns:
            total_return *= max(1.0 + float(value), 1e-6)
        pair_rows.append(
            {
                "model_version": model_version,
                "symbol": symbol,
                "horizon": horizon,
                "signal_count": len(items),
                "direction_accuracy": float(hits / max(len(items), 1)),
                "avg_absolute_error_usd": float(sum(float(item.get("absolute_error_usd", 0.0)) for item in items) / max(len(items), 1)),
                "avg_absolute_error_pct": float(sum(float(item.get("absolute_error_pct", 0.0)) for item in items) / max(len(items), 1)),
                "avg_net_return_after_costs": float(sum(net_returns) / max(len(net_returns), 1)),
                "total_return_after_costs": float(total_return - 1.0),
                "max_drawdown_after_costs": _max_drawdown(net_returns),
                "passes_shadow_gate": bool(
                    len(items) >= 5
                    and (hits / max(len(items), 1)) >= 0.50
                    and (sum(net_returns) / max(len(net_returns), 1)) > 0.0
                    and _max_drawdown(net_returns) >= -0.45
                ),
            }
        )

    overall_total = 1.0
    for value in all_net_returns:
        overall_total *= max(1.0 + float(value), 1e-6)
    overall = {
        "signal_count": len(settled_rows),
        "pair_count": len(pair_rows),
        "direction_accuracy": float(all_hits / max(len(settled_rows), 1)) if settled_rows else 0.0,
        "avg_net_return_after_costs": float(sum(all_net_returns) / max(len(all_net_returns), 1)) if all_net_returns else 0.0,
        "total_return_after_costs": float(overall_total - 1.0) if all_net_returns else 0.0,
        "max_drawdown_after_costs": _max_drawdown(all_net_returns),
        "passing_pairs": sum(bool(item.get("passes_shadow_gate")) for item in pair_rows),
    }
    return overall, pair_rows


def build_shadow_report_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate settled shadow rows into a machine-readable report payload."""
    settled_rows = [row for row in rows if str(row.get("status", "")) == "settled"]
    model_versions = sorted({str(row.get("model_version", "")) for row in settled_rows if str(row.get("model_version", ""))})

    overall, pair_rows = _summarize_shadow_rows(settled_rows)
    models: dict[str, dict[str, object]] = {}
    for model_version in model_versions:
        model_rows = [row for row in settled_rows if str(row.get("model_version", "")) == model_version]
        model_overall, model_pairs = _summarize_shadow_rows(model_rows)
        models[model_version] = {
            "model_version": model_version,
            "overall": model_overall,
            "pairs": model_pairs,
        }

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_version": model_versions[0] if len(model_versions) == 1 else None,
        "model_versions": model_versions,
        "mixed_model_versions": len(model_versions) > 1,
        "overall": overall,
        "pairs": pair_rows,
        "models": models,
    }
    return report


def _report(args: argparse.Namespace) -> None:
    """Aggregate settled shadow rows into a machine-readable report."""
    report = build_shadow_report_payload(read_shadow_predictions())
    output = Path(args.output) if args.output else shadow_report_path()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    """Run subcommand dispatcher."""
    parser = argparse.ArgumentParser(description="Shadow trading utilities for frozen champion validation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture live shadow predictions")
    capture.add_argument("--model-version", default="", help="Override champion model version")
    capture.add_argument("--pairs", default="", help="Optional SYMBOL:HORIZON CSV subset")
    capture.set_defaults(handler=_capture)

    settle = subparsers.add_parser("settle", help="Settle matured shadow predictions")
    settle.add_argument("--grace-seconds", type=int, default=None, help="Optional extra wait before settlement")
    settle.set_defaults(handler=_settle)

    report = subparsers.add_parser("report", help="Aggregate settled shadow predictions into a report")
    report.add_argument("--output", default="", help="Optional output path override")
    report.set_defaults(handler=_report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
