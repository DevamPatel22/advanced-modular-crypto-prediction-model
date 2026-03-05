#!/usr/bin/env python3
"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.market_data import GRANULARITY_TO_SECONDS

DEFAULT_GRANULARITIES = ["1m", "5m", "15m", "1h", "6h", "1d"]
DEFAULT_MIN_ROWS = {
    "1m": 20000,
    "5m": 12000,
    "15m": 9000,
    "1h": 4000,
    "6h": 1800,
    "1d": 1200,
}
DEFAULT_MAX_STALE_STEPS = {
    "1m": 180,
    "5m": 72,
    "15m": 72,
    "1h": 36,
    "6h": 16,
    "1d": 5,
}


def _parse_csv(raw: str) -> list[str]:
    """Parse CSV. Internal helper."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_symbols(raw: str) -> list[str]:
    """Parse symbols. Internal helper."""
    return sorted(set(item.upper() for item in _parse_csv(raw)))


def _parse_granularities(raw: str) -> list[str]:
    """Parse granularities. Internal helper."""
    out: list[str] = []
    for granularity in _parse_csv(raw):
        if granularity in GRANULARITY_TO_SECONDS:
            out.append(granularity)
    return out


def _parse_int_map(raw: str, fallback: dict[str, int]) -> dict[str, int]:
    """Parse int map. Internal helper."""
    parsed = dict(fallback)
    if not raw.strip():
        return parsed
    for token in _parse_csv(raw):
        if ":" not in token:
            continue
        key_raw, value_raw = token.split(":", 1)
        key = key_raw.strip()
        try:
            value = int(value_raw.strip())
        except ValueError:
            continue
        if key in GRANULARITY_TO_SECONDS and value > 0:
            parsed[key] = value
    return parsed


def _db_path() -> Path:
    """Internal helper to compute database path."""
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _query_timestamps(conn: sqlite3.Connection, symbol: str, granularity: str) -> list[int]:
    """Internal helper to compute query timestamps."""
    rows = conn.execute(
        """
        SELECT start_time
        FROM candles
        WHERE symbol = ? AND granularity = ?
        ORDER BY start_time ASC
        """,
        (symbol, granularity),
    ).fetchall()
    return [int(row[0]) for row in rows]


def _pair_quality(
    symbol: str,
    granularity: str,
    timestamps: list[int],
    min_rows: int,
    max_stale_steps: int,
    max_gap_ratio: float,
) -> dict[str, object]:
    """Internal helper to compute pair quality."""
    now_ts = int(datetime.now(tz=UTC).timestamp())
    step_seconds = GRANULARITY_TO_SECONDS[granularity]
    row_count = len(timestamps)

    earliest = timestamps[0] if timestamps else None
    latest = timestamps[-1] if timestamps else None
    freshness_steps = ((now_ts - latest) / step_seconds) if latest is not None else math.inf

    off_grid_intervals = 0
    gap_intervals = 0
    expected_rows = row_count
    if row_count >= 2:
        expected_rows = int(((latest - earliest) / step_seconds) + 1) if earliest is not None and latest is not None else row_count
        prev = timestamps[0]
        for current in timestamps[1:]:
            diff = current - prev
            if diff <= 0 or (diff % step_seconds) != 0:
                off_grid_intervals += 1
            if diff > step_seconds:
                gap_intervals += 1
            prev = current

    missing_rows = max(expected_rows - row_count, 0)
    coverage_ratio = (row_count / expected_rows) if expected_rows > 0 else 0.0
    gap_ratio = (gap_intervals / max(row_count - 1, 1)) if row_count > 1 else 0.0
    history_days = (row_count * step_seconds) / 86400.0

    checks = {
        "rows_ok": bool(row_count >= min_rows),
        "fresh_ok": bool(freshness_steps <= max_stale_steps),
        "off_grid_ok": bool(off_grid_intervals == 0),
        "gap_ok": bool(gap_ratio <= max_gap_ratio),
    }
    gate_passed = all(checks.values())

    return {
        "symbol": symbol,
        "granularity": granularity,
        "row_count": row_count,
        "min_rows_required": min_rows,
        "history_days_estimate": round(history_days, 2),
        "earliest_start_time": earliest,
        "latest_start_time": latest,
        "staleness_steps": (round(float(freshness_steps), 2) if math.isfinite(freshness_steps) else None),
        "max_stale_steps_allowed": max_stale_steps,
        "expected_rows_by_span": expected_rows,
        "missing_rows_by_span": missing_rows,
        "coverage_ratio": round(float(coverage_ratio), 6),
        "gap_intervals": gap_intervals,
        "gap_ratio": round(float(gap_ratio), 6),
        "max_gap_ratio_allowed": max_gap_ratio,
        "off_grid_intervals": off_grid_intervals,
        "checks": checks,
        "gate_passed": gate_passed,
    }


def main() -> None:
    """Run the script entrypoint."""
    parser = argparse.ArgumentParser(description="Generate data quality report for market candle history")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Defaults to SUPPORTED_SYMBOLS.")
    parser.add_argument(
        "--granularities",
        default=",".join(DEFAULT_GRANULARITIES),
        help="Comma-separated granularities to audit",
    )
    parser.add_argument(
        "--min-rows-map",
        default="",
        help="Override minimum rows per granularity, format: 1m:20000,1h:4000",
    )
    parser.add_argument(
        "--max-stale-steps-map",
        default="",
        help="Override max stale steps per granularity, format: 1m:180,1h:36",
    )
    parser.add_argument(
        "--max-gap-ratio",
        type=float,
        default=0.10,
        help="Maximum allowed gap interval ratio per pair",
    )
    parser.add_argument("--output", default="reports/data_quality_report.json", help="Output JSON path")
    args = parser.parse_args()

    settings = get_settings()
    symbols = _parse_symbols(args.symbols) if args.symbols.strip() else _parse_symbols(settings.supported_symbols)
    granularities = _parse_granularities(args.granularities)
    if not symbols:
        raise SystemExit("No symbols configured for quality report")
    if not granularities:
        raise SystemExit("No valid granularities configured for quality report")

    min_rows_map = _parse_int_map(args.min_rows_map, DEFAULT_MIN_ROWS)
    max_stale_steps_map = _parse_int_map(args.max_stale_steps_map, DEFAULT_MAX_STALE_STEPS)
    max_gap_ratio = float(max(args.max_gap_ratio, 0.0))

    db_path = _db_path()
    if not db_path.exists():
        raise SystemExit(f"Market data DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        report_pairs: list[dict[str, object]] = []
        for symbol in symbols:
            for granularity in granularities:
                timestamps = _query_timestamps(conn, symbol=symbol, granularity=granularity)
                pair_report = _pair_quality(
                    symbol=symbol,
                    granularity=granularity,
                    timestamps=timestamps,
                    min_rows=int(min_rows_map.get(granularity, 100)),
                    max_stale_steps=int(max_stale_steps_map.get(granularity, 24)),
                    max_gap_ratio=max_gap_ratio,
                )
                report_pairs.append(pair_report)

    failing_pairs = [item for item in report_pairs if not bool(item.get("gate_passed"))]
    symbol_summary: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        entries = [item for item in report_pairs if item["symbol"] == symbol]
        symbol_summary[symbol] = {
            "pair_count": len(entries),
            "failing_pair_count": sum(1 for item in entries if not bool(item.get("gate_passed"))),
            "gate_passed": all(bool(item.get("gate_passed")) for item in entries),
        }

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "db_path": str(db_path),
        "symbols": symbols,
        "granularities": granularities,
        "config": {
            "min_rows_map": {k: int(v) for k, v in min_rows_map.items() if k in granularities},
            "max_stale_steps_map": {k: int(v) for k, v in max_stale_steps_map.items() if k in granularities},
            "max_gap_ratio": max_gap_ratio,
        },
        "pair_count": len(report_pairs),
        "failing_pair_count": len(failing_pairs),
        "gate_passed": len(failing_pairs) == 0,
        "symbol_summary": symbol_summary,
        "failing_pairs": failing_pairs,
        "pairs": report_pairs,
    }

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
