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


def _parse_csv(raw: str) -> list[str]:
    """Parse CSV. Internal helper."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_symbols(raw: str) -> list[str]:
    """Parse symbols. Internal helper."""
    return sorted(set(item.upper() for item in _parse_csv(raw)))


def _parse_granularities(raw: str) -> list[str]:
    """Parse granularities. Internal helper."""
    out: list[str] = []
    for value in _parse_csv(raw):
        if value in GRANULARITY_TO_SECONDS:
            out.append(value)
    return out


def _db_path() -> Path:
    """Internal helper to compute database path."""
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _timestamp_to_iso(value: int | None) -> str | None:
    """Convert unix timestamps to ISO-8601 for reporting."""
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), tz=UTC).isoformat()


def _query_sources(conn: sqlite3.Connection, symbol: str, granularity: str) -> dict[str, int]:
    """Build a source-row-count breakdown for one symbol/granularity archive."""
    rows = conn.execute(
        """
        SELECT source, COUNT(*) AS row_count
        FROM candles
        WHERE symbol = ? AND granularity = ?
        GROUP BY source
        ORDER BY row_count DESC, source ASC
        """,
        (symbol, granularity),
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def _pair_archive_payload(conn: sqlite3.Connection, symbol: str, granularity: str) -> dict[str, object]:
    """Compute archive coverage for one symbol/granularity pair."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS row_count, MIN(start_time) AS earliest, MAX(start_time) AS latest
        FROM candles
        WHERE symbol = ? AND granularity = ?
        """,
        (symbol, granularity),
    ).fetchone()
    row_count = int(row[0]) if row and row[0] is not None else 0
    earliest = int(row[1]) if row and row[1] is not None else None
    latest = int(row[2]) if row and row[2] is not None else None
    step_seconds = int(GRANULARITY_TO_SECONDS[granularity])
    expected_rows = 0
    if earliest is not None and latest is not None and latest >= earliest:
        expected_rows = int(((latest - earliest) / step_seconds) + 1)
    coverage_ratio = (row_count / expected_rows) if expected_rows > 0 else 0.0
    history_days = (row_count * step_seconds) / 86400.0
    staleness_steps = None
    if latest is not None:
        staleness_steps = round(float((int(datetime.now(tz=UTC).timestamp()) - latest) / step_seconds), 2)
    source_row_counts = _query_sources(conn, symbol, granularity)

    return {
        "symbol": symbol,
        "granularity": granularity,
        "row_count": row_count,
        "earliest_start_time": earliest,
        "earliest_iso": _timestamp_to_iso(earliest),
        "latest_start_time": latest,
        "latest_iso": _timestamp_to_iso(latest),
        "history_days_estimate": round(float(history_days), 2),
        "expected_rows_by_span": expected_rows,
        "coverage_ratio": round(float(coverage_ratio), 6) if math.isfinite(coverage_ratio) else None,
        "staleness_steps": staleness_steps,
        "source_row_counts": source_row_counts,
        "sources": list(source_row_counts.keys()),
    }


def main() -> None:
    """Run the script entrypoint."""
    parser = argparse.ArgumentParser(description="Report stored market-data archive coverage by symbol/granularity")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list. Defaults to SUPPORTED_SYMBOLS.")
    parser.add_argument(
        "--granularities",
        default="1m,5m,15m,1h,6h,1d",
        help="Comma-separated granularities",
    )
    parser.add_argument("--output", default="reports/history_completeness_report.json", help="Output JSON path")
    args = parser.parse_args()

    settings = get_settings()
    symbols = _parse_symbols(args.symbols) if args.symbols.strip() else _parse_symbols(settings.supported_symbols)
    granularities = _parse_granularities(args.granularities)
    if not symbols:
        raise SystemExit("No symbols configured for history completeness report")
    if not granularities:
        raise SystemExit("No valid granularities configured for history completeness report")

    db_path = _db_path()
    if not db_path.exists():
        raise SystemExit(f"Market data DB not found: {db_path}")

    pairs: list[dict[str, object]] = []
    summary_by_granularity: dict[str, dict[str, object]] = {}
    with sqlite3.connect(db_path) as conn:
        for symbol in symbols:
            for granularity in granularities:
                pairs.append(_pair_archive_payload(conn, symbol, granularity))

    for granularity in granularities:
        subset = [pair for pair in pairs if pair["granularity"] == granularity]
        row_counts = [int(pair["row_count"]) for pair in subset]
        earliest_values = [int(pair["earliest_start_time"]) for pair in subset if isinstance(pair["earliest_start_time"], int)]
        latest_values = [int(pair["latest_start_time"]) for pair in subset if isinstance(pair["latest_start_time"], int)]
        summary_by_granularity[granularity] = {
            "pairs_covered": len(subset),
            "total_rows": int(sum(row_counts)),
            "min_rows": int(min(row_counts)) if row_counts else 0,
            "max_rows": int(max(row_counts)) if row_counts else 0,
            "earliest_start_time_min": int(min(earliest_values)) if earliest_values else None,
            "earliest_iso_min": _timestamp_to_iso(min(earliest_values)) if earliest_values else None,
            "latest_start_time_max": int(max(latest_values)) if latest_values else None,
            "latest_iso_max": _timestamp_to_iso(max(latest_values)) if latest_values else None,
        }

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "db_path": str(db_path),
        "symbols": symbols,
        "granularities": granularities,
        "pairs": pairs,
        "summary_by_granularity": summary_by_granularity,
    }

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
