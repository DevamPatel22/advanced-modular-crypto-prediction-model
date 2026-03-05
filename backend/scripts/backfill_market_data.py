#!/usr/bin/env python3
"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.market_data import GRANULARITY_TO_SECONDS, get_candles

DEFAULT_TARGET_ROWS = {
    "1m": 30000,
    "5m": 18000,
    "15m": 12000,
    "1h": 6000,
    "6h": 2500,
    "1d": 1500,
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
    for value in _parse_csv(raw):
        if value in GRANULARITY_TO_SECONDS:
            out.append(value)
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


def _row_count(conn: sqlite3.Connection, symbol: str, granularity: str) -> int:
    """Internal helper to compute row count."""
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM candles
        WHERE symbol = ? AND granularity = ?
        """,
        (symbol, granularity),
    ).fetchone()
    return int(row[0]) if row else 0


async def _run_backfill(
    symbols: list[str],
    granularities: list[str],
    target_rows: dict[str, int],
    max_passes: int,
    sleep_ms: int,
) -> dict[str, object]:
    """Run backfill. Internal helper."""
    db_path = _db_path()
    if not db_path.exists():
        raise RuntimeError(f"Market data DB not found: {db_path}")

    passes: list[dict[str, object]] = []
    delay_seconds = max(sleep_ms, 0) / 1000.0

    for pass_index in range(1, max_passes + 1):
        pass_events: list[dict[str, object]] = []
        pass_progress = {"pass_index": pass_index, "events": pass_events}

        with sqlite3.connect(db_path) as conn:
            for symbol in symbols:
                for granularity in granularities:
                    target = int(target_rows.get(granularity, 500))
                    before_rows = _row_count(conn, symbol=symbol, granularity=granularity)
                    if before_rows >= target:
                        pass_events.append(
                            {
                                "symbol": symbol,
                                "granularity": granularity,
                                "target_rows": target,
                                "before_rows": before_rows,
                                "after_rows": before_rows,
                                "status": "target_already_met",
                            }
                        )
                        continue

                    try:
                        source, candles = await get_candles(
                            symbol=symbol,
                            granularity=granularity,
                            limit=target,
                            refresh=True,
                        )
                        after_rows = _row_count(conn, symbol=symbol, granularity=granularity)
                        stale_fallback = source == "sqlite_cache_stale"
                        status = "stale_fallback" if stale_fallback else "ok"
                        pass_events.append(
                            {
                                "symbol": symbol,
                                "granularity": granularity,
                                "target_rows": target,
                                "before_rows": before_rows,
                                "fetched_candles": len(candles),
                                "after_rows": after_rows,
                                "status": status,
                                "source": source,
                                "refresh_applied": not stale_fallback,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        pass_events.append(
                            {
                                "symbol": symbol,
                                "granularity": granularity,
                                "target_rows": target,
                                "before_rows": before_rows,
                                "after_rows": before_rows,
                                "status": "failed",
                                "error": str(exc),
                            }
                        )

                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)

        passes.append(pass_progress)

        # Stop early if all targets are met.
        all_met = True
        with sqlite3.connect(db_path) as conn:
            for symbol in symbols:
                for granularity in granularities:
                    if _row_count(conn, symbol=symbol, granularity=granularity) < int(target_rows.get(granularity, 500)):
                        all_met = False
                        break
                if not all_met:
                    break
        if all_met:
            break

    final_counts: list[dict[str, object]] = []
    unmet_pairs: list[dict[str, object]] = []
    stale_fallback_pairs: list[dict[str, object]] = []
    with sqlite3.connect(db_path) as conn:
        for symbol in symbols:
            for granularity in granularities:
                target = int(target_rows.get(granularity, 500))
                rows = _row_count(conn, symbol=symbol, granularity=granularity)
                entry = {
                    "symbol": symbol,
                    "granularity": granularity,
                    "target_rows": target,
                    "rows": rows,
                    "target_met": rows >= target,
                }
                final_counts.append(entry)
                if rows < target:
                    unmet_pairs.append(entry)
                stale_hit = any(
                    event.get("symbol") == symbol
                    and event.get("granularity") == granularity
                    and event.get("status") == "stale_fallback"
                    for batch in passes
                    for event in batch.get("events", [])
                )
                if stale_hit:
                    stale_fallback_pairs.append(entry)

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "db_path": str(db_path),
        "symbols": symbols,
        "granularities": granularities,
        "target_rows": {key: int(value) for key, value in target_rows.items() if key in granularities},
        "passes_executed": len(passes),
        "passes": passes,
        "final_counts": final_counts,
        "unmet_pairs": unmet_pairs,
        "stale_fallback_pairs": stale_fallback_pairs,
        "all_targets_met": len(unmet_pairs) == 0,
    }


def main() -> None:
    """Run the script entrypoint."""
    parser = argparse.ArgumentParser(description="Backfill market candle data to target row depth")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list. Defaults to SUPPORTED_SYMBOLS.")
    parser.add_argument(
        "--granularities",
        default="1m,5m,15m,1h,6h,1d",
        help="Comma-separated granularities",
    )
    parser.add_argument(
        "--target-rows-map",
        default="",
        help="Override target rows map, format: 1m:30000,1h:6000",
    )
    parser.add_argument("--max-passes", type=int, default=3, help="Maximum backfill passes")
    parser.add_argument("--sleep-ms", type=int, default=50, help="Delay between API calls in milliseconds")
    parser.add_argument("--output", default="reports/backfill_market_data.json", help="Output report path")
    args = parser.parse_args()

    settings = get_settings()
    symbols = _parse_symbols(args.symbols) if args.symbols.strip() else _parse_symbols(settings.supported_symbols)
    granularities = _parse_granularities(args.granularities)
    if not symbols:
        raise SystemExit("No symbols configured for backfill")
    if not granularities:
        raise SystemExit("No valid granularities configured for backfill")

    target_rows = _parse_int_map(args.target_rows_map, DEFAULT_TARGET_ROWS)
    payload = asyncio.run(
        _run_backfill(
            symbols=symbols,
            granularities=granularities,
            target_rows=target_rows,
            max_passes=max(1, args.max_passes),
            sleep_ms=max(0, args.sleep_ms),
        )
    )

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
