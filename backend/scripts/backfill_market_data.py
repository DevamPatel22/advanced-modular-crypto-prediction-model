#!/usr/bin/env python3
"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from __future__ import annotations

import argparse
import asyncio
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
from app.services.market_data import GRANULARITY_TO_SECONDS, get_candles

DEFAULT_TARGET_ROWS = {
    "1m": 30000,
    "5m": 18000,
    "15m": 12000,
    "1h": 6000,
    "6h": 2500,
    "1d": 1500,
}
DEFAULT_MAX_STALE_STEPS = {
    "1m": 180,
    "5m": 72,
    "15m": 72,
    "1h": 36,
    "6h": 16,
    "1d": 5,
}
DEFAULT_DEEP_HISTORY_MAX_LIMIT = {
    "1m": 60000,
    "5m": 30000,
    "15m": 20000,
    "1h": 15000,
    "6h": 12000,
    "1d": 5000,
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


def _earliest_start_time(conn: sqlite3.Connection, symbol: str, granularity: str) -> int | None:
    """Load earliest candle timestamp for a symbol/granularity pair."""
    row = conn.execute(
        """
        SELECT MIN(start_time)
        FROM candles
        WHERE symbol = ? AND granularity = ?
        """,
        (symbol, granularity),
    ).fetchone()
    value = row[0] if row else None
    return int(value) if value is not None else None


def _latest_start_time(conn: sqlite3.Connection, symbol: str, granularity: str) -> int | None:
    """Load latest candle timestamp for a symbol/granularity pair."""
    row = conn.execute(
        """
        SELECT MAX(start_time)
        FROM candles
        WHERE symbol = ? AND granularity = ?
        """,
        (symbol, granularity),
    ).fetchone()
    value = row[0] if row else None
    return int(value) if value is not None else None


def _timestamp_to_iso(value: int | None) -> str | None:
    """Convert unix timestamps to ISO-8601 for reporting."""
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), tz=UTC).isoformat()


def _history_days(row_count: int, granularity: str) -> float:
    """Estimate archive depth in days from row count and cadence."""
    step_seconds = int(GRANULARITY_TO_SECONDS[granularity])
    return round(float((row_count * step_seconds) / 86400.0), 2)


def _staleness_steps(latest_start_time: int | None, granularity: str) -> float:
    """Convert latest timestamp to staleness expressed in granularity steps."""
    if latest_start_time is None:
        return math.inf
    now_ts = int(datetime.now(tz=UTC).timestamp())
    step_seconds = int(GRANULARITY_TO_SECONDS[granularity])
    if step_seconds <= 0:
        return math.inf
    return float((now_ts - int(latest_start_time)) / step_seconds)


def _apply_anchor_parity_targets(
    *,
    db_path: Path,
    target_rows: dict[str, int],
    granularities: list[str],
    anchor_symbol: str,
) -> tuple[dict[str, int], dict[str, object]]:
    """Raise target row floors to anchor symbol depth for selected granularities."""
    adjusted = dict(target_rows)
    anchor_rows: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for granularity in granularities:
            rows = _row_count(conn, symbol=anchor_symbol, granularity=granularity)
            anchor_rows[granularity] = int(rows)
            if rows > int(adjusted.get(granularity, 0)):
                adjusted[granularity] = int(rows)
    parity_payload: dict[str, object] = {
        "enabled": True,
        "anchor_symbol": anchor_symbol,
        "anchor_rows": anchor_rows,
        "target_rows_after_parity": {key: int(adjusted.get(key, 0)) for key in granularities},
    }
    return adjusted, parity_payload


def _next_deep_history_limit(
    *,
    before_rows: int,
    target_rows: int,
    max_limit: int,
) -> int:
    """Choose the next fetch window for archive-deepening passes."""
    floor = max(int(target_rows), 1)
    cap = max(int(max_limit), floor)
    if before_rows <= 0:
        return min(floor, cap)
    growth_base = max(before_rows + 1, floor)
    doubled = max(int(math.ceil(before_rows * 1.8)), growth_base)
    return min(doubled, cap)


def _final_count_entry(
    *,
    symbol: str,
    granularity: str,
    target: int,
    rows: int,
    earliest: int | None,
    latest: int | None,
    deep_history: bool,
    deep_history_max_limit: int | None,
) -> dict[str, object]:
    """Build final pair-level archive status payload."""
    entry = {
        "symbol": symbol,
        "granularity": granularity,
        "target_rows": target,
        "rows": rows,
        "target_met": rows >= target,
        "earliest_start_time": earliest,
        "earliest_iso": _timestamp_to_iso(earliest),
        "latest_start_time": latest,
        "latest_iso": _timestamp_to_iso(latest),
        "history_days_estimate": _history_days(rows, granularity),
    }
    if deep_history and deep_history_max_limit is not None:
        entry["deep_history_max_limit"] = int(deep_history_max_limit)
        entry["at_deep_history_cap"] = rows >= int(deep_history_max_limit)
    return entry


async def _run_backfill(
    symbols: list[str],
    granularities: list[str],
    target_rows: dict[str, int],
    max_stale_steps_map: dict[str, int],
    max_passes: int,
    sleep_ms: int,
    refresh_if_stale: bool,
    parity_payload: dict[str, object] | None = None,
    *,
    deep_history: bool,
    deep_history_max_limit_map: dict[str, int],
) -> dict[str, object]:
    """Run backfill. Internal helper."""
    db_path = _db_path()
    if not db_path.exists():
        raise RuntimeError(f"Market data DB not found: {db_path}")

    passes: list[dict[str, object]] = []
    delay_seconds = max(sleep_ms, 0) / 1000.0
    termination_reason = "max_passes_reached"

    for pass_index in range(1, max_passes + 1):
        pass_events: list[dict[str, object]] = []
        pass_rows_added = 0
        pass_history_extensions = 0
        pass_refreshes = 0

        with sqlite3.connect(db_path) as conn:
            for symbol in symbols:
                for granularity in granularities:
                    target = int(target_rows.get(granularity, 500))
                    max_stale_steps = int(max_stale_steps_map.get(granularity, 0))
                    before_rows = _row_count(conn, symbol=symbol, granularity=granularity)
                    earliest_before = _earliest_start_time(conn, symbol=symbol, granularity=granularity)
                    latest_before = _latest_start_time(conn, symbol=symbol, granularity=granularity)
                    staleness_before_steps = _staleness_steps(latest_before, granularity)
                    stale_refresh_needed = (
                        bool(refresh_if_stale)
                        and (
                            not math.isfinite(staleness_before_steps)
                            or staleness_before_steps > max_stale_steps
                        )
                    )

                    request_limit = target
                    deep_history_max_limit: int | None = None
                    if deep_history:
                        deep_history_max_limit = int(deep_history_max_limit_map.get(granularity, target))
                        request_limit = _next_deep_history_limit(
                            before_rows=before_rows,
                            target_rows=target,
                            max_limit=deep_history_max_limit,
                        )
                        if request_limit <= before_rows and not stale_refresh_needed:
                            pass_events.append(
                                {
                                    "symbol": symbol,
                                    "granularity": granularity,
                                    "target_rows": target,
                                    "before_rows": before_rows,
                                    "after_rows": before_rows,
                                    "request_limit": request_limit,
                                    "deep_history_max_limit": deep_history_max_limit,
                                    "status": "deep_history_cap_reached",
                                    "earliest_start_time_before": earliest_before,
                                    "latest_start_time_before": latest_before,
                                    "staleness_before_steps": (
                                        round(float(staleness_before_steps), 3)
                                        if math.isfinite(staleness_before_steps)
                                        else None
                                    ),
                                    "max_stale_steps_allowed": max_stale_steps,
                                }
                            )
                            continue
                    else:
                        if before_rows >= target and not stale_refresh_needed:
                            pass_events.append(
                                {
                                    "symbol": symbol,
                                    "granularity": granularity,
                                    "target_rows": target,
                                    "before_rows": before_rows,
                                    "after_rows": before_rows,
                                    "request_limit": target,
                                    "status": "target_already_met",
                                    "earliest_start_time_before": earliest_before,
                                    "latest_start_time_before": latest_before,
                                    "staleness_before_steps": (
                                        round(float(staleness_before_steps), 3)
                                        if math.isfinite(staleness_before_steps)
                                        else None
                                    ),
                                    "max_stale_steps_allowed": max_stale_steps,
                                }
                            )
                            continue

                    try:
                        source, candles = await get_candles(
                            symbol=symbol,
                            granularity=granularity,
                            limit=request_limit,
                            refresh=True,
                        )
                        after_rows = _row_count(conn, symbol=symbol, granularity=granularity)
                        earliest_after = _earliest_start_time(conn, symbol=symbol, granularity=granularity)
                        latest_after = _latest_start_time(conn, symbol=symbol, granularity=granularity)
                        staleness_after_steps = _staleness_steps(latest_after, granularity)
                        row_delta = max(after_rows - before_rows, 0)
                        history_extended = (
                            earliest_after is not None
                            and (earliest_before is None or earliest_after < earliest_before)
                        )
                        latest_refreshed = (
                            latest_after is not None
                            and (latest_before is None or latest_after > latest_before)
                        )
                        stale_fallback = source == "sqlite_cache_stale"
                        if stale_fallback:
                            status = "stale_fallback"
                        elif history_extended:
                            status = "deep_history_extended" if deep_history else "target_extended"
                        elif stale_refresh_needed and latest_refreshed:
                            status = "stale_refresh_ok"
                        else:
                            status = "no_material_change"

                        pass_rows_added += row_delta
                        pass_history_extensions += 1 if history_extended else 0
                        pass_refreshes += 1 if latest_refreshed else 0

                        pass_events.append(
                            {
                                "symbol": symbol,
                                "granularity": granularity,
                                "target_rows": target,
                                "before_rows": before_rows,
                                "after_rows": after_rows,
                                "row_delta": row_delta,
                                "request_limit": request_limit,
                                "deep_history_max_limit": deep_history_max_limit,
                                "fetched_candles": len(candles),
                                "source": source,
                                "status": status,
                                "refresh_applied": not stale_fallback,
                                "history_extended": history_extended,
                                "latest_refreshed": latest_refreshed,
                                "earliest_start_time_before": earliest_before,
                                "earliest_start_time_after": earliest_after,
                                "latest_start_time_before": latest_before,
                                "latest_start_time_after": latest_after,
                                "staleness_before_steps": (
                                    round(float(staleness_before_steps), 3)
                                    if math.isfinite(staleness_before_steps)
                                    else None
                                ),
                                "staleness_after_steps": (
                                    round(float(staleness_after_steps), 3)
                                    if math.isfinite(staleness_after_steps)
                                    else None
                                ),
                                "max_stale_steps_allowed": max_stale_steps,
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
                                "request_limit": request_limit,
                                "deep_history_max_limit": deep_history_max_limit,
                                "status": "failed",
                                "error": str(exc),
                                "earliest_start_time_before": earliest_before,
                                "latest_start_time_before": latest_before,
                                "staleness_before_steps": (
                                    round(float(staleness_before_steps), 3)
                                    if math.isfinite(staleness_before_steps)
                                    else None
                                ),
                                "max_stale_steps_allowed": max_stale_steps,
                            }
                        )

                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)

        passes.append(
            {
                "pass_index": pass_index,
                "rows_added": pass_rows_added,
                "history_extensions": pass_history_extensions,
                "latest_refreshes": pass_refreshes,
                "events": pass_events,
            }
        )

        if deep_history:
            if pass_rows_added == 0 and pass_history_extensions == 0 and pass_refreshes == 0:
                termination_reason = "no_archive_progress"
                break
            continue

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
            termination_reason = "all_targets_met"
            break

    final_counts: list[dict[str, object]] = []
    unmet_pairs: list[dict[str, object]] = []
    stale_fallback_pairs: list[dict[str, object]] = []
    deep_history_cap_pairs: list[dict[str, object]] = []

    with sqlite3.connect(db_path) as conn:
        for symbol in symbols:
            for granularity in granularities:
                target = int(target_rows.get(granularity, 500))
                rows = _row_count(conn, symbol=symbol, granularity=granularity)
                earliest = _earliest_start_time(conn, symbol=symbol, granularity=granularity)
                latest = _latest_start_time(conn, symbol=symbol, granularity=granularity)
                deep_cap = (
                    int(deep_history_max_limit_map.get(granularity, target))
                    if deep_history
                    else None
                )
                entry = _final_count_entry(
                    symbol=symbol,
                    granularity=granularity,
                    target=target,
                    rows=rows,
                    earliest=earliest,
                    latest=latest,
                    deep_history=deep_history,
                    deep_history_max_limit=deep_cap,
                )
                final_counts.append(entry)
                if rows < target:
                    unmet_pairs.append(entry)
                if deep_history and deep_cap is not None and rows >= deep_cap:
                    deep_history_cap_pairs.append(entry)
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
        "max_stale_steps_map": {key: int(value) for key, value in max_stale_steps_map.items() if key in granularities},
        "refresh_if_stale": bool(refresh_if_stale),
        "parity": parity_payload or {"enabled": False},
        "deep_history": {
            "enabled": bool(deep_history),
            "max_limit_map": (
                {key: int(value) for key, value in deep_history_max_limit_map.items() if key in granularities}
                if deep_history
                else {}
            ),
        },
        "passes_executed": len(passes),
        "termination_reason": termination_reason,
        "passes": passes,
        "final_counts": final_counts,
        "unmet_pairs": unmet_pairs,
        "stale_fallback_pairs": stale_fallback_pairs,
        "deep_history_cap_pairs": deep_history_cap_pairs,
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
    parser.add_argument(
        "--max-stale-steps-map",
        default="",
        help="Override max stale steps map, format: 1m:180,1h:36",
    )
    parser.add_argument(
        "--deep-history-max-limit-map",
        default="",
        help="Deep-history cap per granularity, format: 1m:60000,1h:15000",
    )
    parser.add_argument("--max-passes", type=int, default=3, help="Maximum backfill passes")
    parser.add_argument("--sleep-ms", type=int, default=50, help="Delay between API calls in milliseconds")
    parser.add_argument(
        "--parity-anchor-symbol",
        default="",
        help="Optional anchor symbol to force target-row parity by granularity (e.g., BTC-USD)",
    )
    parser.add_argument(
        "--deep-history",
        action="store_true",
        help="Increase fetch windows pass-by-pass until archive depth stops extending or caps are reached",
    )
    parser.add_argument(
        "--refresh-if-stale",
        dest="refresh_if_stale",
        action="store_true",
        help="Force refresh for stale pairs even when target rows are already met",
    )
    parser.add_argument(
        "--no-refresh-if-stale",
        dest="refresh_if_stale",
        action="store_false",
        help="Disable stale refresh when target rows are already met",
    )
    parser.add_argument("--output", default="reports/backfill_market_data.json", help="Output report path")
    parser.set_defaults(refresh_if_stale=True)
    args = parser.parse_args()

    settings = get_settings()
    symbols = _parse_symbols(args.symbols) if args.symbols.strip() else _parse_symbols(settings.supported_symbols)
    granularities = _parse_granularities(args.granularities)
    if not symbols:
        raise SystemExit("No symbols configured for backfill")
    if not granularities:
        raise SystemExit("No valid granularities configured for backfill")

    target_rows = _parse_int_map(args.target_rows_map, DEFAULT_TARGET_ROWS)
    max_stale_steps_map = _parse_int_map(args.max_stale_steps_map, DEFAULT_MAX_STALE_STEPS)
    deep_history_max_limit_map = _parse_int_map(args.deep_history_max_limit_map, DEFAULT_DEEP_HISTORY_MAX_LIMIT)
    parity_payload: dict[str, object] | None = None
    if args.parity_anchor_symbol.strip():
        anchor_symbol = args.parity_anchor_symbol.strip().upper()
        target_rows, parity_payload = _apply_anchor_parity_targets(
            db_path=_db_path(),
            target_rows=target_rows,
            granularities=granularities,
            anchor_symbol=anchor_symbol,
        )
    payload = asyncio.run(
        _run_backfill(
            symbols=symbols,
            granularities=granularities,
            target_rows=target_rows,
            max_stale_steps_map=max_stale_steps_map,
            max_passes=max(1, args.max_passes),
            sleep_ms=max(0, args.sleep_ms),
            refresh_if_stale=bool(args.refresh_if_stale),
            parity_payload=parity_payload,
            deep_history=bool(args.deep_history),
            deep_history_max_limit_map=deep_history_max_limit_map,
        )
    )

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
