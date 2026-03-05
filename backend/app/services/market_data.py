from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.config import get_settings
from app.schemas.market_data import CandlePoint, TickerResponse

GRANULARITY_TO_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}
BINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "6h": "6h",
    "1d": "1d",
}

MAX_RECENCY_STEPS = 24
MAX_MEDIAN_CLOSE_DIVERGENCE = 0.12


def _to_binance_symbol(symbol: str) -> str:
    # Binance spot uses quote assets such as USDT for most crypto USD equivalents.
    base, _quote = symbol.split("-", 1)
    return f"{base}USDT"


def _database_path() -> Path:
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _connect_db() -> sqlite3.Connection:
    db_path = _database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_market_data_db() -> None:
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                granularity TEXT NOT NULL,
                start_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, granularity, start_time)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_candles_lookup
            ON candles(symbol, granularity, start_time DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                granularity TEXT NOT NULL,
                requested_limit INTEGER NOT NULL,
                primary_status TEXT NOT NULL,
                secondary_status TEXT NOT NULL,
                selected_source TEXT NOT NULL,
                primary_rows INTEGER NOT NULL,
                secondary_rows INTEGER NOT NULL,
                merged_rows INTEGER NOT NULL,
                divergence REAL,
                used_stale_cache INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_source_health_time
            ON source_health_events(event_time DESC)
            """
        )
        conn.commit()


def _record_source_health_event(
    *,
    symbol: str,
    granularity: str,
    requested_limit: int,
    primary_status: str,
    secondary_status: str,
    selected_source: str,
    primary_rows: int,
    secondary_rows: int,
    merged_rows: int,
    divergence: float | None,
    used_stale_cache: bool,
    error_message: str | None = None,
) -> None:
    event_time = datetime.now(tz=UTC).isoformat()
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO source_health_events (
                event_time,
                symbol,
                granularity,
                requested_limit,
                primary_status,
                secondary_status,
                selected_source,
                primary_rows,
                secondary_rows,
                merged_rows,
                divergence,
                used_stale_cache,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_time,
                symbol,
                granularity,
                int(requested_limit),
                primary_status,
                secondary_status,
                selected_source,
                int(primary_rows),
                int(secondary_rows),
                int(merged_rows),
                (float(divergence) if divergence is not None else None),
                1 if used_stale_cache else 0,
                error_message,
            ),
        )
        conn.commit()


def _load_candles_from_db(symbol: str, granularity: str, limit: int) -> list[CandlePoint]:
    with _connect_db() as conn:
        rows = conn.execute(
            """
            SELECT start_time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND granularity = ?
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (symbol, granularity, limit),
        ).fetchall()

    rows = list(reversed(rows))
    return [
        CandlePoint(
            start_time=int(row["start_time"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for row in rows
    ]


def _save_candles(symbol: str, granularity: str, candles: list[CandlePoint], source: str) -> None:
    if not candles:
        return
    with _connect_db() as conn:
        conn.executemany(
            """
            INSERT INTO candles (symbol, granularity, start_time, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, granularity, start_time) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                source = excluded.source
            """,
            [
                (
                    symbol,
                    granularity,
                    candle.start_time,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    source,
                )
                for candle in candles
            ],
        )
        conn.commit()


async def _fetch_exchange_candles(symbol: str, granularity: str, limit: int) -> list[CandlePoint]:
    settings = get_settings()
    granularity_seconds = GRANULARITY_TO_SECONDS[granularity]
    url = f"{settings.market_data_source_base_url}/products/{symbol}/candles"
    end_cursor = int(datetime.now(tz=UTC).timestamp())
    remaining = max(limit, 1)
    collected: dict[int, CandlePoint] = {}
    max_requests = 80

    async with httpx.AsyncClient(timeout=15.0) as client:
        for _ in range(max_requests):
            if remaining <= 0:
                break
            batch_size = min(remaining, 300)
            start_cursor = end_cursor - (granularity_seconds * batch_size)
            params = {
                "granularity": granularity_seconds,
                "start": datetime.fromtimestamp(start_cursor, tz=UTC).isoformat(),
                "end": datetime.fromtimestamp(end_cursor, tz=UTC).isoformat(),
            }
            response = await client.get(url, params=params, headers={"Accept": "application/json"})
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break

            oldest = end_cursor
            for row in rows:
                if not isinstance(row, list) or len(row) < 6:
                    continue
                candle = CandlePoint(
                    start_time=int(row[0]),
                    low=float(row[1]),
                    high=float(row[2]),
                    open=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                collected[candle.start_time] = candle
                oldest = min(oldest, candle.start_time)

            remaining = limit - len(collected)
            if len(rows) < batch_size:
                break
            next_end = oldest - granularity_seconds
            if next_end >= end_cursor:
                break
            end_cursor = next_end

    candles = sorted(collected.values(), key=lambda item: item.start_time)
    return candles[-limit:]


async def _fetch_binance_candles(symbol: str, granularity: str, limit: int) -> list[CandlePoint]:
    settings = get_settings()
    interval = BINANCE_INTERVAL_MAP[granularity]
    url = f"{settings.market_data_secondary_source_base_url}/api/v3/klines"
    binance_symbol = _to_binance_symbol(symbol)
    end_cursor_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    remaining = max(limit, 1)
    collected: dict[int, CandlePoint] = {}
    max_requests = 80

    async with httpx.AsyncClient(timeout=15.0) as client:
        for _ in range(max_requests):
            if remaining <= 0:
                break
            batch_size = min(remaining, 1000)
            params = {
                "symbol": binance_symbol,
                "interval": interval,
                "limit": batch_size,
                "endTime": end_cursor_ms,
            }
            response = await client.get(url, params=params, headers={"Accept": "application/json"})
            response.raise_for_status()
            rows = response.json()
            if not rows:
                break

            oldest_open_ms = end_cursor_ms
            for row in rows:
                if not isinstance(row, list) or len(row) < 6:
                    continue
                start_ms = int(row[0])
                candle = CandlePoint(
                    start_time=start_ms // 1000,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                collected[candle.start_time] = candle
                oldest_open_ms = min(oldest_open_ms, start_ms)

            remaining = limit - len(collected)
            if len(rows) < batch_size:
                break
            next_end_ms = oldest_open_ms - 1
            if next_end_ms >= end_cursor_ms:
                break
            end_cursor_ms = next_end_ms

    candles = sorted(collected.values(), key=lambda item: item.start_time)
    return candles[-limit:]


def _merge_candle_sets(primary: list[CandlePoint], secondary: list[CandlePoint], limit: int) -> list[CandlePoint]:
    merged: dict[int, CandlePoint] = {item.start_time: item for item in secondary}
    # Primary source takes precedence for overlapping timestamps.
    for item in primary:
        merged[item.start_time] = item
    candles = sorted(merged.values(), key=lambda item: item.start_time)
    return candles[-limit:]


def _is_candle_ohlcv_valid(candle: CandlePoint) -> bool:
    values = [candle.open, candle.high, candle.low, candle.close, candle.volume]
    if any((not math.isfinite(value)) for value in values):
        return False
    if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
        return False
    if candle.volume < 0:
        return False
    if candle.low > candle.high:
        return False
    if candle.open < candle.low or candle.open > candle.high:
        return False
    if candle.close < candle.low or candle.close > candle.high:
        return False
    return True


def _is_series_fresh(candles: list[CandlePoint], granularity: str) -> bool:
    if not candles:
        return False
    now_ts = int(datetime.now(tz=UTC).timestamp())
    step = GRANULARITY_TO_SECONDS[granularity]
    latest = candles[-1].start_time
    return (now_ts - latest) <= (step * MAX_RECENCY_STEPS)


def _passes_interval_consistency(candles: list[CandlePoint], granularity: str) -> bool:
    if len(candles) < 2:
        return True
    step = GRANULARITY_TO_SECONDS[granularity]
    prev = candles[0].start_time
    for candle in candles[1:]:
        current = candle.start_time
        if current <= prev:
            return False
        # allow occasional holes, but disallow pathological jumps / off-grid timestamps
        if ((current - prev) % step) != 0:
            return False
        prev = current
    return True


def _is_source_credible(candles: list[CandlePoint], granularity: str) -> bool:
    if not candles:
        return False
    if not _is_series_fresh(candles, granularity):
        return False
    if not _passes_interval_consistency(candles, granularity):
        return False
    if not all(_is_candle_ohlcv_valid(item) for item in candles):
        return False
    return True


def _median_close_divergence(primary: list[CandlePoint], secondary: list[CandlePoint]) -> float | None:
    secondary_by_ts = {item.start_time: item for item in secondary}
    ratios: list[float] = []
    for item in primary:
        other = secondary_by_ts.get(item.start_time)
        if other is None:
            continue
        denom = max(abs(item.close), 1e-9)
        ratios.append(abs(item.close - other.close) / denom)
    if not ratios:
        return None
    ratios.sort()
    mid = len(ratios) // 2
    if len(ratios) % 2 == 1:
        return ratios[mid]
    return 0.5 * (ratios[mid - 1] + ratios[mid])


async def get_candles(symbol: str, granularity: str, limit: int, refresh: bool = False) -> tuple[str, list[CandlePoint]]:
    normalized_symbol = symbol.upper().strip()
    if granularity not in GRANULARITY_TO_SECONDS:
        raise ValueError(f"Unsupported granularity: {granularity}")

    cached = _load_candles_from_db(normalized_symbol, granularity, limit)
    if cached and not refresh:
        return "sqlite_cache", cached

    primary: list[CandlePoint] = []
    secondary: list[CandlePoint] = []
    primary_ok = False
    secondary_ok = False
    primary_status = "not_attempted"
    secondary_status = "not_attempted"
    source = "coinbase_exchange"
    divergence: float | None = None

    try:
        primary = await _fetch_exchange_candles(normalized_symbol, granularity, limit)
        primary_ok = True
        primary_status = "fetched"
    except Exception:
        primary = []
        primary_status = "fetch_failed"

    settings = get_settings()
    if settings.market_data_secondary_source_enabled:
        try:
            secondary = await _fetch_binance_candles(normalized_symbol, granularity, limit)
            secondary_ok = True
            secondary_status = "fetched"
        except Exception:
            secondary = []
            secondary_status = "fetch_failed"
    else:
        secondary_status = "disabled"

    primary_ok = primary_ok and _is_source_credible(primary, granularity)
    secondary_ok = secondary_ok and _is_source_credible(secondary, granularity)
    if primary_status == "fetched":
        primary_status = "credible" if primary_ok else "not_credible"
    if secondary_status == "fetched":
        secondary_status = "credible" if secondary_ok else "not_credible"

    if primary_ok and secondary_ok:
        divergence = _median_close_divergence(primary, secondary)
        if divergence is not None and divergence > MAX_MEDIAN_CLOSE_DIVERGENCE:
            merged = primary
            source = "coinbase_exchange_divergence_guard"
        else:
            merged = _merge_candle_sets(primary=primary, secondary=secondary, limit=limit)
            source = "coinbase_binance_merged"
    elif primary_ok:
        merged = primary
        source = "coinbase_exchange"
    elif secondary_ok:
        merged = secondary
        source = "binance_spot"
    else:
        if cached:
            _record_source_health_event(
                symbol=normalized_symbol,
                granularity=granularity,
                requested_limit=limit,
                primary_status=primary_status,
                secondary_status=secondary_status,
                selected_source="sqlite_cache_stale",
                primary_rows=len(primary),
                secondary_rows=len(secondary),
                merged_rows=len(cached),
                divergence=divergence,
                used_stale_cache=True,
                error_message="both_live_sources_unavailable_or_not_credible",
            )
            return "sqlite_cache_stale", cached
        _record_source_health_event(
            symbol=normalized_symbol,
            granularity=granularity,
            requested_limit=limit,
            primary_status=primary_status,
            secondary_status=secondary_status,
            selected_source="none",
            primary_rows=len(primary),
            secondary_rows=len(secondary),
            merged_rows=0,
            divergence=divergence,
            used_stale_cache=False,
            error_message="both_live_sources_unavailable_or_not_credible",
        )
        raise RuntimeError(f"Unable to fetch candles from configured sources for {normalized_symbol}")

    _save_candles(normalized_symbol, granularity, merged, source=source)
    _record_source_health_event(
        symbol=normalized_symbol,
        granularity=granularity,
        requested_limit=limit,
        primary_status=primary_status,
        secondary_status=secondary_status,
        selected_source=source,
        primary_rows=len(primary),
        secondary_rows=len(secondary),
        merged_rows=len(merged),
        divergence=divergence,
        used_stale_cache=False,
    )
    return source, merged


async def get_ticker(symbol: str) -> TickerResponse:
    settings = get_settings()
    normalized_symbol = symbol.upper().strip()
    url = f"{settings.market_data_source_base_url}/products/{normalized_symbol}/ticker"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers={"Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()

        price = float(payload["price"])
        bid = float(payload["bid"]) if payload.get("bid") is not None else None
        ask = float(payload["ask"]) if payload.get("ask") is not None else None
        volume = float(payload["volume"]) if payload.get("volume") is not None else None
        return TickerResponse(
            symbol=normalized_symbol,
            price=price,
            bid=bid,
            ask=ask,
            volume=volume,
            source="coinbase_exchange_ticker",
        )
    except Exception:
        candles = _load_candles_from_db(normalized_symbol, "1m", limit=1)
        if candles:
            fallback_price = candles[-1].close
            return TickerResponse(
                symbol=normalized_symbol,
                price=fallback_price,
                source="sqlite_last_candle",
            )
        raise
