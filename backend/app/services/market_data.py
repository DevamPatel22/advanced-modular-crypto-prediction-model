from __future__ import annotations

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
    max_requests = 20

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
    max_requests = 20

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
    source = "coinbase_exchange"

    try:
        primary = await _fetch_exchange_candles(normalized_symbol, granularity, limit)
        primary_ok = True
    except Exception:
        primary = []

    settings = get_settings()
    if settings.market_data_secondary_source_enabled:
        try:
            secondary = await _fetch_binance_candles(normalized_symbol, granularity, limit)
            secondary_ok = True
        except Exception:
            secondary = []

    if primary_ok and secondary_ok:
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
            return "sqlite_cache_stale", cached
        raise RuntimeError(f"Unable to fetch candles from configured sources for {normalized_symbol}")

    _save_candles(normalized_symbol, granularity, merged, source=source)
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
