"""Market data fetch/cache service with source credibility checks and fallbacks."""

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
CRYPTOCOMPARE_HISTO_MAP: dict[str, tuple[str, int]] = {
    "1m": ("histominute", 1),
    "5m": ("histominute", 5),
    "15m": ("histominute", 15),
    "1h": ("histohour", 1),
    "6h": ("histohour", 6),
    "1d": ("histoday", 1),
}
CRYPTOCOMPARE_MAX_LIMIT = 2000

MAX_RECENCY_STEPS = 24
MAX_MEDIAN_CLOSE_DIVERGENCE = 0.12


def _split_symbol(symbol: str) -> tuple[str, str]:
    """Split symbol. Internal helper."""
    base, quote = symbol.split("-", 1)
    return base.upper(), quote.upper()


def _to_binance_symbol(symbol: str) -> str:
    # Binance spot uses quote assets such as USDT for most crypto USD equivalents.
    """Convert to binance symbol. Internal helper."""
    base, _quote = _split_symbol(symbol)
    return f"{base}USDT"


def _cryptocompare_quote_candidates(symbol: str) -> list[str]:
    """Build CryptoCompare quote candidates. Internal helper."""
    _base, quote = _split_symbol(symbol)
    candidates = [quote]
    if quote == "USD":
        candidates.append("USDT")
    deduped: list[str] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _database_path() -> Path:
    """Internal helper to compute database path."""
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _connect_db() -> sqlite3.Connection:
    """Internal helper to compute connect database."""
    db_path = _database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_market_data_db() -> None:
    """Initialize market data database."""
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
    """Internal helper to compute record source health event."""
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
    """Load candles from database. Internal helper."""
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
    """Save candles. Internal helper."""
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
    """Fetch exchange candles. Internal helper."""
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
    """Fetch binance candles. Internal helper."""
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


async def _fetch_cryptocompare_candles(symbol: str, granularity: str, limit: int) -> list[CandlePoint]:
    """Fetch cryptocompare candles. Internal helper."""
    settings = get_settings()
    histo, aggregate = CRYPTOCOMPARE_HISTO_MAP[granularity]
    url = f"{settings.market_data_tertiary_source_base_url}/data/v2/{histo}"
    base, _quote = _split_symbol(symbol)
    quote_candidates = _cryptocompare_quote_candidates(symbol)
    headers = {"Accept": "application/json"}
    api_key = settings.market_data_tertiary_source_api_key.strip()
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"

    best_collected: dict[int, CandlePoint] = {}
    last_error: Exception | None = None

    async with httpx.AsyncClient(timeout=15.0) as client:
        for quote in quote_candidates:
            end_cursor = int(datetime.now(tz=UTC).timestamp())
            remaining = max(limit, 1)
            collected: dict[int, CandlePoint] = {}
            max_requests = 80

            try:
                for _ in range(max_requests):
                    if remaining <= 0:
                        break
                    batch_size = min(remaining, CRYPTOCOMPARE_MAX_LIMIT)
                    params = {
                        "fsym": base,
                        "tsym": quote,
                        "aggregate": aggregate,
                        "limit": max(batch_size - 1, 1),
                        "toTs": end_cursor,
                    }
                    response = await client.get(url, params=params, headers=headers)
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        break
                    if str(payload.get("Response", "Success")).lower() != "success":
                        message = str(payload.get("Message", "unknown_error"))
                        raise RuntimeError(f"cryptocompare_error:{message}")

                    data_payload = payload.get("Data", {})
                    rows = data_payload.get("Data") if isinstance(data_payload, dict) else None
                    if not isinstance(rows, list) or not rows:
                        break

                    oldest = end_cursor
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        start_time = int(row.get("time", 0) or 0)
                        if start_time <= 0:
                            continue

                        open_price = float(row.get("open", 0.0) or 0.0)
                        high_price = float(row.get("high", 0.0) or 0.0)
                        low_price = float(row.get("low", 0.0) or 0.0)
                        close_price = float(row.get("close", 0.0) or 0.0)
                        volume = float(row.get("volumefrom", row.get("volumeto", 0.0)) or 0.0)

                        # Ignore placeholder intervals with zeroed prices.
                        if open_price <= 0 and high_price <= 0 and low_price <= 0 and close_price <= 0:
                            continue

                        candle = CandlePoint(
                            start_time=start_time,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=volume,
                        )
                        collected[candle.start_time] = candle
                        oldest = min(oldest, candle.start_time)

                    remaining = limit - len(collected)
                    if len(rows) < batch_size:
                        break
                    next_end = oldest - GRANULARITY_TO_SECONDS[granularity]
                    if next_end >= end_cursor:
                        break
                    end_cursor = next_end
            except Exception as exc:
                last_error = exc
                continue

            if len(collected) > len(best_collected):
                best_collected = collected

    if best_collected:
        candles = sorted(best_collected.values(), key=lambda item: item.start_time)
        return candles[-limit:]
    if last_error is not None:
        raise RuntimeError("cryptocompare_fetch_failed") from last_error
    return []


def _merge_candle_sets(primary: list[CandlePoint], secondary: list[CandlePoint], limit: int) -> list[CandlePoint]:
    """Merge candle sets. Internal helper."""
    merged: dict[int, CandlePoint] = {item.start_time: item for item in secondary}
    # Primary source takes precedence for overlapping timestamps.
    for item in primary:
        merged[item.start_time] = item
    candles = sorted(merged.values(), key=lambda item: item.start_time)
    return candles[-limit:]


def _is_candle_ohlcv_valid(candle: CandlePoint) -> bool:
    """Return whether candle OHLCV valid holds. Internal helper."""
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
    """Return whether series fresh holds. Internal helper."""
    if not candles:
        return False
    now_ts = int(datetime.now(tz=UTC).timestamp())
    step = GRANULARITY_TO_SECONDS[granularity]
    latest = candles[-1].start_time
    return (now_ts - latest) <= (step * MAX_RECENCY_STEPS)


def _passes_interval_consistency(candles: list[CandlePoint], granularity: str) -> bool:
    """Internal helper to compute passes interval consistency."""
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
    """Return whether source credible holds. Internal helper."""
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
    """Internal helper to compute median close divergence."""
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


def _select_preferred_backup(
    binance: list[CandlePoint],
    cryptocompare: list[CandlePoint],
) -> tuple[list[CandlePoint], str]:
    """Select preferred backup series. Internal helper."""
    if binance and not cryptocompare:
        return binance, "binance_spot"
    if cryptocompare and not binance:
        return cryptocompare, "cryptocompare_spot"
    if not binance and not cryptocompare:
        return [], "none"

    binance_latest = binance[-1].start_time if binance else -1
    cryptocompare_latest = cryptocompare[-1].start_time if cryptocompare else -1
    if cryptocompare_latest > binance_latest:
        return cryptocompare, "cryptocompare_spot"
    if binance_latest > cryptocompare_latest:
        return binance, "binance_spot"
    if len(cryptocompare) > len(binance):
        return cryptocompare, "cryptocompare_spot"
    return binance, "binance_spot"


async def get_candles(symbol: str, granularity: str, limit: int, refresh: bool = False) -> tuple[str, list[CandlePoint]]:
    """Get candles."""
    normalized_symbol = symbol.upper().strip()
    if granularity not in GRANULARITY_TO_SECONDS:
        raise ValueError(f"Unsupported granularity: {granularity}")

    cached = _load_candles_from_db(normalized_symbol, granularity, limit)
    if cached and not refresh:
        return "sqlite_cache", cached

    primary: list[CandlePoint] = []
    secondary: list[CandlePoint] = []
    binance: list[CandlePoint] = []
    tertiary: list[CandlePoint] = []
    primary_ok = False
    secondary_ok = False
    binance_ok = False
    tertiary_ok = False
    primary_status = "not_attempted"
    secondary_status = "not_attempted"
    binance_status = "not_attempted"
    tertiary_status = "not_attempted"
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
            binance = await _fetch_binance_candles(normalized_symbol, granularity, limit)
            binance_ok = True
            binance_status = "fetched"
        except Exception:
            binance = []
            binance_status = "fetch_failed"
    else:
        binance_status = "disabled"

    if settings.market_data_tertiary_source_enabled:
        try:
            tertiary = await _fetch_cryptocompare_candles(normalized_symbol, granularity, limit)
            tertiary_ok = True
            tertiary_status = "fetched"
        except Exception:
            tertiary = []
            tertiary_status = "fetch_failed"
    else:
        tertiary_status = "disabled"

    primary_ok = primary_ok and _is_source_credible(primary, granularity)
    binance_ok = binance_ok and _is_source_credible(binance, granularity)
    tertiary_ok = tertiary_ok and _is_source_credible(tertiary, granularity)
    if primary_status == "fetched":
        primary_status = "credible" if primary_ok else "not_credible"
    if binance_status == "fetched":
        binance_status = "credible" if binance_ok else "not_credible"
    if tertiary_status == "fetched":
        tertiary_status = "credible" if tertiary_ok else "not_credible"

    secondary, secondary_source = _select_preferred_backup(binance=binance if binance_ok else [], cryptocompare=tertiary if tertiary_ok else [])
    secondary_ok = bool(secondary)
    secondary_status = f"binance={binance_status};tertiary={tertiary_status}"
    secondary_rows_observed = max(len(secondary), len(binance), len(tertiary))

    # Source selection policy: merge when sources agree, otherwise prefer primary under divergence guard.
    if primary_ok and secondary_ok:
        divergence = _median_close_divergence(primary, secondary)
        if divergence is not None and divergence > MAX_MEDIAN_CLOSE_DIVERGENCE:
            merged = primary
            source = "coinbase_exchange_divergence_guard"
        else:
            merged = _merge_candle_sets(primary=primary, secondary=secondary, limit=limit)
            source = f"coinbase_{secondary_source}_merged"
    elif primary_ok:
        merged = primary
        source = "coinbase_exchange"
    elif secondary_ok:
        merged = secondary
        source = secondary_source
    else:
        # Last-resort behavior favors stale cached continuity over hard outage.
        if cached:
            _record_source_health_event(
                symbol=normalized_symbol,
                granularity=granularity,
                requested_limit=limit,
                primary_status=primary_status,
                secondary_status=secondary_status,
                selected_source="sqlite_cache_stale",
                primary_rows=len(primary),
                secondary_rows=secondary_rows_observed,
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
            secondary_rows=secondary_rows_observed,
            merged_rows=0,
            divergence=divergence,
            used_stale_cache=False,
            error_message="both_live_sources_unavailable_or_not_credible",
        )
        raise RuntimeError(f"Unable to fetch candles from configured sources for {normalized_symbol}")

    # Persist merged/selected candles so future requests can hit warm cache.
    _save_candles(normalized_symbol, granularity, merged, source=source)
    _record_source_health_event(
        symbol=normalized_symbol,
        granularity=granularity,
        requested_limit=limit,
        primary_status=primary_status,
        secondary_status=secondary_status,
        selected_source=source,
        primary_rows=len(primary),
        secondary_rows=secondary_rows_observed,
        merged_rows=len(merged),
        divergence=divergence,
        used_stale_cache=False,
    )
    return source, merged


async def get_ticker(symbol: str) -> TickerResponse:
    """Get ticker."""
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
