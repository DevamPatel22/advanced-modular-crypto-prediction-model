from __future__ import annotations

import asyncio

from app.config import get_settings
from app.services.market_data import get_candles
from app.services.markets import fetch_symbols_by_quote

SUPPORTED_GRANULARITIES = {"1m", "5m", "15m", "1h", "6h", "1d"}


def _parse_granularities(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    filtered = [value for value in values if value in SUPPORTED_GRANULARITIES]
    return filtered or ["1h"]


async def run_ingestion_cycle() -> None:
    settings = get_settings()
    symbols = await fetch_symbols_by_quote(
        quote=settings.ingestion_quote_currency,
        limit=max(settings.ingestion_symbol_limit, 1),
    )
    granularities = _parse_granularities(settings.ingestion_granularities)
    limit = max(settings.ingestion_limit_per_symbol, 50)

    for market in symbols:
        for granularity in granularities:
            try:
                await get_candles(
                    symbol=market.symbol,
                    granularity=granularity,
                    limit=limit,
                    refresh=True,
                )
            except Exception:
                continue
            await asyncio.sleep(0.05)


async def run_ingestion_loop(stop_event: asyncio.Event) -> None:
    settings = get_settings()
    cycle_seconds = max(settings.ingestion_cycle_seconds, 60)
    while not stop_event.is_set():
        await run_ingestion_cycle()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=cycle_seconds)
        except asyncio.TimeoutError:
            pass
