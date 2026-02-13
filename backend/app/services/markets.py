from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.schemas.markets import MarketSymbol


@dataclass
class SymbolCache:
    quote: str
    expires_at: float
    symbols: list[MarketSymbol]


_cache: dict[str, SymbolCache] = {}


def _fallback_symbols(quote: str) -> list[MarketSymbol]:
    base = ["BTC", "ETH", "SOL", "LTC", "XRP", "ADA", "DOGE"]
    return [
        MarketSymbol(
            symbol=f"{asset}-{quote}",
            base_currency=asset,
            quote_currency=quote,
            status="online",
        )
        for asset in base
    ]


async def fetch_symbols_by_quote(quote: str, limit: int = 300) -> list[MarketSymbol]:
    settings = get_settings()
    normalized_quote = quote.upper().strip()
    now = time.time()

    cached = _cache.get(normalized_quote)
    if cached and cached.expires_at > now:
        return cached.symbols[:limit]

    symbols: list[MarketSymbol] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                settings.markets_source_url,
                headers={"Accept": "application/json", "User-Agent": "crypto-prediction-platform/0.1"},
            )
            response.raise_for_status()
            rows = response.json()
    except Exception:
        rows = []

    for row in rows:
        symbol = row.get("id")
        base_currency = row.get("base_currency")
        quote_currency = row.get("quote_currency")
        status = row.get("status", "unknown")
        trading_disabled = bool(row.get("trading_disabled", False))
        cancel_only = bool(row.get("cancel_only", False))
        post_only = bool(row.get("post_only", False))
        limit_only = bool(row.get("limit_only", False))

        if not symbol or not base_currency or not quote_currency:
            continue
        if quote_currency.upper() != normalized_quote:
            continue
        if status.lower() != "online":
            continue
        if trading_disabled or cancel_only or post_only or limit_only:
            continue

        symbols.append(
            MarketSymbol(
                symbol=symbol.upper(),
                base_currency=base_currency.upper(),
                quote_currency=quote_currency.upper(),
                status=status.lower(),
            )
        )

    if not symbols:
        symbols = _fallback_symbols(normalized_quote)

    symbols = sorted(symbols, key=lambda item: item.symbol)
    _cache[normalized_quote] = SymbolCache(
        quote=normalized_quote,
        expires_at=now + settings.symbol_cache_ttl_seconds,
        symbols=symbols,
    )
    return symbols[:limit]


async def is_tradable_symbol(symbol: str, quote: str = "USD") -> bool:
    normalized = symbol.upper().strip()
    universe = await fetch_symbols_by_quote(quote=quote, limit=2000)
    return any(item.symbol == normalized for item in universe)
