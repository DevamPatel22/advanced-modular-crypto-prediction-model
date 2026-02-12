from fastapi import APIRouter, Query

from app.schemas.markets import MarketSymbolsResponse
from app.services.markets import fetch_symbols_by_quote

router = APIRouter(prefix="/markets", tags=["markets"])


@router.get("/symbols", response_model=MarketSymbolsResponse)
async def list_symbols(
    quote: str = Query(default="USD", min_length=3, max_length=6),
    limit: int = Query(default=300, ge=1, le=2000),
) -> MarketSymbolsResponse:
    normalized_quote = quote.upper()
    symbols = await fetch_symbols_by_quote(normalized_quote, limit=limit)
    return MarketSymbolsResponse(
        quote=normalized_quote,
        count=len(symbols),
        symbols=symbols,
        source="coinbase_exchange_products",
    )

