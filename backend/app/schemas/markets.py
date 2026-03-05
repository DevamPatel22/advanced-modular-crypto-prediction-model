"""Pydantic contracts for tradable symbol discovery responses."""

from pydantic import BaseModel, Field


class MarketSymbol(BaseModel):
    symbol: str = Field(..., description="Exchange product symbol")
    base_currency: str = Field(..., description="Base asset")
    quote_currency: str = Field(..., description="Quote asset")
    status: str = Field(..., description="Exchange status")


class MarketSymbolsResponse(BaseModel):
    quote: str
    count: int
    symbols: list[MarketSymbol]
    source: str
