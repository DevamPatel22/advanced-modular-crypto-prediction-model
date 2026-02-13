from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Granularity = Literal["1m", "5m", "15m", "1h", "6h", "1d"]


class CandlePoint(BaseModel):
    start_time: int = Field(..., description="Unix timestamp in seconds for candle start")
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleSeriesResponse(BaseModel):
    symbol: str
    granularity: Granularity
    count: int
    source: str
    candles: list[CandlePoint]


class TickerResponse(BaseModel):
    symbol: str
    price: float = Field(..., gt=0.0)
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str

