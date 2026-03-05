"""Pydantic contracts for prediction request/response payloads."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

SupportedHorizon = Literal["5m", "1h", "3h", "6h", "12h", "1d", "1w", "1mo", "3mo"]


class PredictionRequest(BaseModel):
    symbol: str = Field(
        ...,
        min_length=5,
        max_length=24,
        pattern=r"^[A-Z0-9]+-[A-Z0-9]+$",
        description="Trading symbol",
    )
    horizon: SupportedHorizon = Field(default="1h", description="Prediction horizon")
    include_debug: bool = Field(default=False, description="Include debug metadata")


class PredictionResponse(BaseModel):
    symbol: str
    horizon: SupportedHorizon
    direction: Literal["up", "down"]
    market_bias: Literal["bullish", "bearish"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    current_price: float = Field(..., gt=0.0)
    predicted_close: float = Field(..., gt=0.0)
    predicted_low_usd: float = Field(..., gt=0.0, description="Projected lower bound close price for selected horizon (USD)")
    predicted_high_usd: float = Field(..., gt=0.0, description="Projected upper bound close price for selected horizon (USD)")
    conformal_low_usd: float = Field(..., gt=0.0, description="Conformal lower bound for selected horizon close (USD)")
    conformal_high_usd: float = Field(..., gt=0.0, description="Conformal upper bound for selected horizon close (USD)")
    conformal_confidence: float = Field(..., ge=0.0, le=1.0, description="Approximate conformal coverage confidence")
    return_range_min_pct: float = Field(..., description="Expected lower bound return for selected horizon (%)")
    return_range_max_pct: float = Field(..., description="Expected upper bound return for selected horizon (%)")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score on a 0-100 scale")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk category")
    horizon_end_at: datetime = Field(..., description="UTC timestamp for selected horizon end")
    model_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    debug: dict[str, str] | None = None
