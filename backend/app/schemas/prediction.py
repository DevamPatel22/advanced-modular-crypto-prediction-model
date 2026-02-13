from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

SupportedHorizon = Literal["5m", "1h", "6h", "12h", "1d", "1w", "1mo", "3mo"]


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
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_close: float = Field(..., gt=0.0)
    return_range_min_pct: float = Field(..., description="Expected lower bound return for selected horizon (%)")
    return_range_max_pct: float = Field(..., description="Expected upper bound return for selected horizon (%)")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score on a 0-100 scale")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk category")
    model_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    debug: dict[str, str] | None = None
