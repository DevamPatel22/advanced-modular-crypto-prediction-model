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
    model_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    debug: dict[str, str] | None = None
