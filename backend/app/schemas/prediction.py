from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

SupportedSymbol = Literal["BTC-USD", "ETH-USD", "SOL-USD"]
SupportedHorizon = Literal["5m", "1h", "4h"]


class PredictionRequest(BaseModel):
    symbol: SupportedSymbol = Field(..., description="Trading symbol")
    horizon: SupportedHorizon = Field(default="1h", description="Prediction horizon")
    include_debug: bool = Field(default=False, description="Include debug metadata")


class PredictionResponse(BaseModel):
    symbol: SupportedSymbol
    horizon: SupportedHorizon
    direction: Literal["up", "down"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_close: float = Field(..., gt=0.0)
    model_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    debug: dict[str, str] | None = None

