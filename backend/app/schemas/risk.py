from __future__ import annotations

from pydantic import BaseModel, Field


class PortfolioRiskRequest(BaseModel):
    returns: list[float] = Field(..., min_length=10, description="PnL or return series in decimal form")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    annualization_factor: float = Field(default=365.0, gt=0.0, description="Periods per year for Sharpe/vol scaling")


class PortfolioRiskResponse(BaseModel):
    sample_count: int
    mean_return: float
    volatility: float
    sharpe: float
    var: float
    cvar: float
    max_drawdown: float


class RiskLimitCheckRequest(BaseModel):
    proposed_weights: dict[str, float] = Field(default_factory=dict)
    current_weights: dict[str, float] = Field(default_factory=dict)
    max_position_abs: float = Field(default=0.35, gt=0.0)
    max_gross_exposure: float = Field(default=1.0, gt=0.0)
    max_turnover: float = Field(default=0.5, ge=0.0)


class RiskLimitCheckResponse(BaseModel):
    accepted_weights: dict[str, float]
    gross_exposure: float
    turnover: float
    breached_limits: list[str]
