from fastapi import APIRouter

from app.schemas.risk import (
    PortfolioRiskRequest,
    PortfolioRiskResponse,
    RiskLimitCheckRequest,
    RiskLimitCheckResponse,
)
from app.services.risk_engine import apply_risk_limits, portfolio_risk_snapshot

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/portfolio-snapshot", response_model=PortfolioRiskResponse)
def portfolio_snapshot(payload: PortfolioRiskRequest) -> PortfolioRiskResponse:
    result = portfolio_risk_snapshot(
        returns=payload.returns,
        confidence_level=payload.confidence_level,
        annualization_factor=payload.annualization_factor,
    )
    return PortfolioRiskResponse(**result)


@router.post("/limit-check", response_model=RiskLimitCheckResponse)
def limit_check(payload: RiskLimitCheckRequest) -> RiskLimitCheckResponse:
    result = apply_risk_limits(
        proposed_weights=payload.proposed_weights,
        current_weights=payload.current_weights,
        max_position_abs=payload.max_position_abs,
        max_gross_exposure=payload.max_gross_exposure,
        max_turnover=payload.max_turnover,
    )
    return RiskLimitCheckResponse(**result)
