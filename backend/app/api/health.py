"""Health and operations status endpoints for service/runtime visibility."""

from fastapi import APIRouter, Query

from app.config import get_settings
from app.services.data_readiness import (
    latest_data_quality_report,
    latest_robust_validation_report,
    latest_scorecard_report,
    latest_shadow_report,
    source_health_summary,
)

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Compute health check."""
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "version": settings.app_version,
    }


@router.get("/health/data-readiness")
def health_data_readiness() -> dict[str, object]:
    """Compute health data readiness."""
    quality = latest_data_quality_report()
    return {
        "status": "ok",
        "quality_report": quality,
    }


@router.get("/health/source-health")
def health_source_health(hours: int = Query(default=24, ge=1, le=168)) -> dict[str, object]:
    """Compute health source health."""
    summary = source_health_summary(hours=hours)
    return {
        "status": "ok",
        "source_health": summary,
    }


@router.get("/health/model-scorecard")
def health_model_scorecard() -> dict[str, object]:
    """Compute health model scorecard."""
    scorecard = latest_scorecard_report()
    return {
        "status": "ok",
        "scorecard": scorecard,
    }


@router.get("/health/robust-validation")
def health_robust_validation() -> dict[str, object]:
    """Expose the newest robust-validation bundle."""
    return {
        "status": "ok",
        "robust_validation": latest_robust_validation_report(),
    }


@router.get("/health/shadow-trading")
def health_shadow_trading() -> dict[str, object]:
    """Expose the newest live shadow-trading summary."""
    return {
        "status": "ok",
        "shadow_trading": latest_shadow_report(),
    }
