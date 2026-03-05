"""Health and operations status endpoints for service/runtime visibility."""

from fastapi import APIRouter, Query

from app.config import get_settings
from app.services.data_readiness import latest_data_quality_report, source_health_summary

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
