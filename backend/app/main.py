from contextlib import asynccontextmanager
import asyncio

"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.market_data import router as market_data_router
from app.api.markets import router as markets_router
from app.api.predictions import router as predictions_router
from app.config import get_settings
from app.services.ingestion import run_ingestion_loop
from app.services.market_data import init_market_data_db

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_market_data_db()
    stop_event = asyncio.Event()
    ingestion_task: asyncio.Task[None] | None = None
    if settings.ingestion_enabled:
        ingestion_task = asyncio.create_task(run_ingestion_loop(stop_event))
    try:
        yield
    finally:
        stop_event.set()
        if ingestion_task is not None:
            ingestion_task.cancel()
            try:
                await ingestion_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

allowed_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(market_data_router, prefix=settings.api_prefix)
app.include_router(markets_router, prefix=settings.api_prefix)
app.include_router(predictions_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "status": "running",
        "docs": "/docs",
    }
