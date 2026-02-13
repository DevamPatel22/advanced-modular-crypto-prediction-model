import asyncio

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.schemas.market_data import CandleSeriesResponse, Granularity, TickerResponse
from app.services.market_data import get_candles, get_ticker

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/candles", response_model=CandleSeriesResponse)
async def candles(
    symbol: str = Query(..., min_length=5, max_length=24, pattern=r"^[A-Z0-9]+-[A-Z0-9]+$"),
    granularity: Granularity = Query(default="1h"),
    limit: int = Query(default=200, ge=1, le=300),
    refresh: bool = Query(default=False),
) -> CandleSeriesResponse:
    try:
        source, points = await get_candles(symbol=symbol, granularity=granularity, limit=limit, refresh=refresh)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Failed to load candles for {symbol}") from error

    return CandleSeriesResponse(
        symbol=symbol.upper(),
        granularity=granularity,
        count=len(points),
        source=source,
        candles=points,
    )


@router.get("/ticker", response_model=TickerResponse)
async def ticker(
    symbol: str = Query(..., min_length=5, max_length=24, pattern=r"^[A-Z0-9]+-[A-Z0-9]+$"),
) -> TickerResponse:
    try:
        return await get_ticker(symbol)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Failed to load ticker for {symbol}") from error


@router.websocket("/ws/ticker")
async def ticker_stream(
    websocket: WebSocket,
    symbol: str = Query(..., min_length=5, max_length=24, pattern=r"^[A-Z0-9]+-[A-Z0-9]+$"),
):
    settings = get_settings()
    interval = max(settings.ticker_stream_interval_seconds, 1)
    await websocket.accept()
    try:
        while True:
            payload = await get_ticker(symbol)
            await websocket.send_json(payload.model_dump(mode="json"))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011, reason="ticker stream unavailable")

