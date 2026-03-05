"""Prediction request endpoint."""

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.markets import is_tradable_symbol
from app.services.predictor import generate_prediction

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    """Compute predict."""
    if not await is_tradable_symbol(symbol=payload.symbol, quote="USD"):
        raise HTTPException(status_code=400, detail=f"{payload.symbol.upper()} is not a supported US-tradable USD pair")
    return generate_prediction(payload)
