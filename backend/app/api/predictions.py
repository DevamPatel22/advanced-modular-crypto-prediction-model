from fastapi import APIRouter

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.predictor import generate_prediction

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    return generate_prediction(payload)

