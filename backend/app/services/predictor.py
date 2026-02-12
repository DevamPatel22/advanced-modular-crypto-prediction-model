from app.config import get_settings
from app.schemas.prediction import PredictionRequest, PredictionResponse


def generate_prediction(payload: PredictionRequest) -> PredictionResponse:
    """
    Temporary baseline inference stub.
    Replace this logic with real model inference once the training pipeline is integrated.
    """
    settings = get_settings()
    direction = "up" if payload.symbol in {"BTC-USD", "ETH-USD"} else "down"
    confidence = 0.64 if direction == "up" else 0.57
    predicted_close = 45000.0 if payload.symbol == "BTC-USD" else 3200.0 if payload.symbol == "ETH-USD" else 120.0

    debug = None
    if payload.include_debug:
        debug = {
            "inference_mode": "stub",
            "note": "Replace with model artifact inference in next milestone",
        }

    return PredictionResponse(
        symbol=payload.symbol,
        horizon=payload.horizon,
        direction=direction,
        confidence=confidence,
        predicted_close=predicted_close,
        model_version=settings.default_model_version,
        debug=debug,
    )

