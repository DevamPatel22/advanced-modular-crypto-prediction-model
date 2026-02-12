from app.config import get_settings
from app.schemas.prediction import PredictionRequest, PredictionResponse


def _deterministic_signal(seed: str) -> tuple[str, float, float]:
    score = sum(ord(ch) for ch in seed)
    direction = "up" if score % 2 == 0 else "down"
    confidence = 0.52 + (score % 33) / 100
    confidence = min(confidence, 0.89)
    predicted_close = 10.0 + float((score * 137) % 100000) / 10
    return direction, round(confidence, 4), round(predicted_close, 2)


def generate_prediction(payload: PredictionRequest) -> PredictionResponse:
    """
    Temporary baseline inference stub.
    Replace this logic with real model inference once the training pipeline is integrated.
    """
    settings = get_settings()
    direction, confidence, predicted_close = _deterministic_signal(f"{payload.symbol}:{payload.horizon}")

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
