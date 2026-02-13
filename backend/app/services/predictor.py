import math
import sqlite3
from pathlib import Path
from statistics import pstdev

from app.config import get_settings
from app.schemas.prediction import PredictionRequest, PredictionResponse


def _deterministic_signal(seed: str) -> tuple[str, float, float]:
    score = sum(ord(ch) for ch in seed)
    direction = "up" if score % 2 == 0 else "down"
    confidence = 0.52 + (score % 33) / 100
    confidence = min(confidence, 0.89)
    predicted_close = 10.0 + float((score * 137) % 100000) / 10
    return direction, round(confidence, 4), round(predicted_close, 2)


HORIZON_TO_DATA_WINDOW: dict[str, tuple[str, int]] = {
    "5m": ("1m", 5),
    "1h": ("1h", 1),
    "6h": ("1h", 6),
    "12h": ("1h", 12),
    "1d": ("1h", 24),
    "1w": ("1d", 7),
    "1mo": ("1d", 30),
    "3mo": ("1d", 90),
}


def _db_path() -> Path:
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _load_recent_closes(symbol: str, granularity: str, limit: int = 500) -> list[float]:
    db_path = _db_path()
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT close
            FROM candles
            WHERE symbol = ? AND granularity = ?
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (symbol.upper(), granularity, limit),
        ).fetchall()
    closes = [float(row[0]) for row in reversed(rows) if row and row[0] is not None]
    return closes


def _realized_volatility(symbol: str, horizon: str) -> tuple[float, str]:
    granularity, steps = HORIZON_TO_DATA_WINDOW.get(horizon, ("1h", 1))
    closes = _load_recent_closes(symbol, granularity, limit=800)
    if len(closes) < 40:
        # fallback baseline volatility if historical data is still thin
        return 0.012 * math.sqrt(max(steps, 1)), granularity

    log_returns: list[float] = []
    for idx in range(1, len(closes)):
        prev_close = closes[idx - 1]
        close = closes[idx]
        if prev_close <= 0 or close <= 0:
            continue
        log_returns.append(math.log(close / prev_close))

    if len(log_returns) < 30:
        return 0.012 * math.sqrt(max(steps, 1)), granularity

    step_vol = pstdev(log_returns)
    horizon_vol = step_vol * math.sqrt(max(steps, 1))
    return max(horizon_vol, 1e-4), granularity


def _prediction_range_and_risk(direction: str, confidence: float, horizon_vol: float) -> tuple[float, float, float, str]:
    sign = 1 if direction == "up" else -1
    center_pct = sign * (0.4 + max(0.0, confidence - 0.5) * 8.0)
    band_pct = max(0.8, min(30.0, horizon_vol * 100 * 1.8 + 0.35))
    min_pct = center_pct - band_pct
    max_pct = center_pct + band_pct
    if min_pct > max_pct:
        min_pct, max_pct = max_pct, min_pct

    risk_score = min(100.0, max(0.0, horizon_vol * 100 * 28.0))
    if risk_score < 33:
        risk_level = "low"
    elif risk_score < 66:
        risk_level = "medium"
    else:
        risk_level = "high"
    return round(min_pct, 2), round(max_pct, 2), round(risk_score, 2), risk_level


def generate_prediction(payload: PredictionRequest) -> PredictionResponse:
    """
    Temporary baseline inference stub.
    Replace this logic with real model inference once the training pipeline is integrated.
    """
    settings = get_settings()
    direction, confidence, predicted_close = _deterministic_signal(f"{payload.symbol}:{payload.horizon}")
    horizon_vol, vol_granularity = _realized_volatility(payload.symbol, payload.horizon)
    range_min_pct, range_max_pct, risk_score, risk_level = _prediction_range_and_risk(
        direction=direction,
        confidence=confidence,
        horizon_vol=horizon_vol,
    )

    debug = None
    if payload.include_debug:
        debug = {
            "inference_mode": "stub",
            "note": "Replace with model artifact inference in next milestone",
            "volatility_source_granularity": vol_granularity,
            "horizon_volatility": f"{horizon_vol:.6f}",
        }

    return PredictionResponse(
        symbol=payload.symbol,
        horizon=payload.horizon,
        direction=direction,
        confidence=confidence,
        predicted_close=predicted_close,
        return_range_min_pct=range_min_pct,
        return_range_max_pct=range_max_pct,
        risk_score=risk_score,
        risk_level=risk_level,
        model_version=settings.default_model_version,
        debug=debug,
    )
