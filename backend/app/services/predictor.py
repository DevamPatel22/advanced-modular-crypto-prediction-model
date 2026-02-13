from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import pstdev

import pandas as pd
from joblib import load

from app.config import get_settings
from app.ml.features import FEATURE_COLUMNS, HORIZON_SPECS, HorizonSpec, build_features
from app.ml.training import load_candles
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_registry import ModelRegistry

_MODEL_CACHE: dict[str, object] = {}
_CALIBRATION_CACHE: dict[str, dict[str, float | str]] = {}


def _deterministic_signal(seed: str) -> tuple[str, float, float]:
    score = sum(ord(ch) for ch in seed)
    direction = "up" if score % 2 == 0 else "down"
    confidence = 0.52 + (score % 33) / 100
    confidence = min(confidence, 0.89)
    predicted_close = 10.0 + float((score * 137) % 100000) / 10
    return direction, round(confidence, 4), round(predicted_close, 2)


def _find_horizon_spec(horizon: str) -> HorizonSpec | None:
    for spec in HORIZON_SPECS:
        if spec.label == horizon:
            return spec
    return None


def _load_cached_model(path: Path) -> object:
    key = str(path.resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load(path)
    return _MODEL_CACHE[key]


def _load_calibration(path: Path) -> dict[str, float | str]:
    key = str(path.resolve())
    if key not in _CALIBRATION_CACHE:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        _CALIBRATION_CACHE[key] = payload
    return _CALIBRATION_CACHE[key]


def _calibrate_confidence(raw_probability_up: float, calibration: dict[str, float | str]) -> float:
    scale = float(calibration.get("scale", 1.0))
    min_conf = float(calibration.get("min_confidence", 0.5))
    max_conf = float(calibration.get("max_confidence", 0.99))

    base = max(raw_probability_up, 1 - raw_probability_up)
    calibrated = base * scale
    return float(min(max_conf, max(min_conf, calibrated)))


def _load_recent_closes(symbol: str, granularity: str, limit: int = 500) -> list[float]:
    candles = load_candles(symbol.upper(), granularity)
    if candles.empty:
        return []
    closes = [float(value) for value in candles["close"].tail(limit).tolist()]
    return closes


def _realized_volatility(symbol: str, horizon: str) -> tuple[float, str]:
    spec = _find_horizon_spec(horizon)
    granularity, steps = ("1h", 1)
    if spec is not None:
        granularity, steps = spec.candidates[0]

    closes = _load_recent_closes(symbol, granularity, limit=800)
    if len(closes) < 40:
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


def _latest_features(symbol: str, spec: HorizonSpec) -> tuple[pd.DataFrame, float, str] | None:
    for granularity, _steps in spec.candidates:
        frame = load_candles(symbol.upper(), granularity)
        if frame.empty or len(frame) < 80:
            continue

        enriched = build_features(frame)
        ready = enriched.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        if ready.empty:
            continue

        latest = ready.iloc[[-1]]
        latest_close = float(latest["close"].iloc[0])
        return latest[FEATURE_COLUMNS], latest_close, granularity

    return None


def _predict_from_artifacts(payload: PredictionRequest) -> tuple[PredictionResponse | None, str | None]:
    registry = ModelRegistry()
    artifact_paths = registry.resolve_artifacts(payload.symbol, payload.horizon)
    if artifact_paths is None:
        return None, "symbol_horizon_not_promoted_or_missing_artifacts"

    spec = _find_horizon_spec(payload.horizon)
    if spec is None:
        return None, "unsupported_horizon"

    feature_row = _latest_features(payload.symbol, spec)
    if feature_row is None:
        return None, "insufficient_recent_data_for_feature_window"

    features, latest_close, feature_granularity = feature_row
    cls_model = _load_cached_model(artifact_paths["classification"])
    reg_model = _load_cached_model(artifact_paths["regression"])
    calibration = _load_calibration(artifact_paths["calibration"])

    raw_prob_up = float(cls_model.predict_proba(features)[0][1])
    pred_close = float(reg_model.predict(features)[0])
    direction = "up" if raw_prob_up >= 0.5 else "down"
    confidence = round(_calibrate_confidence(raw_prob_up, calibration), 4)

    horizon_vol, vol_granularity = _realized_volatility(payload.symbol, payload.horizon)
    range_min_pct, range_max_pct, risk_score, risk_level = _prediction_range_and_risk(
        direction=direction,
        confidence=confidence,
        horizon_vol=horizon_vol,
    )

    model_version = registry.get_active_model_version()
    debug = None
    if payload.include_debug:
        debug = {
            "inference_mode": "model",
            "raw_probability_up": f"{raw_prob_up:.6f}",
            "feature_granularity": feature_granularity,
            "latest_close": f"{latest_close:.6f}",
            "volatility_source_granularity": vol_granularity,
            "horizon_volatility": f"{horizon_vol:.6f}",
            "artifact_metrics": str(artifact_paths["metrics"]),
        }

    return (
        PredictionResponse(
            symbol=payload.symbol,
            horizon=payload.horizon,
            direction=direction,
            confidence=confidence,
            predicted_close=round(max(pred_close, 1e-8), 8),
            return_range_min_pct=range_min_pct,
            return_range_max_pct=range_max_pct,
            risk_score=risk_score,
            risk_level=risk_level,
            model_version=model_version,
            debug=debug,
        ),
        None,
    )


def generate_prediction(payload: PredictionRequest) -> PredictionResponse:
    settings = get_settings()
    model_response, fallback_reason = _predict_from_artifacts(payload)
    if model_response is not None:
        return model_response

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
            "inference_mode": "fallback",
            "fallback_reason": fallback_reason or "model_inference_failed",
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
        model_version=ModelRegistry().get_active_model_version() or settings.default_model_version,
        debug=debug,
    )
