"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import pstdev

import numpy as np
import pandas as pd
from joblib import load

from app.config import get_settings
from app.ml.features import FEATURE_COLUMNS, HORIZON_SPECS, HorizonSpec, build_features, feature_columns_for_horizon
from app.ml.training import apply_directional_tilt_to_close, load_candles, regression_output_to_close
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.model_registry import ModelRegistry

_MODEL_CACHE: dict[str, object] = {}
_CALIBRATION_CACHE: dict[str, dict[str, float | str]] = {}
_METRICS_CACHE: dict[str, dict[str, object]] = {}


def _deterministic_signal(seed: str) -> tuple[str, float, float]:
    score = sum(ord(ch) for ch in seed)
    direction = "up" if score % 2 == 0 else "down"
    confidence = 0.52 + (score % 33) / 100
    confidence = min(confidence, 0.89)
    predicted_close = 10.0 + float((score * 137) % 100000) / 10
    return direction, round(confidence, 4), round(predicted_close, 2)


def _fallback_reference_close(symbol: str, horizon: str) -> float | None:
    spec = _find_horizon_spec(horizon)
    granularity = "1h"
    if spec is not None and spec.candidates:
        granularity = spec.candidates[0][0]
    closes = _load_recent_closes(symbol, granularity, limit=8)
    if not closes:
        return None
    latest = float(closes[-1])
    if latest <= 0:
        return None
    return latest


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


def _load_metrics(path: Path) -> dict[str, object]:
    key = str(path.resolve())
    if key not in _METRICS_CACHE:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        _METRICS_CACHE[key] = payload
    return _METRICS_CACHE[key]


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


def _market_bias(direction: str) -> str:
    return "bullish" if direction == "up" else "bearish"


def _horizon_seconds(horizon: str) -> int:
    label = horizon.strip().lower()
    if label.endswith("mo"):
        return int(label[:-2]) * 30 * 24 * 60 * 60
    unit_seconds = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
    unit = label[-1:] if label else ""
    if unit in unit_seconds and label[:-1].isdigit():
        return int(label[:-1]) * unit_seconds[unit]
    return 3600


def _price_bounds_from_return_range(current_price: float, min_pct: float, max_pct: float) -> tuple[float, float]:
    low = max(current_price * (1.0 + (min_pct / 100.0)), 1e-8)
    high = max(current_price * (1.0 + (max_pct / 100.0)), 1e-8)
    return (low, high) if low <= high else (high, low)


def _latest_features(symbol: str, spec: HorizonSpec, selected_features: list[str] | None = None) -> tuple[pd.DataFrame, float, str, int] | None:
    active_features = selected_features or feature_columns_for_horizon(spec.label)
    for granularity, _steps in spec.candidates:
        frame = load_candles(symbol.upper(), granularity)
        if frame.empty or len(frame) < 80:
            continue

        enriched = build_features(frame)
        ready = enriched.dropna(subset=active_features).reset_index(drop=True)
        if ready.empty:
            continue

        latest = ready.iloc[[-1]]
        latest_close = float(latest["close"].iloc[0])
        regime = int(np.argmax(latest[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float), axis=1)[0])
        return latest[active_features], latest_close, granularity, regime

    return None


def _predict_from_artifacts(payload: PredictionRequest) -> tuple[PredictionResponse | None, str | None]:
    registry = ModelRegistry()
    artifact_paths = registry.resolve_artifacts(payload.symbol, payload.horizon)
    if artifact_paths is None:
        return None, "symbol_horizon_not_promoted_or_missing_artifacts"

    spec = _find_horizon_spec(payload.horizon)
    if spec is None:
        return None, "unsupported_horizon"

    metrics_meta = _load_metrics(artifact_paths["metrics"])
    selected_features = metrics_meta.get("feature_columns")
    if not isinstance(selected_features, list) or not selected_features:
        selected_features = FEATURE_COLUMNS
    feature_row = _latest_features(payload.symbol, spec, selected_features=[str(item) for item in selected_features])
    if feature_row is None:
        return None, "insufficient_recent_data_for_feature_window"

    features, latest_close, feature_granularity, regime_id = feature_row
    cls_model = _load_cached_model(artifact_paths["classification"])
    reg_model = _load_cached_model(artifact_paths["regression"])
    calibration = _load_calibration(artifact_paths["calibration"])
    decision_threshold = float(calibration.get("decision_threshold", 0.5))

    raw_prob_up = float(cls_model.predict_proba(features)[0][1])
    reg_raw = float(reg_model.predict(features)[0])
    regime_name = {0: "down", 1: "flat", 2: "up"}.get(regime_id, "flat")
    regime_paths = metrics_meta.get("regime_artifact_paths")
    if isinstance(regime_paths, dict):
        cls_paths = regime_paths.get("classification")
        reg_paths = regime_paths.get("regression")
        if isinstance(cls_paths, dict):
            cls_regime_path = cls_paths.get(regime_name)
            if isinstance(cls_regime_path, str) and Path(cls_regime_path).exists():
                regime_cls_model = _load_cached_model(Path(cls_regime_path))
                raw_prob_up = float(regime_cls_model.predict_proba(features)[0][1])
        if isinstance(reg_paths, dict):
            reg_regime_path = reg_paths.get(regime_name)
            if isinstance(reg_regime_path, str) and Path(reg_regime_path).exists():
                regime_reg_model = _load_cached_model(Path(reg_regime_path))
                reg_raw = float(regime_reg_model.predict(features)[0])
    regression_target = str(metrics_meta.get("regression_target", "price_close"))
    model_pred_close = float(
        regression_output_to_close(
            raw_prediction=np.array([reg_raw], dtype=float),
            current_close=np.array([latest_close], dtype=float),
            horizon=payload.horizon,
            regression_target=regression_target,
        )[0]
    )
    regression_blend_alpha = float(metrics_meta.get("regression_blend_alpha", 1.0))
    regression_blend_alpha = float(min(1.0, max(0.0, regression_blend_alpha)))
    pred_close = (regression_blend_alpha * model_pred_close) + ((1.0 - regression_blend_alpha) * latest_close)
    directional_tilt_gamma = float(metrics_meta.get("directional_tilt_gamma", 0.0))
    feature_volatility_20 = float(features["volatility_20"].iloc[0]) if "volatility_20" in features.columns else 0.0
    pred_close = float(
        apply_directional_tilt_to_close(
            predicted_close=np.array([pred_close], dtype=float),
            current_close=np.array([latest_close], dtype=float),
            probability_up=np.array([raw_prob_up], dtype=float),
            volatility_feature=np.array([feature_volatility_20], dtype=float),
            horizon=payload.horizon,
            gamma=directional_tilt_gamma,
        )[0]
    )
    direction = "up" if raw_prob_up >= decision_threshold else "down"
    confidence = round(_calibrate_confidence(raw_prob_up, calibration), 4)
    settings = get_settings()
    if settings.prediction_abstain_to_fallback and confidence < float(settings.prediction_confidence_min_for_model):
        return None, "model_confidence_below_threshold"

    horizon_vol, vol_granularity = _realized_volatility(payload.symbol, payload.horizon)
    range_min_pct, range_max_pct, risk_score, risk_level = _prediction_range_and_risk(
        direction=direction,
        confidence=confidence,
        horizon_vol=horizon_vol,
    )
    low_usd, high_usd = _price_bounds_from_return_range(latest_close, range_min_pct, range_max_pct)

    model_version = registry.get_active_model_version()
    debug = None
    if payload.include_debug:
        debug = {
            "inference_mode": "model",
            "raw_probability_up": f"{raw_prob_up:.6f}",
            "feature_granularity": feature_granularity,
            "latest_close": f"{latest_close:.6f}",
            "regime_routing": regime_name,
            "decision_threshold": f"{decision_threshold:.4f}",
            "regression_target": regression_target,
            "regression_blend_alpha": f"{regression_blend_alpha:.4f}",
            "directional_tilt_gamma": f"{directional_tilt_gamma:.4f}",
            "volatility_source_granularity": vol_granularity,
            "horizon_volatility": f"{horizon_vol:.6f}",
            "model_confidence_threshold": f"{float(settings.prediction_confidence_min_for_model):.4f}",
            "artifact_metrics": str(artifact_paths["metrics"]),
        }

    return (
        PredictionResponse(
            symbol=payload.symbol,
            horizon=payload.horizon,
            direction=direction,
            market_bias=_market_bias(direction),
            confidence=confidence,
            current_price=round(max(latest_close, 1e-8), 8),
            predicted_close=round(max(pred_close, 1e-8), 8),
            predicted_low_usd=round(low_usd, 8),
            predicted_high_usd=round(high_usd, 8),
            return_range_min_pct=range_min_pct,
            return_range_max_pct=range_max_pct,
            risk_score=risk_score,
            risk_level=risk_level,
            horizon_end_at=datetime.now(UTC) + timedelta(seconds=_horizon_seconds(payload.horizon)),
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

    direction, confidence, fallback_stub_close = _deterministic_signal(f"{payload.symbol}:{payload.horizon}")
    horizon_vol, vol_granularity = _realized_volatility(payload.symbol, payload.horizon)
    range_min_pct, range_max_pct, risk_score, risk_level = _prediction_range_and_risk(
        direction=direction,
        confidence=confidence,
        horizon_vol=horizon_vol,
    )
    reference_close = _fallback_reference_close(payload.symbol, payload.horizon)
    implied_return_pct = (range_min_pct + range_max_pct) / 2.0
    if reference_close is not None:
        predicted_close = max(reference_close * (1.0 + implied_return_pct / 100.0), 1e-8)
    else:
        predicted_close = max(fallback_stub_close, 1e-8)
    current_price = reference_close if reference_close is not None else predicted_close
    low_usd, high_usd = _price_bounds_from_return_range(current_price, range_min_pct, range_max_pct)

    debug = None
    if payload.include_debug:
        debug = {
            "inference_mode": "fallback",
            "fallback_reason": fallback_reason or "model_inference_failed",
            "volatility_source_granularity": vol_granularity,
            "horizon_volatility": f"{horizon_vol:.6f}",
            "fallback_reference_close": f"{reference_close:.8f}" if reference_close is not None else "unavailable",
            "fallback_implied_return_pct": f"{implied_return_pct:.4f}",
        }

    return PredictionResponse(
        symbol=payload.symbol,
        horizon=payload.horizon,
        direction=direction,
        market_bias=_market_bias(direction),
        confidence=confidence,
        current_price=round(max(current_price, 1e-8), 8),
        predicted_close=round(predicted_close, 8),
        predicted_low_usd=round(low_usd, 8),
        predicted_high_usd=round(high_usd, 8),
        return_range_min_pct=range_min_pct,
        return_range_max_pct=range_max_pct,
        risk_score=risk_score,
        risk_level=risk_level,
        horizon_end_at=datetime.now(UTC) + timedelta(seconds=_horizon_seconds(payload.horizon)),
        model_version=ModelRegistry().get_active_model_version() or settings.default_model_version,
        debug=debug,
    )
