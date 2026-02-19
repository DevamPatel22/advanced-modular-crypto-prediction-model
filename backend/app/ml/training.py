from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.ml.features import FEATURE_COLUMNS, FEATURE_VERSION, HORIZON_SPECS, HorizonSpec, build_features

MIN_SAMPLES = 320
LOOKBACK_BUFFER = 50
HORIZON_MIN_SAMPLES: dict[str, int] = {
    "5m": 300,
    "1h": 280,
    "6h": 260,
    "12h": 240,
    "1d": 220,
    "1w": 180,
    "1mo": 140,
    "3mo": 120,
}
HORIZON_LOG_RETURN_CLIP: dict[str, float] = {
    "5m": 0.08,
    "1h": 0.18,
    "6h": 0.35,
    "12h": 0.50,
    "1d": 0.75,
    "1w": 1.25,
    "1mo": 1.75,
    "3mo": 2.50,
}


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train_cls: pd.Series
    y_val_cls: pd.Series
    y_test_cls: np.ndarray
    y_train_reg: pd.Series
    y_val_reg: pd.Series
    y_test_reg: np.ndarray
    current_close_val: np.ndarray
    current_close_test: np.ndarray
    target_close_val: np.ndarray
    target_close_test: np.ndarray


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    model: Pipeline
    val_score: float


def _triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sigma: np.ndarray,
    steps_ahead: int,
    sigma_mult: float,
) -> np.ndarray:
    labels = np.full(close.shape[0], np.nan, dtype=float)
    if steps_ahead <= 0:
        return labels

    for idx in range(0, close.shape[0] - steps_ahead):
        current_close = float(close[idx])
        if current_close <= 0:
            continue

        local_sigma = float(sigma[idx]) if np.isfinite(sigma[idx]) else 0.0
        # Keep barriers non-zero even during very low-volatility windows.
        barrier_pct = max(0.001, abs(local_sigma) * sigma_mult)
        upper = current_close * (1.0 + barrier_pct)
        lower = current_close * (1.0 - barrier_pct)

        decided: float | None = None
        for forward in range(idx + 1, idx + steps_ahead + 1):
            up_touch = bool(high[forward] >= upper)
            down_touch = bool(low[forward] <= lower)
            if up_touch and down_touch:
                decided = 1.0 if close[forward] >= current_close else 0.0
                break
            if up_touch:
                decided = 1.0
                break
            if down_touch:
                decided = 0.0
                break

        if decided is None:
            terminal_close = float(close[idx + steps_ahead])
            decided = 1.0 if terminal_close > current_close else 0.0

        labels[idx] = decided

    return labels


def _database_path() -> Path:
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path())
    connection.row_factory = sqlite3.Row
    return connection


def load_candles(symbol: str, granularity: str) -> pd.DataFrame:
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT start_time, open, high, low, close, volume
                FROM candles
                WHERE symbol = ? AND granularity = ?
                ORDER BY start_time ASC
                """,
                (symbol.upper(), granularity),
            ).fetchall()
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["start_time", "open", "high", "low", "close", "volume"])

    if not rows:
        return pd.DataFrame(columns=["start_time", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame([dict(row) for row in rows])
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return frame


def list_symbols_with_any_candles(min_rows: int = 100) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT symbol
            FROM candles
            GROUP BY symbol
            HAVING COUNT(*) >= ?
            ORDER BY symbol ASC
            """,
            (min_rows,),
        ).fetchall()
    return [str(row["symbol"]) for row in rows]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-12, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _split_indices(length: int) -> tuple[int, int]:
    train_end = int(length * 0.7)
    val_end = int(length * 0.85)
    return train_end, val_end


def min_samples_for_horizon(horizon: str) -> int:
    return int(HORIZON_MIN_SAMPLES.get(horizon, MIN_SAMPLES))


def clip_log_return_predictions(values: np.ndarray, horizon: str) -> np.ndarray:
    clip_value = float(HORIZON_LOG_RETURN_CLIP.get(horizon, 1.0))
    return np.clip(values, -clip_value, clip_value)


def resolve_horizon_data(symbol: str, spec: HorizonSpec, min_samples: int | None = None) -> tuple[str, int, pd.DataFrame] | None:
    required = min_samples if min_samples is not None else min_samples_for_horizon(spec.label)
    for granularity, steps in spec.candidates:
        candidate = load_candles(symbol, granularity)
        if len(candidate) >= (required + steps + LOOKBACK_BUFFER):
            return granularity, steps, candidate
    return None


def _prepare_supervised(df: pd.DataFrame, steps_ahead: int, min_samples: int) -> tuple[pd.DataFrame, SplitData] | None:
    settings = get_settings()
    enriched = build_features(df)
    enriched["target_close"] = enriched["close"].shift(-steps_ahead)
    enriched["target_log_return"] = np.log((enriched["target_close"] + 1e-12) / (enriched["close"] + 1e-12))
    if settings.classification_label_mode.strip().lower() == "triple_barrier":
        labels = _triple_barrier_labels(
            close=enriched["close"].to_numpy(dtype=float),
            high=enriched["high"].to_numpy(dtype=float),
            low=enriched["low"].to_numpy(dtype=float),
            sigma=enriched["sigma_20"].to_numpy(dtype=float),
            steps_ahead=steps_ahead,
            sigma_mult=float(settings.triple_barrier_sigma_mult),
        )
        enriched["target_up"] = labels
    else:
        enriched["target_up"] = (enriched["target_close"] > enriched["close"]).astype(int)
    enriched = enriched[(enriched["close"] > 0) & (enriched["target_close"] > 0)]
    enriched = enriched.dropna(subset=FEATURE_COLUMNS + ["target_close", "target_log_return", "target_up"]).reset_index(drop=True)
    enriched["target_up"] = enriched["target_up"].astype(int)

    if len(enriched) < min_samples:
        return None

    train_end, val_end = _split_indices(len(enriched))
    train_df = enriched.iloc[:train_end]
    val_df = enriched.iloc[train_end:val_end]
    test_df = enriched.iloc[val_end:]

    if len(test_df) < 30 or train_df["target_up"].nunique() < 2:
        return None

    split = SplitData(
        x_train=train_df[FEATURE_COLUMNS],
        x_val=val_df[FEATURE_COLUMNS],
        x_test=test_df[FEATURE_COLUMNS],
        y_train_cls=train_df["target_up"],
        y_val_cls=val_df["target_up"],
        y_test_cls=test_df["target_up"].to_numpy(dtype=float),
        y_train_reg=train_df["target_log_return"],
        y_val_reg=val_df["target_log_return"],
        y_test_reg=test_df["target_log_return"].to_numpy(dtype=float),
        current_close_val=val_df["close"].to_numpy(dtype=float),
        current_close_test=test_df["close"].to_numpy(dtype=float),
        target_close_val=val_df["target_close"].to_numpy(dtype=float),
        target_close_test=test_df["target_close"].to_numpy(dtype=float),
    )
    return enriched, split


def _classification_candidates() -> list[tuple[str, Pipeline]]:
    return [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
                ]
            ),
        ),
        (
            "gradient_boosting_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=180,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _regression_candidates() -> list[tuple[str, Pipeline]]:
    return [
        (
            "random_forest_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=250,
                            max_depth=12,
                            min_samples_leaf=4,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "gradient_boosting_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hist_gradient_boosting_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=350,
                            learning_rate=0.04,
                            max_depth=6,
                            min_samples_leaf=8,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=350,
                            max_depth=14,
                            min_samples_leaf=3,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _select_best_classifier(split: SplitData) -> CandidateResult:
    best: CandidateResult | None = None
    for model_name, model in _classification_candidates():
        model.fit(split.x_train, split.y_train_cls)
        val_pred = model.predict(split.x_val)
        score = float(f1_score(split.y_val_cls, val_pred, zero_division=0))
        if best is None or score > best.val_score:
            best = CandidateResult(model_name=model_name, model=model, val_score=score)
    if best is None:
        raise RuntimeError("No classification candidate available")
    return best


def _select_best_regressor(split: SplitData, horizon: str) -> CandidateResult:
    best: CandidateResult | None = None
    for model_name, model in _regression_candidates():
        model.fit(split.x_train, split.y_train_reg)
        val_pred_log_return = model.predict(split.x_val)
        val_pred_log_return = clip_log_return_predictions(val_pred_log_return, horizon)
        val_pred_close = split.current_close_val * np.exp(val_pred_log_return)
        score = -float(math.sqrt(mean_squared_error(split.target_close_val, val_pred_close)))
        if best is None or score > best.val_score:
            best = CandidateResult(model_name=model_name, model=model, val_score=score)
    if best is None:
        raise RuntimeError("No regression candidate available")
    return best


def _build_calibration_payload(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    avg_confidence = float(np.mean(np.maximum(clipped, 1 - clipped)))
    avg_accuracy = float(np.mean((clipped >= 0.5) == (y_true >= 0.5)))
    scale = avg_accuracy / max(avg_confidence, 1e-6)
    scale = float(min(1.3, max(0.7, scale)))
    return {
        "method": "linear_scale",
        "scale": scale,
        "min_confidence": 0.50,
        "max_confidence": 0.99,
    }


def calibrate_confidence(raw_probability_up: float, calibration: dict[str, float | str]) -> float:
    scale = float(calibration.get("scale", 1.0))
    min_conf = float(calibration.get("min_confidence", 0.5))
    max_conf = float(calibration.get("max_confidence", 0.99))

    raw = max(raw_probability_up, 1 - raw_probability_up)
    calibrated = min(max_conf, max(min_conf, raw * scale))
    return float(calibrated)


def _metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_proba_cls: np.ndarray, y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "precision": float(precision_score(y_true_cls, y_pred_cls, zero_division=0)),
        "recall": float(recall_score(y_true_cls, y_pred_cls, zero_division=0)),
        "f1": float(f1_score(y_true_cls, y_pred_cls, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_cls, y_proba_cls)) if len(np.unique(y_true_cls)) > 1 else math.nan,
        "mae": float(mean_absolute_error(y_true_reg, y_pred_reg)),
        "rmse": float(math.sqrt(mean_squared_error(y_true_reg, y_pred_reg))),
        "mape": _mape(y_true_reg, y_pred_reg),
    }


def _confidence_slice_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_proba_cls: np.ndarray, threshold: float) -> dict[str, float]:
    confidence = np.maximum(y_proba_cls, 1 - y_proba_cls)
    mask = confidence >= threshold
    selected = int(np.sum(mask))
    total = int(len(y_true_cls))
    coverage = float(selected / total) if total > 0 else 0.0
    if selected == 0:
        return {
            "threshold": float(threshold),
            "coverage": coverage,
            "selected_samples": 0.0,
            "precision": math.nan,
            "accuracy": math.nan,
        }
    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "selected_samples": float(selected),
        "precision": float(precision_score(y_true_cls[mask], y_pred_cls[mask], zero_division=0)),
        "accuracy": float(accuracy_score(y_true_cls[mask], y_pred_cls[mask])),
    }


def _regime_metrics(split: SplitData, y_pred_cls: np.ndarray) -> dict[str, dict[str, float]]:
    if split.x_test.empty:
        return {}

    probs = split.x_test[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float)
    if probs.size == 0:
        return {}
    regime_ids = np.argmax(probs, axis=1)
    regime_name = {0: "down", 1: "flat", 2: "up"}

    out: dict[str, dict[str, float]] = {}
    for regime_idx in [0, 1, 2]:
        mask = regime_ids == regime_idx
        count = int(np.sum(mask))
        if count == 0:
            continue
        out[regime_name[regime_idx]] = {
            "count": float(count),
            "accuracy": float(accuracy_score(split.y_test_cls[mask], y_pred_cls[mask])),
            "f1": float(f1_score(split.y_test_cls[mask], y_pred_cls[mask], zero_division=0)),
        }
    return out


def _baseline_metrics(y_true_cls: np.ndarray, y_true_reg: np.ndarray, current_close: np.ndarray) -> dict[str, float]:
    cls_baseline = np.ones_like(y_true_cls)
    reg_baseline = current_close
    return {
        "accuracy": float(accuracy_score(y_true_cls, cls_baseline)),
        "f1": float(f1_score(y_true_cls, cls_baseline, zero_division=0)),
        "rmse": float(math.sqrt(mean_squared_error(y_true_reg, reg_baseline))),
        "mae": float(mean_absolute_error(y_true_reg, reg_baseline)),
        "mape": _mape(y_true_reg, reg_baseline),
    }


def _martingale_residual_diagnostic(y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> dict[str, float | bool]:
    residuals = y_true_reg - y_pred_reg
    if len(residuals) < 30:
        return {
            "acf1": math.nan,
            "variance": float(np.var(residuals)) if len(residuals) else math.nan,
            "pass": False,
        }

    lagged = residuals[:-1]
    lead = residuals[1:]
    lag_std = float(np.std(lagged))
    lead_std = float(np.std(lead))
    if lag_std <= 1e-12 or lead_std <= 1e-12:
        acf1 = 0.0
    else:
        acf1 = float(np.corrcoef(lagged, lead)[0, 1])
    acf1 = float(np.clip(acf1, -1.0, 1.0))
    return {
        "acf1": acf1,
        "variance": float(np.var(residuals)),
        "pass": abs(acf1) <= 0.10,
    }


def evaluate_symbol_horizon(symbol: str, spec: HorizonSpec, model_version: str, models_root: Path, write_artifacts: bool = True) -> dict[str, object]:
    min_samples = min_samples_for_horizon(spec.label)
    resolved = resolve_horizon_data(symbol, spec, min_samples=min_samples)
    if resolved is None:
        return {
            "symbol": symbol,
            "horizon": spec.label,
            "status": "insufficient_data",
            "reason": "no_granularity_meets_minimum",
            "required_min_samples": min_samples,
        }

    granularity, steps, frame = resolved
    prepared = _prepare_supervised(frame, steps, min_samples=min_samples)
    if prepared is None:
        return {
            "symbol": symbol,
            "horizon": spec.label,
            "status": "insufficient_data",
            "reason": "insufficient_rows_after_feature_pipeline",
            "granularity": granularity,
            "steps_ahead": steps,
            "required_min_samples": min_samples,
        }

    enriched, split = prepared

    cls_best = _select_best_classifier(split)
    reg_best = _select_best_regressor(split, spec.label)

    cls_pred = cls_best.model.predict(split.x_test)
    cls_prob = cls_best.model.predict_proba(split.x_test)[:, 1]
    reg_pred_log_return = reg_best.model.predict(split.x_test)
    reg_pred_log_return = clip_log_return_predictions(reg_pred_log_return, spec.label)
    reg_pred = split.current_close_test * np.exp(reg_pred_log_return)

    metrics = _metrics(
        y_true_cls=split.y_test_cls,
        y_pred_cls=cls_pred,
        y_proba_cls=cls_prob,
        y_true_reg=split.target_close_test,
        y_pred_reg=reg_pred,
    )
    settings = get_settings()
    confidence_slice = _confidence_slice_metrics(
        y_true_cls=split.y_test_cls,
        y_pred_cls=cls_pred,
        y_proba_cls=cls_prob,
        threshold=float(settings.high_confidence_threshold),
    )
    regime_breakdown = _regime_metrics(split=split, y_pred_cls=cls_pred)
    baseline = _baseline_metrics(
        y_true_cls=split.y_test_cls,
        y_true_reg=split.target_close_test,
        current_close=split.current_close_test,
    )
    martingale_diag = _martingale_residual_diagnostic(split.target_close_test, reg_pred)
    martingale_enforced = settings.martingale_gate_mode.strip().lower() == "strict"

    promotion_pass = (
        metrics["f1"] > baseline["f1"]
        and metrics["accuracy"] > baseline["accuracy"]
        and metrics["rmse"] < baseline["rmse"]
        and (bool(martingale_diag["pass"]) if martingale_enforced else True)
    )

    failed_reasons: list[str] = []
    if not (metrics["f1"] > baseline["f1"]):
        failed_reasons.append("classification_f1_not_above_baseline")
    if not (metrics["accuracy"] > baseline["accuracy"]):
        failed_reasons.append("classification_accuracy_not_above_baseline")
    if not (metrics["rmse"] < baseline["rmse"]):
        failed_reasons.append("regression_rmse_not_below_baseline")
    if martingale_enforced and (not bool(martingale_diag["pass"])):
        failed_reasons.append("martingale_residual_autocorrelation_too_high")

    calibration = _build_calibration_payload(split.y_test_cls, cls_prob)

    symbol_dir = models_root / model_version / symbol
    artifact_paths = {
        "classification": symbol_dir / f"cls_{spec.label}.joblib",
        "regression": symbol_dir / f"reg_{spec.label}.joblib",
        "calibration": symbol_dir / f"calibration_{spec.label}.json",
        "metrics": symbol_dir / f"metrics_{spec.label}.json",
    }

    metrics_payload: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "symbol": symbol,
        "horizon": spec.label,
        "status": "ok",
        "model_version": model_version,
        "feature_version": FEATURE_VERSION,
        "granularity": granularity,
        "steps_ahead": steps,
        "rows_total": int(len(enriched)),
        "rows_test": int(len(split.x_test)),
        "selected_models": {
            "classifier": cls_best.model_name,
            "regressor": reg_best.model_name,
        },
        "metrics": metrics,
        "baseline": baseline,
        "martingale_diagnostic": martingale_diag,
        "martingale_enforced": martingale_enforced,
        "regression_target": "log_return",
        "regression_output_transform": "predicted_close = current_close * exp(clipped_log_return)",
        "classification_label_mode": settings.classification_label_mode,
        "triple_barrier_sigma_mult": float(settings.triple_barrier_sigma_mult),
        "log_return_clip": float(HORIZON_LOG_RETURN_CLIP.get(spec.label, 1.0)),
        "confidence_slice": confidence_slice,
        "regime_breakdown": regime_breakdown,
        "promotion_gate": {
            "passed": promotion_pass,
            "failed_reasons": failed_reasons,
            "rules": {
                "classification": ["f1 > baseline.f1", "accuracy > baseline.accuracy"],
                "regression": ["rmse < baseline.rmse"],
                "stochastic": ["abs(residual_acf1) <= 0.10 (strict mode only)"],
            },
        },
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
    }

    if write_artifacts:
        symbol_dir.mkdir(parents=True, exist_ok=True)
        dump(cls_best.model, artifact_paths["classification"])
        dump(reg_best.model, artifact_paths["regression"])
        artifact_paths["calibration"].write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        artifact_paths["metrics"].write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metrics_payload["calibration"] = calibration
    return metrics_payload


def training_universe_snapshot(symbols: list[str]) -> dict[str, object]:
    return {
        "count": len(symbols),
        "symbols": sorted(symbols),
    }


def model_manifest_payload(model_version: str, universe: list[str], symbol_horizon_results: dict[str, dict[str, object]]) -> dict[str, object]:
    earliest = None
    latest = None

    for symbol in universe:
        for entry in symbol_horizon_results.get(symbol, {}).values():
            if entry.get("status") != "ok":
                continue
            horizon_min = entry.get("train_start_time")
            horizon_max = entry.get("train_end_time")
            if isinstance(horizon_min, (int, float)):
                earliest = horizon_min if earliest is None else min(earliest, horizon_min)
            if isinstance(horizon_max, (int, float)):
                latest = horizon_max if latest is None else max(latest, horizon_max)

    return {
        "model_version": model_version,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "feature_version": FEATURE_VERSION,
        "universe": training_universe_snapshot(universe),
        "data_range": {
            "min_start_time": earliest,
            "max_start_time": latest,
        },
    }


def all_horizon_specs() -> list[HorizonSpec]:
    return list(HORIZON_SPECS)


def as_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=False)


def as_plain_dict(obj: object) -> dict[str, object]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)  # type: ignore[arg-type]
    raise TypeError("Object is not a dataclass instance")
