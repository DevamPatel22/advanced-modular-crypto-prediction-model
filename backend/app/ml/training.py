"""Copyright (c) 2026 Devam Patel. All rights reserved.
Proprietary software. Unauthorized copying, modification, distribution, or use is prohibited.
"""

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
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
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
from app.ml.features import FEATURE_COLUMNS, FEATURE_VERSION, HORIZON_SPECS, HorizonSpec, build_features, feature_columns_for_horizon

try:  # Optional free libraries if installed in runtime.
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]
    XGBRegressor = None  # type: ignore[assignment]

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]
    LGBMRegressor = None  # type: ignore[assignment]

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[assignment]
    CatBoostRegressor = None  # type: ignore[assignment]

MIN_SAMPLES = 320
LOOKBACK_BUFFER = 50
HORIZON_MIN_SAMPLES: dict[str, int] = {
    "5m": 300,
    "1h": 280,
    "3h": 270,
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
    "3h": 0.26,
    "6h": 0.35,
    "12h": 0.50,
    "1d": 0.75,
    "1w": 1.25,
    "1mo": 1.75,
    "3mo": 2.50,
}


@dataclass(frozen=True)
class SplitData:
    feature_columns: list[str]
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
    regime_train: np.ndarray
    regime_val: np.ndarray
    regime_test: np.ndarray


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    model: Pipeline
    val_score: float
    decision_threshold: float = 0.5
    regression_blend_alpha: float = 1.0


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


def _prepare_supervised(df: pd.DataFrame, horizon: str, steps_ahead: int, min_samples: int) -> tuple[pd.DataFrame, SplitData] | None:
    settings = get_settings()
    selected_features = feature_columns_for_horizon(horizon)
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
    enriched = enriched.dropna(subset=selected_features + ["target_close", "target_log_return", "target_up"]).reset_index(drop=True)
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
        feature_columns=selected_features,
        x_train=train_df[selected_features],
        x_val=val_df[selected_features],
        x_test=test_df[selected_features],
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
        regime_train=np.argmax(train_df[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float), axis=1),
        regime_val=np.argmax(val_df[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float), axis=1),
        regime_test=np.argmax(test_df[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float), axis=1),
    )
    return enriched, split


def _classification_candidates() -> list[tuple[str, Pipeline]]:
    base: list[tuple[str, Pipeline]] = [
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
            "random_forest_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=320,
                            max_depth=10,
                            min_samples_leaf=4,
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=400,
                            max_depth=12,
                            min_samples_leaf=3,
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
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
    if XGBClassifier is not None:
        base.append(
            (
                "xgboost_classifier",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=220,
                                learning_rate=0.05,
                                max_depth=5,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                random_state=42,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            )
        )
    if LGBMClassifier is not None:
        base.append(
            (
                "lightgbm_classifier",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LGBMClassifier(
                                n_estimators=260,
                                learning_rate=0.04,
                                max_depth=7,
                                random_state=42,
                                n_jobs=1,
                                verbosity=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    if CatBoostClassifier is not None:
        base.append(
            (
                "catboost_classifier",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            CatBoostClassifier(
                                iterations=220,
                                learning_rate=0.05,
                                depth=6,
                                loss_function="Logloss",
                                verbose=False,
                                random_seed=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    base_estimators = [
        ("lr", base[0][1]),
        ("rf", base[1][1]),
        ("et", base[2][1]),
        ("gb", base[3][1]),
    ]
    stacking = Pipeline(
        steps=[
            (
                "model",
                StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
                    stack_method="predict_proba",
                    passthrough=False,
                    n_jobs=1,
                ),
            )
        ]
    )
    return base + [("stacking_classifier", stacking)]


def _regression_candidates() -> list[tuple[str, Pipeline]]:
    base: list[tuple[str, Pipeline]] = [
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
    if XGBRegressor is not None:
        base.append(
            (
                "xgboost_regressor",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=260,
                                learning_rate=0.04,
                                max_depth=6,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                objective="reg:squarederror",
                                random_state=42,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
            )
        )
    if LGBMRegressor is not None:
        base.append(
            (
                "lightgbm_regressor",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LGBMRegressor(
                                n_estimators=320,
                                learning_rate=0.035,
                                max_depth=8,
                                random_state=42,
                                n_jobs=1,
                                verbosity=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    if CatBoostRegressor is not None:
        base.append(
            (
                "catboost_regressor",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            CatBoostRegressor(
                                iterations=260,
                                learning_rate=0.04,
                                depth=7,
                                loss_function="RMSE",
                                verbose=False,
                                random_seed=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    base_estimators = [
        ("rf", base[0][1]),
        ("gbr", base[1][1]),
        ("hgb", base[2][1]),
        ("etr", base[3][1]),
    ]
    stacking = Pipeline(
        steps=[
            (
                "model",
                StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=GradientBoostingRegressor(
                        n_estimators=180,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42,
                    ),
                    passthrough=False,
                    n_jobs=1,
                ),
            )
        ]
    )
    return base + [("stacking_regressor", stacking)]


def _class_balance_sample_weight(y: pd.Series) -> np.ndarray:
    values = y.to_numpy(dtype=int)
    total = len(values)
    positive = max(int(np.sum(values == 1)), 1)
    negative = max(int(np.sum(values == 0)), 1)
    w_pos = total / (2.0 * positive)
    w_neg = total / (2.0 * negative)
    return np.where(values == 1, w_pos, w_neg).astype(float)


def _fit_pipeline_with_optional_weight(model: Pipeline, x: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None) -> None:
    if sample_weight is None:
        model.fit(x, y)
        return
    last_step = model.steps[-1][0]
    try:
        model.fit(x, y, **{f"{last_step}__sample_weight": sample_weight})
    except (TypeError, ValueError):
        model.fit(x, y)


def _best_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    baseline_pred = np.ones_like(y_true)
    baseline_f1 = float(f1_score(y_true, baseline_pred, zero_division=0))
    baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
    rows: list[tuple[float, float, float, float, float, float]] = []
    for threshold in np.linspace(0.30, 0.75, 46):
        y_pred = (y_prob >= threshold).astype(int)
        score_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        score_acc = float(accuracy_score(y_true, y_pred))
        delta_f1 = score_f1 - baseline_f1
        delta_acc = score_acc - baseline_accuracy
        score = min(delta_f1, delta_acc) + (0.35 * delta_f1) + (0.20 * delta_acc) + (0.01 * score_f1)
        rows.append((float(threshold), score_f1, score_acc, delta_f1, delta_acc, score))

    # First priority: clear both classification gate dimensions.
    both_pass = [row for row in rows if row[3] > 0 and row[4] > 0]
    if both_pass:
        best = max(both_pass, key=lambda row: (min(row[3], row[4]), row[3], row[4]))
        return best[0], best[1], best[2]

    # Second priority: keep accuracy at/above baseline while maximizing F1 lift.
    acc_ok = [row for row in rows if row[4] >= 0]
    if acc_ok:
        best = max(acc_ok, key=lambda row: (row[3], row[1], row[4], row[5]))
        return best[0], best[1], best[2]

    # Fallback to previous composite scoring when no gate-friendly threshold exists.
    best = max(rows, key=lambda row: (row[5], row[1], row[2]))
    return best[0], best[1], best[2]


def _select_best_classifier(split: SplitData) -> CandidateResult:
    best: CandidateResult | None = None
    sample_weight = _class_balance_sample_weight(split.y_train_cls)
    val_true = split.y_val_cls.to_numpy(dtype=int)
    val_baseline_pred = np.ones_like(val_true)
    val_baseline_f1 = float(f1_score(val_true, val_baseline_pred, zero_division=0))
    val_baseline_accuracy = float(accuracy_score(val_true, val_baseline_pred))
    for model_name, model in _classification_candidates():
        _fit_pipeline_with_optional_weight(model, split.x_train, split.y_train_cls, sample_weight)
        val_prob = model.predict_proba(split.x_val)[:, 1]
        threshold, score_f1, score_acc = _best_threshold_for_f1(val_true, val_prob)
        delta_f1 = score_f1 - val_baseline_f1
        delta_acc = score_acc - val_baseline_accuracy
        # Rank by gate-oriented validation margins rather than raw F1 alone.
        score = min(delta_f1, delta_acc) + (0.35 * delta_f1) + (0.20 * delta_acc)
        if best is None or score > best.val_score:
            best = CandidateResult(model_name=model_name, model=model, val_score=score, decision_threshold=threshold)
    if best is None:
        raise RuntimeError("No classification candidate available")
    return best


def _best_regression_blend_alpha(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> tuple[float, float]:
    best_alpha = 1.0
    best_rmse = math.inf
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 41),
                np.array([0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99]),
            ]
        )
    )
    for alpha in candidates:
        blended = (alpha * y_pred_model) + ((1.0 - alpha) * y_pred_baseline)
        rmse = float(math.sqrt(mean_squared_error(y_true, blended)))
        if rmse < best_rmse:
            best_alpha = float(alpha)
            best_rmse = rmse
    return best_alpha, best_rmse


def _select_best_regressor(split: SplitData, horizon: str) -> CandidateResult:
    best: CandidateResult | None = None
    for model_name, model in _regression_candidates():
        model.fit(split.x_train, split.y_train_reg)
        val_pred_log_return = model.predict(split.x_val)
        val_pred_log_return = clip_log_return_predictions(val_pred_log_return, horizon)
        val_pred_close_model = split.current_close_val * np.exp(val_pred_log_return)
        alpha, best_rmse = _best_regression_blend_alpha(
            y_true=split.target_close_val,
            y_pred_model=val_pred_close_model,
            y_pred_baseline=split.current_close_val,
        )
        score = -best_rmse
        if best is None or score > best.val_score:
            best = CandidateResult(model_name=model_name, model=model, val_score=score, regression_blend_alpha=alpha)
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


def _extract_top_feature_importance(model: Pipeline, feature_names: list[str], top_k: int = 8) -> list[dict[str, float | str]]:
    try:
        estimator = model.steps[-1][1]
        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            importances = np.abs(coef[0] if coef.ndim > 1 else coef)
        if importances is None or importances.size != len(feature_names):
            return []
        pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)[:top_k]
        return [{"feature": name, "importance": float(value)} for name, value in pairs]
    except Exception:
        return []


def _near_pass_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "f1_vs_baseline": float(metrics["f1"] - baseline["f1"]),
        "accuracy_vs_baseline": float(metrics["accuracy"] - baseline["accuracy"]),
        "rmse_vs_baseline": float(baseline["rmse"] - metrics["rmse"]),
    }


def _train_regime_models(split: SplitData, horizon: str) -> tuple[dict[str, Pipeline], dict[str, Pipeline]]:
    cls_models: dict[str, Pipeline] = {}
    reg_models: dict[str, Pipeline] = {}
    regime_names = {0: "down", 1: "flat", 2: "up"}
    for regime_id, regime_name in regime_names.items():
        idx = np.where(split.regime_train == regime_id)[0]
        if len(idx) < 120:
            continue
        x_sub = split.x_train.iloc[idx]
        y_cls_sub = split.y_train_cls.iloc[idx]
        y_reg_sub = split.y_train_reg.iloc[idx]
        if y_cls_sub.nunique() < 2:
            continue
        cls_best = _select_best_classifier(
            SplitData(
                feature_columns=split.feature_columns,
                x_train=x_sub,
                x_val=split.x_val,
                x_test=split.x_test,
                y_train_cls=y_cls_sub,
                y_val_cls=split.y_val_cls,
                y_test_cls=split.y_test_cls,
                y_train_reg=y_reg_sub,
                y_val_reg=split.y_val_reg,
                y_test_reg=split.y_test_reg,
                current_close_val=split.current_close_val,
                current_close_test=split.current_close_test,
                target_close_val=split.target_close_val,
                target_close_test=split.target_close_test,
                regime_train=split.regime_train,
                regime_val=split.regime_val,
                regime_test=split.regime_test,
            )
        )
        reg_best = _select_best_regressor(
            SplitData(
                feature_columns=split.feature_columns,
                x_train=x_sub,
                x_val=split.x_val,
                x_test=split.x_test,
                y_train_cls=y_cls_sub,
                y_val_cls=split.y_val_cls,
                y_test_cls=split.y_test_cls,
                y_train_reg=y_reg_sub,
                y_val_reg=split.y_val_reg,
                y_test_reg=split.y_test_reg,
                current_close_val=split.current_close_val,
                current_close_test=split.current_close_test,
                target_close_val=split.target_close_val,
                target_close_test=split.target_close_test,
                regime_train=split.regime_train,
                regime_val=split.regime_val,
                regime_test=split.regime_test,
            ),
            horizon,
        )
        cls_models[regime_name] = cls_best.model
        reg_models[regime_name] = reg_best.model
    return cls_models, reg_models


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
    prepared = _prepare_supervised(frame, spec.label, steps, min_samples=min_samples)
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
    settings = get_settings()
    regime_cls_models: dict[str, Pipeline] = {}
    regime_reg_models: dict[str, Pipeline] = {}
    if settings.regime_models_enabled:
        regime_cls_models, regime_reg_models = _train_regime_models(split, spec.label)

    cls_prob = cls_best.model.predict_proba(split.x_test)[:, 1]
    cls_pred = (cls_prob >= cls_best.decision_threshold).astype(int)
    reg_pred_log_return = reg_best.model.predict(split.x_test)
    reg_pred_log_return = clip_log_return_predictions(reg_pred_log_return, spec.label)
    reg_pred = split.current_close_test * np.exp(reg_pred_log_return)

    if settings.regime_models_enabled and (regime_cls_models or regime_reg_models):
        blended_prob = cls_prob.copy()
        blended_reg = reg_pred.copy()
        regime_names = {0: "down", 1: "flat", 2: "up"}
        for regime_id, regime_name in regime_names.items():
            mask = split.regime_test == regime_id
            if not np.any(mask):
                continue
            if regime_name in regime_cls_models:
                regime_prob = regime_cls_models[regime_name].predict_proba(split.x_test.iloc[mask])[:, 1]
                blended_prob[mask] = regime_prob
            if regime_name in regime_reg_models:
                regime_reg_log = regime_reg_models[regime_name].predict(split.x_test.iloc[mask])
                regime_reg_log = clip_log_return_predictions(np.asarray(regime_reg_log, dtype=float), spec.label)
                blended_reg[mask] = split.current_close_test[mask] * np.exp(regime_reg_log)
        cls_prob = blended_prob
        cls_pred = (cls_prob >= cls_best.decision_threshold).astype(int)
        reg_pred = blended_reg

    reg_blend_alpha = float(np.clip(reg_best.regression_blend_alpha, 0.0, 1.0))
    reg_pred = (reg_blend_alpha * reg_pred) + ((1.0 - reg_blend_alpha) * split.current_close_test)

    metrics = _metrics(
        y_true_cls=split.y_test_cls,
        y_pred_cls=cls_pred,
        y_proba_cls=cls_prob,
        y_true_reg=split.target_close_test,
        y_pred_reg=reg_pred,
    )
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
    calibration["decision_threshold"] = float(cls_best.decision_threshold)

    symbol_dir = models_root / model_version / symbol
    artifact_paths = {
        "classification": symbol_dir / f"cls_{spec.label}.joblib",
        "regression": symbol_dir / f"reg_{spec.label}.joblib",
        "calibration": symbol_dir / f"calibration_{spec.label}.json",
        "metrics": symbol_dir / f"metrics_{spec.label}.json",
    }
    regime_artifact_paths = {
        "classification": {
            name: symbol_dir / f"cls_{spec.label}_regime_{name}.joblib" for name in regime_cls_models
        },
        "regression": {
            name: symbol_dir / f"reg_{spec.label}_regime_{name}.joblib" for name in regime_reg_models
        },
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
            "regression_blend_alpha": reg_blend_alpha,
        },
        "feature_columns": split.feature_columns,
        "metrics": metrics,
        "baseline": baseline,
        "near_pass_delta": _near_pass_delta(metrics, baseline),
        "martingale_diagnostic": martingale_diag,
        "martingale_enforced": martingale_enforced,
        "regression_target": "log_return",
        "regression_output_transform": "predicted_close = current_close * exp(clipped_log_return)",
        "regression_blend_alpha": reg_blend_alpha,
        "classification_label_mode": settings.classification_label_mode,
        "triple_barrier_sigma_mult": float(settings.triple_barrier_sigma_mult),
        "regime_models_enabled": bool(settings.regime_models_enabled),
        "log_return_clip": float(HORIZON_LOG_RETURN_CLIP.get(spec.label, 1.0)),
        "confidence_slice": confidence_slice,
        "regime_breakdown": regime_breakdown,
        "top_features": {
            "classifier": _extract_top_feature_importance(cls_best.model, split.feature_columns),
            "regressor": _extract_top_feature_importance(reg_best.model, split.feature_columns),
        },
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
        "regime_artifact_paths": {
            "classification": {key: str(path) for key, path in regime_artifact_paths["classification"].items()},
            "regression": {key: str(path) for key, path in regime_artifact_paths["regression"].items()},
        },
    }

    if write_artifacts:
        symbol_dir.mkdir(parents=True, exist_ok=True)
        dump(cls_best.model, artifact_paths["classification"])
        dump(reg_best.model, artifact_paths["regression"])
        for name, model in regime_cls_models.items():
            dump(model, regime_artifact_paths["classification"][name])
        for name, model in regime_reg_models.items():
            dump(model, regime_artifact_paths["regression"][name])
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
