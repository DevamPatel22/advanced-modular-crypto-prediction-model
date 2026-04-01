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
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LogisticRegression, Ridge
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
from app.ml.evaluation import (
    bootstrap_metric_confidence_intervals,
    choose_threshold_walk_forward,
    execution_aware_metrics,
    execution_stress_metrics,
    paper_trading_metrics,
    strict_gate_walk_forward_diagnostic,
    walk_forward_splits,
)
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
DEFAULT_TRAIN_HORIZON_WINDOW_ROWS: dict[str, int] = {
    "5m": 12000,
    "1h": 12000,
    "3h": 11000,
    "6h": 9000,
    "12h": 8000,
    "1d": 7000,
    "1w": 5000,
    "1mo": 3500,
    "3mo": 2500,
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
RESIDUAL_TARGET_MIN_REL_IMPROVEMENT = 0.0025
HORIZON_DIRECTIONAL_TILT_MAX_LOG_SHIFT: dict[str, float] = {
    "5m": 0.035,
    "1h": 0.06,
    "3h": 0.08,
    "6h": 0.10,
    "12h": 0.12,
    "1d": 0.16,
    "1w": 0.24,
    "1mo": 0.32,
    "3mo": 0.45,
}
ALPHA_SIGNAL_DIRECTIONS: dict[str, float] = {
    # Positive value means "higher signal should align with higher future return".
    "reversal_pressure_5": 1.0,
    "mean_reversion_z_20": 1.0,
    "volatility_cluster_20_60": -1.0,
    "kalman_level_ratio": -1.0,
    "kalman_trend_5": 1.0,
}
ALPHA_SIGNAL_RATIONALES: dict[str, str] = {
    "reversal_pressure_5": "Short-horizon return shocks mean-revert under stable liquidity and bounded impact.",
    "mean_reversion_z_20": "Large z-score displacement from local fair value tends to compress.",
    "volatility_cluster_20_60": "Short-over-long volatility expansion flags unstable regimes with weaker carry.",
    "kalman_level_ratio": "Deviation from state-space smoothed level indicates potential reversion pressure.",
    "kalman_trend_5": "Persistent filtered trend can continue when flow remains one-sided.",
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
    current_close_train: np.ndarray
    current_close_val: np.ndarray
    current_close_test: np.ndarray
    target_close_train: np.ndarray
    target_close_val: np.ndarray
    target_close_test: np.ndarray
    regime_train: np.ndarray
    regime_val: np.ndarray
    regime_test: np.ndarray
    leakage_diagnostic: dict[str, object]
    alpha_kill_diagnostics: dict[str, object]


@dataclass(frozen=True)
class CandidateResult:
    model_name: str
    model: object
    val_score: float
    decision_threshold: float = 0.5
    regression_blend_alpha: float = 1.0
    class_down_weight_boost: float = 1.0
    regression_target: str = "log_return"
    threshold_tuning: dict[str, object] | None = None
    stability: dict[str, float] | None = None


@dataclass(frozen=True)
class ResolvedHorizonData:
    granularity: str
    steps_ahead: int
    frame: pd.DataFrame
    archive_rows_before_window: int
    archive_start_time: int | None
    archive_end_time: int | None
    horizon_window_rows_limit: int | None
    horizon_window_trim_applied: bool


@dataclass(frozen=True)
class ClassifierValidationCandidate:
    model_name: str
    model: object
    down_weight_boost: float
    val_prob: np.ndarray
    threshold: float
    score_f1: float
    score_acc: float
    score: float


@dataclass(frozen=True)
class MetaLabelingResult:
    enabled: bool
    model: Pipeline | None
    threshold: float
    min_take_rate: float
    val_take_rate: float
    val_net_mean_return: float
    val_max_drawdown: float
    reason: str | None = None


@dataclass(frozen=True)
class ReturnsTuningResult:
    decision_threshold: float
    edge_threshold_scale: float
    direction_policy: str
    position_scale: float
    objective: float
    diagnostics: dict[str, object]


class ProbabilityBlendClassifier:
    def __init__(self, models: list[object], weights: list[float]) -> None:
        """Initialize ProbabilityBlendClassifier state."""
        if len(models) != len(weights) or not models:
            raise ValueError("models and weights must have matching non-zero length")
        weights_arr = np.asarray(weights, dtype=float)
        total = float(np.sum(weights_arr))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.models = models
        self.weights = (weights_arr / total).tolist()

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Compute predict proba."""
        prob_up = np.zeros(len(x), dtype=float)
        for model, weight in zip(self.models, self.weights):
            prob_up += float(weight) * np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        prob_up = np.clip(prob_up, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - prob_up, prob_up])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Compute predict."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def _triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sigma: np.ndarray,
    steps_ahead: int,
    sigma_mult: float,
) -> np.ndarray:
    """Internal helper to compute triple barrier labels."""
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
    """Internal helper to compute database path."""
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _connect() -> sqlite3.Connection:
    """Internal helper to compute connect."""
    connection = sqlite3.connect(_database_path())
    connection.row_factory = sqlite3.Row
    return connection


def load_candles(symbol: str, granularity: str, as_of_start_time: int | None = None) -> pd.DataFrame:
    """Load candles."""
    as_of = int(as_of_start_time) if isinstance(as_of_start_time, int) else None
    try:
        with _connect() as conn:
            if as_of is None:
                rows = conn.execute(
                    """
                    SELECT start_time, open, high, low, close, volume
                    FROM candles
                    WHERE symbol = ? AND granularity = ?
                    ORDER BY start_time ASC
                    """,
                    (symbol.upper(), granularity),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT start_time, open, high, low, close, volume
                    FROM candles
                    WHERE symbol = ? AND granularity = ? AND start_time <= ?
                    ORDER BY start_time ASC
                    """,
                    (symbol.upper(), granularity, as_of),
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


def _parse_horizon_rows_map(raw: str, fallback: dict[str, int]) -> dict[str, int]:
    """Parse horizon-to-row-limit configuration text."""
    parsed = dict(fallback)
    valid_horizons = {spec.label for spec in HORIZON_SPECS}
    for token in raw.split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        key_raw, value_raw = item.split(":", 1)
        key = key_raw.strip().lower()
        try:
            value = int(value_raw.strip())
        except ValueError:
            continue
        if key in valid_horizons and value > 0:
            parsed[key] = value
    return parsed


def horizon_training_window_rows_map() -> dict[str, int]:
    """Resolve horizon-specific recent-window row caps from settings."""
    settings = get_settings()
    return _parse_horizon_rows_map(settings.train_horizon_window_rows_map, DEFAULT_TRAIN_HORIZON_WINDOW_ROWS)


def training_window_config_snapshot() -> dict[str, object]:
    """Expose current horizon-window configuration for reports/manifests."""
    settings = get_settings()
    rows_map = horizon_training_window_rows_map()
    return {
        "enabled": bool(settings.train_horizon_windowing_enabled),
        "rows_map": {key: int(value) for key, value in rows_map.items()},
    }


def _apply_horizon_training_window(
    frame: pd.DataFrame,
    *,
    horizon: str,
    required_rows: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Trim to a horizon-specific recent window while preserving minimum training depth."""
    settings = get_settings()
    rows_map = horizon_training_window_rows_map()
    configured_limit = int(rows_map.get(horizon, 0))
    minimum_safe_rows = max(int(required_rows), 1)
    effective_limit = max(configured_limit, minimum_safe_rows) if configured_limit > 0 else minimum_safe_rows
    archive_rows = int(len(frame))
    trim_applied = bool(settings.train_horizon_windowing_enabled) and archive_rows > effective_limit
    trimmed = frame.tail(effective_limit).reset_index(drop=True) if trim_applied else frame.reset_index(drop=True)
    return trimmed, {
        "enabled": bool(settings.train_horizon_windowing_enabled),
        "configured_rows_limit": configured_limit if configured_limit > 0 else None,
        "effective_rows_limit": int(effective_limit),
        "archive_rows_before_window": archive_rows,
        "rows_after_window": int(len(trimmed)),
        "trim_applied": trim_applied,
    }


def list_symbols_with_any_candles(min_rows: int = 100) -> list[str]:
    """List symbols with any candles."""
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
    """Internal helper to compute MAPE."""
    denom = np.clip(np.abs(y_true), 1e-12, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _periods_per_year_from_horizon(horizon: str) -> float:
    """Internal helper to map horizon label to annualization factor."""
    value = horizon.strip().lower()
    if value.endswith("mo") and value[:-2].isdigit():
        months = max(int(value[:-2]), 1)
        return 12.0 / months
    if not value:
        return 365.0 * 24.0
    unit = value[-1]
    qty_raw = value[:-1]
    if not qty_raw.isdigit():
        return 365.0 * 24.0
    qty = max(int(qty_raw), 1)
    if unit == "m":
        return (365.0 * 24.0 * 60.0) / qty
    if unit == "h":
        return (365.0 * 24.0) / qty
    if unit == "d":
        return 365.0 / qty
    if unit == "w":
        return 52.0 / qty
    return 365.0 * 24.0


def _vol_target_position_scale(volatility_20: np.ndarray, horizon: str, target_annual_vol: float, max_leverage: float) -> np.ndarray:
    """Compute volatility-targeted position scale."""
    annual_target = max(float(target_annual_vol), 1e-6)
    max_leverage = float(np.clip(max_leverage, 0.05, 5.0))
    periods = max(_periods_per_year_from_horizon(horizon), 1.0)
    target_step_vol = annual_target / math.sqrt(periods)
    vol = np.asarray(volatility_20, dtype=float)
    vol = np.nan_to_num(vol, nan=np.nanmedian(vol[np.isfinite(vol)]) if np.any(np.isfinite(vol)) else target_step_vol)
    vol = np.clip(vol, 1e-6, 5.0)
    scale = target_step_vol / vol
    return np.clip(scale, 0.0, max_leverage)


def _signal_quality_metrics(oriented_signal: np.ndarray, target_return: np.ndarray) -> tuple[float, float, float]:
    """Compute signal-quality metrics for alpha hypothesis governance."""
    if oriented_signal.size == 0 or target_return.size == 0 or oriented_signal.size != target_return.size:
        return math.nan, math.nan, math.nan
    if np.std(oriented_signal) <= 1e-12 or np.std(target_return) <= 1e-12:
        ic = math.nan
    else:
        ic = float(np.corrcoef(oriented_signal, target_return)[0, 1])
    try:
        p90 = float(np.quantile(oriented_signal, 0.90))
        p10 = float(np.quantile(oriented_signal, 0.10))
        top = target_return[oriented_signal >= p90]
        bottom = target_return[oriented_signal <= p10]
        decile_spread_bps = float((np.mean(top) - np.mean(bottom)) * 10000.0) if len(top) and len(bottom) else math.nan
    except Exception:
        decile_spread_bps = math.nan
    alignment = float(np.mean(np.sign(oriented_signal) == np.sign(target_return)))
    return ic, decile_spread_bps, alignment


def _alpha_kill_decisions(frame: pd.DataFrame, selected_features: list[str], horizon: str) -> tuple[list[str], dict[str, object]]:
    """Apply rolling alpha-signal kill rules and return active feature list."""
    settings = get_settings()
    if not bool(settings.alpha_kill_switch_enabled):
        return list(selected_features), {"enabled": False, "reason": "alpha_kill_switch_disabled"}

    alpha_features = [feature for feature in selected_features if feature in ALPHA_SIGNAL_DIRECTIONS]
    if not alpha_features:
        return list(selected_features), {"enabled": True, "reason": "no_alpha_features_in_selection", "rows": []}

    lookback = max(int(settings.alpha_kill_lookback_rows), 120)
    scoped = frame.tail(lookback).copy()
    if scoped.empty:
        return list(selected_features), {"enabled": True, "reason": "empty_scope", "rows": []}

    current = np.clip(scoped["close"].to_numpy(dtype=float), 1e-12, None)
    target = np.asarray(scoped["target_close"].to_numpy(dtype=float), dtype=float)
    future_return = (target / current) - 1.0
    idx = np.arange(future_return.size, dtype=int)
    windows = np.array_split(idx, 3)
    rows: list[dict[str, object]] = []
    surviving: set[str] = set()

    min_ic = float(settings.alpha_kill_min_ic)
    min_spread = float(settings.alpha_kill_min_decile_spread_bps)
    min_align = float(settings.alpha_kill_min_sign_alignment)
    min_survival = float(np.clip(settings.alpha_kill_min_survival_ratio, 0.0, 1.0))
    require_financial_pass = bool(settings.alpha_kill_require_financial_pass)
    min_net_mean_return = float(settings.alpha_kill_min_signal_net_mean_return)
    min_sharpe_net = float(settings.alpha_kill_min_signal_sharpe_net)
    max_drawdown_limit = float(settings.alpha_kill_max_signal_drawdown_limit)
    fee_bps = float(settings.execution_fee_bps)
    slippage_bps = float(settings.execution_slippage_bps)
    max_turnover_per_step = float(settings.execution_max_turnover_per_step)
    initial_capital = float(settings.paper_trade_initial_capital)

    def _signal_position(signal_values: np.ndarray) -> np.ndarray:
        """Scale raw signal into bounded tradable positions."""
        magnitude = np.asarray(signal_values, dtype=float)
        scale = float(np.nanpercentile(np.abs(magnitude), 75)) if magnitude.size else 0.0
        scale = max(scale, 1e-6)
        return np.clip(np.tanh(magnitude / scale), -1.0, 1.0)

    for feature in alpha_features:
        direction = float(ALPHA_SIGNAL_DIRECTIONS[feature])
        signal = np.asarray(scoped[feature].to_numpy(dtype=float), dtype=float) * direction
        window_pass = 0
        total_windows = 0
        window_rows: list[dict[str, float]] = []
        for window in windows:
            if window.size < 25:
                continue
            sig_w = signal[window]
            ret_w = future_return[window]
            mask = np.isfinite(sig_w) & np.isfinite(ret_w)
            if int(np.sum(mask)) < 25:
                continue
            total_windows += 1
            sig_valid = sig_w[mask]
            ret_valid = ret_w[mask]
            current_valid = current[window][mask]
            target_valid = target[window][mask]
            ic, spread_bps, align = _signal_quality_metrics(sig_valid, ret_valid)
            stat_pass = (
                (math.isfinite(ic) and ic >= min_ic)
                and (math.isfinite(spread_bps) and spread_bps >= min_spread)
                and (math.isfinite(align) and align >= min_align)
            )
            positions = _signal_position(sig_valid)
            direction_up = (positions >= 0).astype(int)
            execution = execution_aware_metrics(
                current_close=current_valid,
                target_close=target_valid,
                direction_up=direction_up,
                horizon=horizon,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                max_turnover_per_step=max_turnover_per_step,
                position_signal=positions,
            )
            paper = paper_trading_metrics(
                current_close=current_valid,
                target_close=target_valid,
                direction_up=direction_up,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                max_turnover_per_step=max_turnover_per_step,
                initial_capital=initial_capital,
                position_signal=positions,
            )
            net_mean_return = float(execution.get("net_mean_return", math.nan))
            sharpe_net = float(execution.get("sharpe_net", math.nan))
            max_drawdown = float(paper.get("max_drawdown", math.nan))
            financial_pass = (
                math.isfinite(net_mean_return)
                and math.isfinite(sharpe_net)
                and math.isfinite(max_drawdown)
                and (net_mean_return >= min_net_mean_return)
                and (sharpe_net >= min_sharpe_net)
                and (max_drawdown >= max_drawdown_limit)
            )
            pass_window = bool(stat_pass and (financial_pass if require_financial_pass else True))
            if pass_window:
                window_pass += 1
            financial_failed_reasons: list[str] = []
            if require_financial_pass:
                if not (math.isfinite(net_mean_return) and net_mean_return >= min_net_mean_return):
                    financial_failed_reasons.append("net_mean_return_below_threshold")
                if not (math.isfinite(sharpe_net) and sharpe_net >= min_sharpe_net):
                    financial_failed_reasons.append("sharpe_net_below_threshold")
                if not (math.isfinite(max_drawdown) and max_drawdown >= max_drawdown_limit):
                    financial_failed_reasons.append("drawdown_below_limit")
            window_rows.append(
                {
                    "window_samples": float(np.sum(mask)),
                    "ic": ic,
                    "decile_spread_bps": spread_bps,
                    "sign_alignment": align,
                    "net_mean_return": net_mean_return,
                    "sharpe_net": sharpe_net,
                    "max_drawdown": max_drawdown,
                    "stat_pass": float(1.0 if stat_pass else 0.0),
                    "financial_pass": float(1.0 if financial_pass else 0.0),
                    "financial_failed_reasons": financial_failed_reasons,
                    "pass": float(1.0 if pass_window else 0.0),
                }
            )

        survival_ratio = (window_pass / total_windows) if total_windows > 0 else 0.0
        keep = survival_ratio >= min_survival
        if keep:
            surviving.add(feature)
        rows.append(
            {
                "feature": feature,
                "expected_direction": "positive" if direction > 0 else "negative",
                "rationale": ALPHA_SIGNAL_RATIONALES.get(feature, ""),
                "window_count": float(total_windows),
                "passed_windows": float(window_pass),
                "survival_ratio": float(survival_ratio),
                "min_required_survival_ratio": float(min_survival),
                "financial_gate_enabled": bool(require_financial_pass),
                "financial_thresholds": {
                    "min_net_mean_return": float(min_net_mean_return),
                    "min_sharpe_net": float(min_sharpe_net),
                    "max_drawdown_limit": float(max_drawdown_limit),
                },
                "kept": bool(keep),
                "windows": window_rows,
            }
        )

    active = [feature for feature in selected_features if (feature not in ALPHA_SIGNAL_DIRECTIONS) or (feature in surviving)]
    return active, {"enabled": True, "rows": rows}


def _split_indices(length: int) -> tuple[int, int]:
    """Internal helper to compute split indices."""
    train_end = int(length * 0.7)
    val_end = int(length * 0.85)
    return train_end, val_end


def _purged_split_frames(
    enriched: pd.DataFrame,
    train_end: int,
    val_end: int,
    steps_ahead: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Internal helper to compute purged split frames."""
    purge = max(int(steps_ahead), 1)
    train_raw = enriched.iloc[:train_end]
    val_raw = enriched.iloc[train_end:val_end]
    test_raw = enriched.iloc[val_end:]

    if len(train_raw) > purge:
        train_df = train_raw.iloc[:-purge]
    else:
        train_df = train_raw.iloc[0:0]

    if len(val_raw) > (purge * 2):
        val_df = val_raw.iloc[purge:-purge]
    else:
        val_df = val_raw.iloc[0:0]

    if len(test_raw) > purge:
        test_df = test_raw.iloc[purge:]
    else:
        test_df = test_raw.iloc[0:0]

    return train_df, val_df, test_df


def _split_leakage_diagnostic(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    steps_ahead: int,
) -> dict[str, object]:
    """Internal helper to compute split leakage diagnostic."""
    if train_df.empty or val_df.empty or test_df.empty:
        return {
            "pass": False,
            "reason": "empty_split_segment",
            "required_gap_rows": int(max(steps_ahead, 1)),
        }

    train_last_idx = int(train_df.index.max())
    val_first_idx = int(val_df.index.min())
    val_last_idx = int(val_df.index.max())
    test_first_idx = int(test_df.index.min())

    gap_train_val = int(val_first_idx - train_last_idx - 1)
    gap_val_test = int(test_first_idx - val_last_idx - 1)
    required_gap = int(max(steps_ahead, 1))
    split_gap_ok = gap_train_val >= required_gap and gap_val_test >= required_gap

    time_order_ok = True
    if "start_time" in train_df.columns and "start_time" in val_df.columns and "start_time" in test_df.columns:
        train_last_time = float(train_df["start_time"].iloc[-1])
        val_first_time = float(val_df["start_time"].iloc[0])
        val_last_time = float(val_df["start_time"].iloc[-1])
        test_first_time = float(test_df["start_time"].iloc[0])
        time_order_ok = bool((train_last_time < val_first_time) and (val_last_time < test_first_time))

    return {
        "pass": bool(split_gap_ok and time_order_ok),
        "required_gap_rows": required_gap,
        "gap_train_to_val_rows": gap_train_val,
        "gap_val_to_test_rows": gap_val_test,
        "time_order_ok": bool(time_order_ok),
        "split_gap_ok": bool(split_gap_ok),
    }


def min_samples_for_horizon(horizon: str) -> int:
    """Compute min samples for horizon."""
    return int(HORIZON_MIN_SAMPLES.get(horizon, MIN_SAMPLES))


def clip_log_return_predictions(values: np.ndarray, horizon: str) -> np.ndarray:
    """Compute clip log return predictions."""
    clip_value = float(HORIZON_LOG_RETURN_CLIP.get(horizon, 1.0))
    return np.clip(values, -clip_value, clip_value)


def residual_clip_abs_from_close(current_close: np.ndarray | float, horizon: str) -> np.ndarray:
    """Compute residual clip abs from close."""
    clip_value = float(HORIZON_LOG_RETURN_CLIP.get(horizon, 1.0))
    close_array = np.asarray(current_close, dtype=float)
    return np.maximum(close_array * (math.exp(clip_value) - 1.0), 1e-8)


def _recent_sample_weights(length: int, min_weight: float = 0.35) -> np.ndarray:
    """Internal helper to compute recent sample weights."""
    if length <= 1:
        return np.ones(max(length, 1), dtype=float)
    min_weight = float(np.clip(min_weight, 0.05, 1.0))
    exponents = np.linspace(math.log(min_weight), 0.0, num=length, dtype=float)
    return np.exp(exponents)


def regression_output_to_close(
    raw_prediction: np.ndarray,
    current_close: np.ndarray,
    horizon: str,
    regression_target: str,
) -> np.ndarray:
    """Compute regression output to close."""
    raw = np.asarray(raw_prediction, dtype=float)
    current = np.asarray(current_close, dtype=float)
    if regression_target == "log_return":
        clipped = clip_log_return_predictions(raw, horizon)
        return np.maximum(current * np.exp(clipped), 1e-8)
    if regression_target == "residual_from_persistence":
        clip_abs = residual_clip_abs_from_close(current, horizon)
        clipped_residual = np.clip(raw, -clip_abs, clip_abs)
        return np.maximum(current + clipped_residual, 1e-8)
    return np.maximum(raw, 1e-8)


def apply_directional_tilt_to_close(
    predicted_close: np.ndarray,
    current_close: np.ndarray,
    probability_up: np.ndarray,
    volatility_feature: np.ndarray,
    horizon: str,
    gamma: float,
) -> np.ndarray:
    """Apply directional tilt to close."""
    pred = np.asarray(predicted_close, dtype=float)
    current = np.asarray(current_close, dtype=float)
    if pred.size == 0:
        return pred
    if abs(float(gamma)) <= 1e-12:
        return np.maximum(pred, 1e-8)

    prob_up = np.asarray(probability_up, dtype=float)
    vol = np.asarray(volatility_feature, dtype=float)
    signal = np.clip((prob_up - 0.5) * 2.0, -1.0, 1.0)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.clip(vol, 0.0, 0.50)
    max_log_shift = float(HORIZON_DIRECTIONAL_TILT_MAX_LOG_SHIFT.get(horizon, 0.12))
    log_shift = np.clip(float(gamma) * signal * vol, -max_log_shift, max_log_shift)
    adjusted = pred * np.exp(log_shift)

    clip_abs = residual_clip_abs_from_close(current, horizon)
    lower = np.maximum(current - clip_abs, 1e-8)
    upper = np.maximum(current + clip_abs, 1e-8)
    adjusted = np.clip(adjusted, lower, upper)
    return np.maximum(adjusted, 1e-8)


def resolve_horizon_data(
    symbol: str,
    spec: HorizonSpec,
    min_samples: int | None = None,
    as_of_cutoff_by_granularity: dict[str, int] | None = None,
    sync_row_depth_by_granularity: dict[str, int] | None = None,
) -> ResolvedHorizonData | None:
    """Resolve horizon data."""
    required = min_samples if min_samples is not None else min_samples_for_horizon(spec.label)
    for granularity, steps in spec.candidates:
        as_of = None
        if isinstance(as_of_cutoff_by_granularity, dict) and granularity in as_of_cutoff_by_granularity:
            as_of = int(as_of_cutoff_by_granularity[granularity])
        candidate = load_candles(symbol, granularity, as_of_start_time=as_of)
        if isinstance(sync_row_depth_by_granularity, dict) and granularity in sync_row_depth_by_granularity:
            sync_rows = int(sync_row_depth_by_granularity[granularity])
            if sync_rows > 0:
                # Enforce cross-symbol parity by training on the synchronized trailing depth.
                candidate = candidate.tail(sync_rows).reset_index(drop=True)
        archive_rows_before_window = int(len(candidate))
        archive_start_time = int(candidate["start_time"].iloc[0]) if archive_rows_before_window else None
        archive_end_time = int(candidate["start_time"].iloc[-1]) if archive_rows_before_window else None
        required_rows = required + steps + LOOKBACK_BUFFER
        candidate, window_meta = _apply_horizon_training_window(
            candidate,
            horizon=spec.label,
            required_rows=required_rows,
        )
        if len(candidate) >= required_rows:
            return ResolvedHorizonData(
                granularity=granularity,
                steps_ahead=steps,
                frame=candidate,
                archive_rows_before_window=archive_rows_before_window,
                archive_start_time=archive_start_time,
                archive_end_time=archive_end_time,
                horizon_window_rows_limit=(
                    int(window_meta["effective_rows_limit"])
                    if isinstance(window_meta.get("effective_rows_limit"), int)
                    else None
                ),
                horizon_window_trim_applied=bool(window_meta.get("trim_applied")),
            )
    return None


def _prepare_supervised(df: pd.DataFrame, horizon: str, steps_ahead: int, min_samples: int) -> tuple[pd.DataFrame, SplitData] | None:
    """Internal helper to compute prepare supervised."""
    settings = get_settings()
    selected_features = feature_columns_for_horizon(horizon)
    enriched = build_features(df)
    enriched["target_close"] = enriched["close"].shift(-steps_ahead)
    enriched["target_log_return"] = np.log((enriched["target_close"] + 1e-12) / (enriched["close"] + 1e-12))
    if settings.classification_label_mode.strip().lower() == "triple_barrier":
        # Triple-barrier labels are more robust in volatile regimes than simple terminal direction.
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

    active_features, alpha_kill_diagnostics = _alpha_kill_decisions(enriched, selected_features, horizon=horizon)
    if not active_features:
        return None

    train_end, val_end = _split_indices(len(enriched))
    # Purged splits reduce leakage from overlapping forecast windows across boundaries.
    train_df, val_df, test_df = _purged_split_frames(
        enriched=enriched,
        train_end=train_end,
        val_end=val_end,
        steps_ahead=steps_ahead,
    )
    leakage_diagnostic = _split_leakage_diagnostic(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        steps_ahead=steps_ahead,
    )

    if len(test_df) < 30 or train_df["target_up"].nunique() < 2:
        return None
    if len(val_df) < 30:
        return None
    if not bool(leakage_diagnostic.get("pass")):
        return None

    split = SplitData(
        feature_columns=active_features,
        x_train=train_df[active_features],
        x_val=val_df[active_features],
        x_test=test_df[active_features],
        y_train_cls=train_df["target_up"],
        y_val_cls=val_df["target_up"],
        y_test_cls=test_df["target_up"].to_numpy(dtype=int),
        y_train_reg=train_df["target_log_return"],
        y_val_reg=val_df["target_log_return"],
        y_test_reg=test_df["target_log_return"].to_numpy(dtype=float),
        current_close_train=train_df["close"].to_numpy(dtype=float),
        current_close_val=val_df["close"].to_numpy(dtype=float),
        current_close_test=test_df["close"].to_numpy(dtype=float),
        target_close_train=train_df["target_close"].to_numpy(dtype=float),
        target_close_val=val_df["target_close"].to_numpy(dtype=float),
        target_close_test=test_df["target_close"].to_numpy(dtype=float),
        regime_train=np.argmax(train_df[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float), axis=1),
        regime_val=np.argmax(val_df[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float), axis=1),
        regime_test=np.argmax(test_df[["markov_prob_down", "markov_prob_flat", "markov_prob_up"]].to_numpy(dtype=float), axis=1),
        leakage_diagnostic=leakage_diagnostic,
        alpha_kill_diagnostics=alpha_kill_diagnostics,
    )
    return enriched, split


def _classification_candidates(horizon: str) -> list[tuple[str, Pipeline]]:
    """Build classification candidates with theory-driven defaults."""
    settings = get_settings()
    mode = settings.model_candidate_mode.strip().lower()
    theory_mode = mode in {"theory", "theory_driven", "theory-driven"}

    logistic = (
        "logistic_regression",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
            ]
        ),
    )
    hist_gb = (
        "hist_gradient_boosting_classifier",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.045,
                        max_depth=6,
                        min_samples_leaf=10,
                        random_state=42,
                    ),
                ),
            ]
        ),
    )
    gradient = (
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
    )

    if theory_mode:
        base: list[tuple[str, Pipeline]] = [logistic, hist_gb]
        if horizon in {"5m", "1h", "3h"}:
            base.append(gradient)
    else:
        base = [
            logistic,
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
            gradient,
            hist_gb,
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

    if not bool(settings.model_enable_ensemble_challenger) or len(base) < 2:
        return base

    # Ensemble is a challenger; it is not mandatory for theory-mode training.
    stack_estimators = [(f"m{idx}", model) for idx, (_name, model) in enumerate(base[: min(len(base), 4)], start=1)]
    stacking = Pipeline(
        steps=[
            (
                "model",
                StackingClassifier(
                    estimators=stack_estimators,
                    final_estimator=LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
                    stack_method="predict_proba",
                    passthrough=False,
                    n_jobs=1,
                ),
            )
        ]
    )
    return base + [("stacking_classifier_challenger", stacking)]


def _regression_candidates(horizon: str) -> list[tuple[str, Pipeline]]:
    """Build regression candidates with theory-driven defaults."""
    settings = get_settings()
    mode = settings.model_candidate_mode.strip().lower()
    theory_mode = mode in {"theory", "theory_driven", "theory-driven"}

    bayesian_arx = (
        "bayesian_ridge_arx_regressor",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    BayesianRidge(
                        alpha_1=1e-6,
                        alpha_2=1e-6,
                        lambda_1=1e-6,
                        lambda_2=1e-6,
                        compute_score=False,
                    ),
                ),
            ]
        ),
    )
    ridge = (
        "ridge_regressor",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.2, random_state=42)),
            ]
        ),
    )
    huber = (
        "huber_regressor",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=500)),
            ]
        ),
    )
    hist_gb = (
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
    )

    if theory_mode:
        base: list[tuple[str, Pipeline]] = [bayesian_arx, ridge, huber, hist_gb]
        if horizon in {"1d", "1w", "1mo", "3mo"}:
            base.append(
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
                )
            )
    else:
        base = [
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
            hist_gb,
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
            ridge,
            (
                "elastic_net_regressor",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(alpha=0.0015, l1_ratio=0.2, random_state=42, max_iter=4000)),
                    ]
                ),
            ),
            huber,
            bayesian_arx,
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

    if not bool(settings.model_enable_ensemble_challenger) or len(base) < 2:
        return base

    stack_estimators = [(f"m{idx}", model) for idx, (_name, model) in enumerate(base[: min(len(base), 4)], start=1)]
    stacking = Pipeline(
        steps=[
            (
                "model",
                StackingRegressor(
                    estimators=stack_estimators,
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
    return base + [("stacking_regressor_challenger", stacking)]


def _class_balance_sample_weight(y: pd.Series, down_weight_boost: float = 1.0) -> np.ndarray:
    """Internal helper to compute class balance sample weight."""
    values = y.to_numpy(dtype=int)
    total = len(values)
    positive = max(int(np.sum(values == 1)), 1)
    negative = max(int(np.sum(values == 0)), 1)
    w_pos = total / (2.0 * positive)
    w_neg = total / (2.0 * negative)
    if down_weight_boost > 1.0:
        w_neg *= float(down_weight_boost)
    return np.where(values == 1, w_pos, w_neg).astype(float)


def _down_weight_boost_candidates(y: pd.Series) -> list[float]:
    """Internal helper to compute down weight boost candidates."""
    values = y.to_numpy(dtype=int)
    if values.size == 0:
        return [1.0]
    up_rate = float(np.mean(values == 1))
    if up_rate >= 0.72:
        return [1.0, 1.25, 1.5, 1.8, 2.2]
    if up_rate >= 0.65:
        return [1.0, 1.2, 1.45, 1.7]
    if up_rate >= 0.58:
        return [1.0, 1.15, 1.32, 1.5]
    if up_rate >= 0.53:
        return [1.0, 1.1, 1.22]
    return [1.0]


def _fit_pipeline_with_optional_weight(model: Pipeline, x: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None) -> None:
    """Internal helper to compute fit pipeline with optional weight."""
    if sample_weight is None:
        model.fit(x, y)
        return
    last_step = model.steps[-1][0]
    try:
        model.fit(x, y, **{f"{last_step}__sample_weight": sample_weight})
    except (TypeError, ValueError):
        model.fit(x, y)


def _best_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float, dict[str, object]]:
    """Internal helper to compute best threshold for F1."""
    settings = get_settings()
    if bool(settings.walk_forward_threshold_enabled):
        threshold, wf_payload = choose_threshold_walk_forward(
            y_true=np.asarray(y_true, dtype=int),
            y_prob=np.asarray(y_prob, dtype=float),
            folds=max(int(settings.walk_forward_threshold_folds), 2),
        )
        y_pred_wf = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
        score_f1_wf = float(f1_score(y_true, y_pred_wf, zero_division=0))
        score_acc_wf = float(accuracy_score(y_true, y_pred_wf))
        return threshold, score_f1_wf, score_acc_wf, wf_payload

    baseline_pred = np.ones_like(y_true)
    baseline_f1 = float(f1_score(y_true, baseline_pred, zero_division=0))
    baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
    quantiles = np.quantile(y_prob, np.linspace(0.05, 0.95, 19))
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.20, 0.90, 71),
                quantiles,
                np.array([0.22, 0.28, 0.34, 0.5, 0.66, 0.72, 0.78, 0.84, 0.88]),
            ]
        )
    )
    rows: list[tuple[float, float, float, float, float, float, float]] = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        score_acc = float(accuracy_score(y_true, y_pred))
        score_precision = float(precision_score(y_true, y_pred, zero_division=0))
        delta_f1 = score_f1 - baseline_f1
        delta_acc = score_acc - baseline_accuracy
        score = min(delta_f1, delta_acc) + (0.45 * delta_f1) + (0.25 * delta_acc) + (0.02 * score_precision)
        rows.append((float(threshold), score_f1, score_acc, delta_f1, delta_acc, score_precision, score))

    # First priority: clear both classification gate dimensions.
    both_pass = [row for row in rows if row[3] > 0 and row[4] > 0]
    if both_pass:
        best = max(both_pass, key=lambda row: (min(row[3], row[4]), row[3], row[4], row[5]))
        return best[0], best[1], best[2], {"mode": "grid_search"}

    # Second priority: keep accuracy at/above baseline while maximizing F1 lift.
    acc_ok = [row for row in rows if row[4] >= 0]
    if acc_ok:
        best = max(acc_ok, key=lambda row: (row[3], row[4], row[5], row[1], row[6]))
        return best[0], best[1], best[2], {"mode": "grid_search"}

    # Fallback to previous composite scoring when no gate-friendly threshold exists.
    best = max(rows, key=lambda row: (row[6], row[1], row[2], row[5]))
    return best[0], best[1], best[2], {"mode": "grid_search"}


def _classifier_stability_penalty(split: SplitData, y_val_pred: np.ndarray, tuning_payload: dict[str, object]) -> tuple[float, dict[str, float]]:
    """Compute validation stability penalty for classifier selection."""
    y_true = np.asarray(split.y_val_cls.to_numpy(dtype=int), dtype=int)
    regime_ids = np.asarray(split.regime_val, dtype=int)
    regime_f1_rows: list[float] = []
    for regime_id in [0, 1, 2]:
        mask = regime_ids == regime_id
        if int(np.sum(mask)) < 20:
            continue
        regime_f1_rows.append(float(f1_score(y_true[mask], y_val_pred[mask], zero_division=0)))
    regime_std = float(np.std(regime_f1_rows)) if regime_f1_rows else 0.0

    fold_f1_delta_std = 0.0
    if isinstance(tuning_payload, dict):
        folds = tuning_payload.get("folds", [])
        if isinstance(folds, list) and folds:
            f1_deltas = [float(item.get("f1_delta", 0.0)) for item in folds if isinstance(item, dict)]
            if f1_deltas:
                fold_f1_delta_std = float(np.std(f1_deltas))
    penalty = (0.30 * regime_std) + (0.15 * fold_f1_delta_std)
    return penalty, {"regime_f1_std": regime_std, "walk_forward_f1_delta_std": fold_f1_delta_std}


def _select_best_classifier(split: SplitData, horizon: str) -> CandidateResult:
    """Select best classifier. Internal helper."""
    best: CandidateResult | None = None
    val_candidates: list[ClassifierValidationCandidate] = []
    val_true = split.y_val_cls.to_numpy(dtype=int)
    val_baseline_pred = np.ones_like(val_true)
    val_baseline_f1 = float(f1_score(val_true, val_baseline_pred, zero_division=0))
    val_baseline_accuracy = float(accuracy_score(val_true, val_baseline_pred))
    # Search both model family and class-imbalance weighting to optimize strict gate margins.
    for down_weight_boost in _down_weight_boost_candidates(split.y_train_cls):
        sample_weight = _class_balance_sample_weight(split.y_train_cls, down_weight_boost=down_weight_boost)
        for model_name, model in _classification_candidates(horizon):
            candidate_model = clone(model)
            _fit_pipeline_with_optional_weight(candidate_model, split.x_train, split.y_train_cls, sample_weight)
            val_prob = candidate_model.predict_proba(split.x_val)[:, 1]
            threshold, score_f1, score_acc, tuning_payload = _best_threshold_for_f1(val_true, val_prob)
            delta_f1 = score_f1 - val_baseline_f1
            delta_acc = score_acc - val_baseline_accuracy
            val_pred = (val_prob >= threshold).astype(int)
            stability_penalty, stability_payload = _classifier_stability_penalty(split, val_pred, tuning_payload)
            # Rank by gate-oriented validation margins rather than raw F1 alone.
            score = min(delta_f1, delta_acc) + (0.45 * delta_f1) + (0.25 * delta_acc) - stability_penalty
            val_candidates.append(
                ClassifierValidationCandidate(
                    model_name=model_name,
                    model=candidate_model,
                    down_weight_boost=float(down_weight_boost),
                    val_prob=val_prob,
                    threshold=threshold,
                    score_f1=score_f1,
                    score_acc=score_acc,
                    score=score,
                )
            )
            if best is None or score > best.val_score:
                best = CandidateResult(
                    model_name=model_name,
                    model=candidate_model,
                    val_score=score,
                    decision_threshold=threshold,
                    class_down_weight_boost=float(down_weight_boost),
                    threshold_tuning=tuning_payload,
                    stability=stability_payload,
                )

    if val_candidates:
        # Try lightweight probability blends among top candidates for extra F1/accuracy edge.
        top = sorted(val_candidates, key=lambda item: item.score, reverse=True)[:3]
        for idx_a in range(len(top)):
            for idx_b in range(idx_a + 1, len(top)):
                cand_a = top[idx_a]
                cand_b = top[idx_b]
                for w_a in [0.5, 0.6, 0.7]:
                    w_b = 1.0 - w_a
                    blended_prob = (w_a * cand_a.val_prob) + (w_b * cand_b.val_prob)
                    threshold, score_f1, score_acc, tuning_payload = _best_threshold_for_f1(val_true, blended_prob)
                    delta_f1 = score_f1 - val_baseline_f1
                    delta_acc = score_acc - val_baseline_accuracy
                    blended_pred = (blended_prob >= threshold).astype(int)
                    stability_penalty, stability_payload = _classifier_stability_penalty(split, blended_pred, tuning_payload)
                    score = min(delta_f1, delta_acc) + (0.45 * delta_f1) + (0.25 * delta_acc) - stability_penalty
                    if best is None or score > best.val_score:
                        best = CandidateResult(
                            model_name=f"blend_{cand_a.model_name}_{cand_b.model_name}_{w_a:.2f}",
                            model=ProbabilityBlendClassifier(
                                models=[cand_a.model, cand_b.model],
                                weights=[w_a, w_b],
                            ),
                            val_score=score,
                            decision_threshold=threshold,
                            class_down_weight_boost=float((w_a * cand_a.down_weight_boost) + (w_b * cand_b.down_weight_boost)),
                            threshold_tuning=tuning_payload,
                            stability=stability_payload,
                        )
    if best is None:
        raise RuntimeError("No classification candidate available")
    return best


def _best_regression_blend_alpha(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> tuple[float, float]:
    """Internal helper to compute best regression blend alpha."""
    best_alpha = 1.0
    best_rmse = math.inf
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 81),
                np.array(
                    [
                        0.0,
                        0.001,
                        0.002,
                        0.003,
                        0.004,
                        0.005,
                        0.0075,
                        0.01,
                        0.015,
                        0.02,
                        0.03,
                        0.04,
                        0.05,
                        0.95,
                        0.96,
                        0.97,
                        0.98,
                        0.985,
                        0.99,
                        0.995,
                        0.998,
                        0.999,
                        1.0,
                    ]
                ),
            ]
        )
    )
    for alpha in candidates:
        blended = (alpha * y_pred_model) + ((1.0 - alpha) * y_pred_baseline)
        rmse = float(math.sqrt(mean_squared_error(y_true, blended)))
        if rmse < best_rmse:
            best_alpha = float(alpha)
            best_rmse = rmse
        elif math.isclose(rmse, best_rmse, rel_tol=1e-12, abs_tol=1e-12) and float(alpha) > best_alpha:
            # If multiple blend weights are equally good on validation,
            # prefer the one that keeps more model signal than pure baseline.
            best_alpha = float(alpha)
    return best_alpha, best_rmse


def _best_directional_tilt_gamma(
    y_true: np.ndarray,
    current_close: np.ndarray,
    base_pred_close: np.ndarray,
    prob_up: np.ndarray,
    volatility_20: np.ndarray,
    horizon: str,
) -> tuple[float, float]:
    """Internal helper to compute best directional tilt gamma."""
    baseline_rmse = float(math.sqrt(mean_squared_error(y_true, base_pred_close)))
    best_gamma = 0.0
    best_rmse = baseline_rmse
    candidates = np.array(
        [
            0.0,
            -1.25,
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            -0.1,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
        ],
        dtype=float,
    )
    for gamma in candidates:
        adjusted = apply_directional_tilt_to_close(
            predicted_close=base_pred_close,
            current_close=current_close,
            probability_up=prob_up,
            volatility_feature=volatility_20,
            horizon=horizon,
            gamma=float(gamma),
        )
        rmse = float(math.sqrt(mean_squared_error(y_true, adjusted)))
        if rmse < best_rmse - 1e-12:
            best_gamma = float(gamma)
            best_rmse = rmse
            continue
        if math.isclose(rmse, best_rmse, rel_tol=1e-12, abs_tol=1e-12) and abs(float(gamma)) < abs(best_gamma):
            best_gamma = float(gamma)
    return best_gamma, best_rmse


def _regression_stability_payload(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> dict[str, float]:
    """Compute walk-forward stability diagnostics for regression candidates."""
    if y_true.size == 0 or y_true.size != y_pred.size or y_true.size != y_baseline.size:
        return {"median_rmse_delta": 0.0, "rmse_delta_std": 0.0}
    splits = walk_forward_splits(n_samples=len(y_true), folds=4, min_train_fraction=0.5, min_test_size=24)
    if not splits:
        return {"median_rmse_delta": 0.0, "rmse_delta_std": 0.0}
    deltas: list[float] = []
    for _train_start, _train_end, test_start, test_end in splits:
        y_fold = y_true[test_start:test_end]
        pred_fold = y_pred[test_start:test_end]
        base_fold = y_baseline[test_start:test_end]
        rmse_model = float(math.sqrt(mean_squared_error(y_fold, pred_fold)))
        rmse_base = float(math.sqrt(mean_squared_error(y_fold, base_fold)))
        deltas.append(rmse_base - rmse_model)
    if not deltas:
        return {"median_rmse_delta": 0.0, "rmse_delta_std": 0.0}
    return {
        "median_rmse_delta": float(np.median(deltas)),
        "rmse_delta_std": float(np.std(deltas)),
    }


def _select_best_regressor(split: SplitData, horizon: str) -> CandidateResult:
    """Select best regressor. Internal helper."""
    best: CandidateResult | None = None
    recent_weight = _recent_sample_weights(len(split.x_train))
    fit_weight_modes: list[tuple[str, np.ndarray | None]] = [("uniform", None)]
    if len(split.x_train) >= 200:
        fit_weight_modes.append(("recent", recent_weight))

    # Evaluate each regressor under both target formulations:
    # 1) horizon log-return, 2) residual from persistence baseline.
    for model_name, model in _regression_candidates(horizon):
        for weight_mode, sample_weight in fit_weight_modes:
            tuned_name = model_name if weight_mode == "uniform" else f"{model_name}_{weight_mode}"
            log_model = clone(model)
            _fit_pipeline_with_optional_weight(log_model, split.x_train, split.y_train_reg, sample_weight)
            val_pred_log_return = np.asarray(log_model.predict(split.x_val), dtype=float)
            val_pred_close_model = regression_output_to_close(
                raw_prediction=val_pred_log_return,
                current_close=split.current_close_val,
                horizon=horizon,
                regression_target="log_return",
            )
            alpha_log, rmse_log = _best_regression_blend_alpha(
                y_true=split.target_close_val,
                y_pred_model=val_pred_close_model,
                y_pred_baseline=split.current_close_val,
            )

            residual_model = clone(model)
            y_train_residual = pd.Series(
                split.target_close_train - split.current_close_train,
                index=split.y_train_reg.index,
            )
            _fit_pipeline_with_optional_weight(residual_model, split.x_train, y_train_residual, sample_weight)
            val_pred_residual = np.asarray(residual_model.predict(split.x_val), dtype=float)
            val_pred_close_residual = regression_output_to_close(
                raw_prediction=val_pred_residual,
                current_close=split.current_close_val,
                horizon=horizon,
                regression_target="residual_from_persistence",
            )
            alpha_residual, rmse_residual = _best_regression_blend_alpha(
                y_true=split.target_close_val,
                y_pred_model=val_pred_close_residual,
                y_pred_baseline=split.current_close_val,
            )

            residual_rel_improvement = (rmse_log - rmse_residual) / max(rmse_log, 1e-12)
            if residual_rel_improvement >= RESIDUAL_TARGET_MIN_REL_IMPROVEMENT:
                candidate_model = residual_model
                candidate_alpha = alpha_residual
                candidate_rmse = rmse_residual
                candidate_target = "residual_from_persistence"
            else:
                candidate_model = log_model
                candidate_alpha = alpha_log
                candidate_rmse = rmse_log
                candidate_target = "log_return"

            val_pred_close_for_stability = (
                (candidate_alpha * val_pred_close_residual) + ((1.0 - candidate_alpha) * split.current_close_val)
                if candidate_target == "residual_from_persistence"
                else (candidate_alpha * val_pred_close_model) + ((1.0 - candidate_alpha) * split.current_close_val)
            )
            stability = _regression_stability_payload(
                y_true=np.asarray(split.target_close_val, dtype=float),
                y_pred=np.asarray(val_pred_close_for_stability, dtype=float),
                y_baseline=np.asarray(split.current_close_val, dtype=float),
            )
            rmse_margin = float(
                math.sqrt(mean_squared_error(split.target_close_val, split.current_close_val))
            ) - float(candidate_rmse)
            score = rmse_margin + (0.20 * float(stability.get("median_rmse_delta", 0.0))) - (
                0.12 * float(stability.get("rmse_delta_std", 0.0))
            )
            if best is None or score > best.val_score:
                best = CandidateResult(
                    model_name=tuned_name,
                    model=candidate_model,
                    val_score=score,
                    regression_blend_alpha=candidate_alpha,
                    regression_target=candidate_target,
                    stability=stability,
                )
    if best is None:
        raise RuntimeError("No regression candidate available")
    return best


def _build_calibration_payload(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    """Build calibration payload. Internal helper."""
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
    """Calibrate confidence."""
    scale = float(calibration.get("scale", 1.0))
    min_conf = float(calibration.get("min_confidence", 0.5))
    max_conf = float(calibration.get("max_confidence", 0.99))

    raw = max(raw_probability_up, 1 - raw_probability_up)
    calibrated = min(max_conf, max(min_conf, raw * scale))
    return float(calibrated)


def _metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray, y_proba_cls: np.ndarray, y_true_reg: np.ndarray, y_pred_reg: np.ndarray) -> dict[str, float]:
    """Internal helper to compute metrics."""
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
    """Internal helper to compute confidence slice metrics."""
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


def _regime_metrics(
    *,
    split: SplitData,
    y_pred_cls: np.ndarray,
    horizon: str,
    position_signal: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
    max_turnover_per_step: float,
    initial_capital: float,
) -> dict[str, dict[str, float | dict[str, float]]]:
    """Compute classification and conditional financial diagnostics by regime."""
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
        execution = execution_aware_metrics(
            current_close=split.current_close_test[mask],
            target_close=split.target_close_test[mask],
            direction_up=y_pred_cls[mask],
            horizon=horizon,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_turnover_per_step=max_turnover_per_step,
            position_signal=np.asarray(position_signal, dtype=float)[mask],
        )
        paper = paper_trading_metrics(
            current_close=split.current_close_test[mask],
            target_close=split.target_close_test[mask],
            direction_up=y_pred_cls[mask],
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_turnover_per_step=max_turnover_per_step,
            initial_capital=initial_capital,
            position_signal=np.asarray(position_signal, dtype=float)[mask],
        )
        out[regime_name[regime_idx]] = {
            "count": float(count),
            "accuracy": float(accuracy_score(split.y_test_cls[mask], y_pred_cls[mask])),
            "f1": float(f1_score(split.y_test_cls[mask], y_pred_cls[mask], zero_division=0)),
            "net_mean_return": float(execution.get("net_mean_return", math.nan)),
            "sharpe_net": float(execution.get("sharpe_net", math.nan)),
            "max_drawdown": float(paper.get("max_drawdown", math.nan)),
            "total_return": float(paper.get("total_return", math.nan)),
        }
    return out


def _baseline_metrics(y_true_cls: np.ndarray, y_true_reg: np.ndarray, current_close: np.ndarray) -> dict[str, float]:
    """Internal helper to compute baseline metrics."""
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
    """Internal helper to compute martingale residual diagnostic."""
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
    """Internal helper to compute extract top feature importance."""
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
    """Internal helper to compute near pass delta."""
    return {
        "f1_vs_baseline": float(metrics["f1"] - baseline["f1"]),
        "accuracy_vs_baseline": float(metrics["accuracy"] - baseline["accuracy"]),
        "rmse_vs_baseline": float(baseline["rmse"] - metrics["rmse"]),
    }


def _alpha_signal_diagnostics(split: SplitData, horizon: str) -> dict[str, object]:
    """Compute out-of-sample diagnostics for explicit alpha hypotheses."""
    if split.x_test.empty:
        return {"available": False, "reason": "empty_test_set"}

    settings = get_settings()
    current = np.clip(np.asarray(split.current_close_test, dtype=float), 1e-12, None)
    future_return = (np.asarray(split.target_close_test, dtype=float) / current) - 1.0
    if future_return.size == 0:
        return {"available": False, "reason": "empty_future_return"}

    def _signal_position(signal_values: np.ndarray) -> np.ndarray:
        """Scale raw signal into bounded tradable positions."""
        magnitude = np.asarray(signal_values, dtype=float)
        scale = float(np.nanpercentile(np.abs(magnitude), 75)) if magnitude.size else 0.0
        scale = max(scale, 1e-6)
        return np.clip(np.tanh(magnitude / scale), -1.0, 1.0)

    signal_rows: list[dict[str, object]] = []
    for signal, direction in ALPHA_SIGNAL_DIRECTIONS.items():
        if signal not in split.x_test.columns:
            continue
        raw_signal = np.asarray(split.x_test[signal].to_numpy(dtype=float), dtype=float)
        mask = np.isfinite(raw_signal) & np.isfinite(future_return)
        if int(np.sum(mask)) < 25:
            continue
        oriented_signal = raw_signal[mask] * float(direction)
        target = future_return[mask]
        if np.std(oriented_signal) <= 1e-12 or np.std(target) <= 1e-12:
            corr = math.nan
        else:
            corr = float(np.corrcoef(oriented_signal, target)[0, 1])
        try:
            p90 = float(np.quantile(oriented_signal, 0.90))
            p10 = float(np.quantile(oriented_signal, 0.10))
            top = target[oriented_signal >= p90]
            bottom = target[oriented_signal <= p10]
            decile_spread_bps = float((np.mean(top) - np.mean(bottom)) * 10000.0) if len(top) and len(bottom) else math.nan
        except Exception:
            decile_spread_bps = math.nan

        alignment = float(np.mean(np.sign(oriented_signal) == np.sign(target)))
        positions = _signal_position(oriented_signal)
        direction_up = (positions >= 0).astype(int)
        execution = execution_aware_metrics(
            current_close=current[mask],
            target_close=split.target_close_test[mask],
            direction_up=direction_up,
            horizon=horizon,
            fee_bps=float(settings.execution_fee_bps),
            slippage_bps=float(settings.execution_slippage_bps),
            max_turnover_per_step=float(settings.execution_max_turnover_per_step),
            position_signal=positions,
        )
        paper = paper_trading_metrics(
            current_close=current[mask],
            target_close=split.target_close_test[mask],
            direction_up=direction_up,
            fee_bps=float(settings.execution_fee_bps),
            slippage_bps=float(settings.execution_slippage_bps),
            max_turnover_per_step=float(settings.execution_max_turnover_per_step),
            initial_capital=float(settings.paper_trade_initial_capital),
            position_signal=positions,
        )
        signal_rows.append(
            {
                "signal": signal,
                "expected_direction": "positive" if direction > 0 else "negative",
                "rationale": ALPHA_SIGNAL_RATIONALES.get(signal, ""),
                "sample_count": int(np.sum(mask)),
                "information_coefficient": corr,
                "decile_spread_bps": decile_spread_bps,
                "sign_alignment": alignment,
                "net_mean_return": float(execution.get("net_mean_return", math.nan)),
                "sharpe_net": float(execution.get("sharpe_net", math.nan)),
                "turnover_rate": float(execution.get("turnover_rate", math.nan)),
                "max_drawdown": float(paper.get("max_drawdown", math.nan)),
                "total_return": float(paper.get("total_return", math.nan)),
            }
        )

    if not signal_rows:
        return {"available": False, "reason": "no_alpha_signals_available"}
    avg_alignment = float(np.mean([float(row["sign_alignment"]) for row in signal_rows]))
    return {
        "available": True,
        "future_return_mean": float(np.mean(future_return)),
        "future_return_std": float(np.std(future_return)),
        "average_signal_alignment": avg_alignment,
        "signals": signal_rows,
    }


def _train_regime_models(split: SplitData, horizon: str) -> tuple[dict[str, Pipeline], dict[str, Pipeline]]:
    """Train regime models. Internal helper."""
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
                current_close_train=split.current_close_train[idx],
                current_close_val=split.current_close_val,
                current_close_test=split.current_close_test,
                target_close_train=split.target_close_train[idx],
                target_close_val=split.target_close_val,
                target_close_test=split.target_close_test,
                regime_train=split.regime_train,
                regime_val=split.regime_val,
                regime_test=split.regime_test,
                leakage_diagnostic=split.leakage_diagnostic,
                alpha_kill_diagnostics=split.alpha_kill_diagnostics,
            ),
            horizon,
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
                current_close_train=split.current_close_train[idx],
                current_close_val=split.current_close_val,
                current_close_test=split.current_close_test,
                target_close_train=split.target_close_train[idx],
                target_close_val=split.target_close_val,
                target_close_test=split.target_close_test,
                regime_train=split.regime_train,
                regime_val=split.regime_val,
                regime_test=split.regime_test,
                leakage_diagnostic=split.leakage_diagnostic,
                alpha_kill_diagnostics=split.alpha_kill_diagnostics,
            ),
            horizon,
        )
        cls_models[regime_name] = cls_best.model
        reg_models[regime_name] = reg_best.model
    return cls_models, reg_models


def _frame_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    """Internal helper to compute frame column."""
    if column not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    return frame[column].to_numpy(dtype=float)


def _predict_outputs_with_regime(
    *,
    x: pd.DataFrame,
    current_close: np.ndarray,
    regime_ids: np.ndarray,
    horizon: str,
    classifier_model: object,
    regressor_model: object,
    regression_target: str,
    regression_blend_alpha: float,
    directional_tilt_gamma: float,
    regime_cls_models: dict[str, Pipeline] | None = None,
    regime_reg_models: dict[str, Pipeline] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal helper to compute predict outputs with regime."""
    cls_prob = np.asarray(classifier_model.predict_proba(x)[:, 1], dtype=float)
    reg_raw = np.asarray(regressor_model.predict(x), dtype=float)
    reg_pred = regression_output_to_close(
        raw_prediction=reg_raw,
        current_close=current_close,
        horizon=horizon,
        regression_target=regression_target,
    )

    if regime_cls_models or regime_reg_models:
        regime_names = {0: "down", 1: "flat", 2: "up"}
        for regime_id, regime_name in regime_names.items():
            mask = regime_ids == regime_id
            if not np.any(mask):
                continue
            if regime_cls_models and regime_name in regime_cls_models:
                cls_prob[mask] = np.asarray(regime_cls_models[regime_name].predict_proba(x.iloc[mask])[:, 1], dtype=float)
            if regime_reg_models and regime_name in regime_reg_models:
                regime_raw = np.asarray(regime_reg_models[regime_name].predict(x.iloc[mask]), dtype=float)
                reg_pred[mask] = regression_output_to_close(
                    raw_prediction=regime_raw,
                    current_close=current_close[mask],
                    horizon=horizon,
                    regression_target=regression_target,
                )

    blend_alpha = float(np.clip(regression_blend_alpha, 0.0, 1.0))
    reg_pred = (blend_alpha * reg_pred) + ((1.0 - blend_alpha) * current_close)
    reg_pred = apply_directional_tilt_to_close(
        predicted_close=reg_pred,
        current_close=current_close,
        probability_up=cls_prob,
        volatility_feature=_frame_column(x, "volatility_20", default=0.0),
        horizon=horizon,
        gamma=float(directional_tilt_gamma),
    )
    return cls_prob, reg_pred


def _meta_feature_frame(
    *,
    x: pd.DataFrame,
    prob_up: np.ndarray,
    decision_threshold: float,
    predicted_close: np.ndarray,
    current_close: np.ndarray,
) -> pd.DataFrame:
    """Internal helper to compute meta feature frame."""
    safe_current = np.clip(np.asarray(current_close, dtype=float), 1e-12, None)
    pred_return = (np.asarray(predicted_close, dtype=float) / safe_current) - 1.0
    prob = np.asarray(prob_up, dtype=float)
    edge = np.abs(prob - float(decision_threshold))

    return pd.DataFrame(
        {
            "prob_up": prob,
            "prob_edge": edge,
            "pred_return": pred_return,
            "abs_pred_return": np.abs(pred_return),
            "volatility_20": _frame_column(x, "volatility_20", default=0.0),
            "markov_prob_down": _frame_column(x, "markov_prob_down", default=0.33),
            "markov_prob_flat": _frame_column(x, "markov_prob_flat", default=0.34),
            "markov_prob_up": _frame_column(x, "markov_prob_up", default=0.33),
        }
    )


def _meta_labels(
    *,
    prob_up: np.ndarray,
    decision_threshold: float,
    current_close: np.ndarray,
    target_close: np.ndarray,
    min_move_bps: float,
) -> np.ndarray:
    """Internal helper to compute meta labels."""
    direction_pred_up = np.asarray(prob_up, dtype=float) >= float(decision_threshold)
    direction_true_up = np.asarray(target_close, dtype=float) > np.asarray(current_close, dtype=float)
    realized_abs_return = np.abs((np.asarray(target_close, dtype=float) / np.clip(np.asarray(current_close, dtype=float), 1e-12, None)) - 1.0)
    min_move = max(float(min_move_bps), 0.0) / 10000.0
    labels = (direction_pred_up == direction_true_up) & (realized_abs_return >= min_move)
    return labels.astype(int)


def _select_meta_threshold(
    *,
    current_close: np.ndarray,
    target_close: np.ndarray,
    cls_prob: np.ndarray,
    meta_prob: np.ndarray,
    decision_threshold: float,
    horizon: str,
    min_take_rate: float,
    fee_bps: float,
    slippage_bps: float,
    max_turnover_per_step: float,
) -> tuple[float, float, float, float]:
    """Internal helper to compute select meta threshold."""
    best_threshold = 0.50
    best_take_rate = 0.0
    best_net_mean_return = -math.inf
    best_max_drawdown = -1.0
    min_take = float(np.clip(min_take_rate, 0.0, 1.0))
    direction_signal = np.where(np.asarray(cls_prob, dtype=float) >= float(decision_threshold), 1.0, -1.0)

    for threshold in np.linspace(0.35, 0.90, 56):
        action = (np.asarray(meta_prob, dtype=float) >= float(threshold)).astype(float)
        take_rate = float(np.mean(action)) if action.size else 0.0
        if take_rate < min_take:
            continue
        signed_positions = direction_signal * action
        execution = execution_aware_metrics(
            current_close=current_close,
            target_close=target_close,
            direction_up=(direction_signal > 0).astype(int),
            horizon=horizon,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_turnover_per_step=max_turnover_per_step,
            position_signal=signed_positions,
        )
        net_mean_return = float(execution.get("net_mean_return", math.nan))
        if not math.isfinite(net_mean_return):
            continue
        drawdown = float(execution.get("max_drawdown_net", -1.0))
        if (net_mean_return > best_net_mean_return) or (
            math.isclose(net_mean_return, best_net_mean_return, rel_tol=1e-12, abs_tol=1e-12)
            and drawdown > best_max_drawdown
        ):
            best_threshold = float(threshold)
            best_take_rate = take_rate
            best_net_mean_return = net_mean_return
            best_max_drawdown = drawdown

    return best_threshold, best_take_rate, best_net_mean_return, best_max_drawdown


def _train_meta_labeler(
    *,
    split: SplitData,
    horizon: str,
    decision_threshold: float,
    cls_prob_train: np.ndarray,
    cls_prob_val: np.ndarray,
    reg_pred_train: np.ndarray,
    reg_pred_val: np.ndarray,
) -> MetaLabelingResult:
    """Internal helper to compute train meta labeler."""
    settings = get_settings()
    min_move_bps = float(settings.meta_label_min_move_bps)
    y_train_meta = _meta_labels(
        prob_up=cls_prob_train,
        decision_threshold=decision_threshold,
        current_close=split.current_close_train,
        target_close=split.target_close_train,
        min_move_bps=min_move_bps,
    )
    if len(np.unique(y_train_meta)) < 2:
        return MetaLabelingResult(
            enabled=False,
            model=None,
            threshold=0.5,
            min_take_rate=float(settings.meta_label_min_take_rate),
            val_take_rate=0.0,
            val_net_mean_return=math.nan,
            val_max_drawdown=math.nan,
            reason="insufficient_meta_label_class_balance",
        )

    x_train_meta = _meta_feature_frame(
        x=split.x_train,
        prob_up=cls_prob_train,
        decision_threshold=decision_threshold,
        predicted_close=reg_pred_train,
        current_close=split.current_close_train,
    )
    x_val_meta = _meta_feature_frame(
        x=split.x_val,
        prob_up=cls_prob_val,
        decision_threshold=decision_threshold,
        predicted_close=reg_pred_val,
        current_close=split.current_close_val,
    )

    meta_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
        ]
    )
    meta_model.fit(x_train_meta, y_train_meta)
    val_meta_prob = np.asarray(meta_model.predict_proba(x_val_meta)[:, 1], dtype=float)
    threshold, take_rate, net_mean_return, max_drawdown = _select_meta_threshold(
        current_close=split.current_close_val,
        target_close=split.target_close_val,
        cls_prob=cls_prob_val,
        meta_prob=val_meta_prob,
        decision_threshold=decision_threshold,
        horizon=horizon,
        min_take_rate=float(settings.meta_label_min_take_rate),
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
    )
    if not math.isfinite(net_mean_return):
        return MetaLabelingResult(
            enabled=False,
            model=None,
            threshold=float(threshold),
            min_take_rate=float(settings.meta_label_min_take_rate),
            val_take_rate=float(take_rate),
            val_net_mean_return=math.nan,
            val_max_drawdown=math.nan,
            reason="meta_threshold_search_failed",
        )

    return MetaLabelingResult(
        enabled=True,
        model=meta_model,
        threshold=float(threshold),
        min_take_rate=float(settings.meta_label_min_take_rate),
        val_take_rate=float(take_rate),
        val_net_mean_return=float(net_mean_return),
        val_max_drawdown=float(max_drawdown),
        reason=None,
    )


def _conformal_quantile(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    """Internal helper to compute conformal quantile."""
    level = float(np.clip(alpha, 0.01, 0.40))
    residual = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if residual.size == 0:
        return 0.0, 0.0
    q_abs = float(np.quantile(residual, 1.0 - level))
    coverage = float(np.mean(residual <= q_abs))
    return max(q_abs, 0.0), float(np.clip(coverage, 0.0, 1.0))


def _edge_action_mask(
    *,
    cls_prob: np.ndarray,
    predicted_close: np.ndarray,
    current_close: np.ndarray,
    conformal_q_abs_usd: float,
    fee_bps: float,
    slippage_bps: float,
    uncertainty_buffer_mult: float,
    threshold_scale: float = 1.0,
    min_take_rate: float | None = None,
) -> np.ndarray:
    """Compute action mask from cost-adjusted expected edge."""
    settings = get_settings()
    prob = np.asarray(cls_prob, dtype=float)
    pred_close = np.asarray(predicted_close, dtype=float)
    current = np.clip(np.asarray(current_close, dtype=float), 1e-12, None)
    pred_return_abs = np.abs((pred_close / current) - 1.0)
    confidence_edge = np.clip(np.abs(prob - 0.5) * 2.0, 0.0, 1.0)
    expected_edge = pred_return_abs * confidence_edge
    per_step_cost = (float(fee_bps) + float(slippage_bps)) / 10000.0
    uncertainty = (float(conformal_q_abs_usd) / current) * float(max(uncertainty_buffer_mult, 0.0))
    threshold = (per_step_cost + uncertainty) * float(max(threshold_scale, 0.0))
    mask = expected_edge > threshold

    # If the strict edge filter fully abstains, relax conservatively to maintain a minimum
    # tradable sample for financial-gate evaluation.
    configured_min_take = settings.edge_action_min_take_rate if min_take_rate is None else min_take_rate
    min_take_rate = float(np.clip(configured_min_take, 0.0, 0.5))
    if mask.size == 0 or min_take_rate <= 0.0:
        return mask
    take_rate = float(np.mean(mask.astype(float)))
    if take_rate >= min_take_rate:
        return mask

    relaxed_threshold = per_step_cost * float(max(settings.edge_action_cost_relax_mult, 0.0))
    relaxed_mask = expected_edge > relaxed_threshold
    if not np.any(relaxed_mask):
        return mask

    needed = int(math.ceil(min_take_rate * float(mask.size)))
    candidate_idx = np.where(relaxed_mask)[0]
    if candidate_idx.size <= needed:
        top_idx = np.argsort(expected_edge)[::-1][:needed]
        out = np.zeros(mask.shape[0], dtype=bool)
        out[top_idx] = True
        return out

    candidate_scores = expected_edge[candidate_idx] - relaxed_threshold
    selected_local = np.argsort(candidate_scores)[::-1][:needed]
    selected_idx = candidate_idx[selected_local]
    out = np.zeros(mask.shape[0], dtype=bool)
    out[selected_idx] = True
    return out


def _baseline_partition_arrays(
    split: SplitData,
    partition: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Resolve split partition arrays used by baseline and tuning diagnostics."""
    label = partition.strip().lower()
    if label == "val":
        return split.x_val, split.current_close_val, split.target_close_val
    return split.x_test, split.current_close_test, split.target_close_test


def _signed_direction_from_policy(cls_pred: np.ndarray, policy: str) -> np.ndarray:
    """Map classifier direction into a trading-direction signal policy."""
    policy_label = policy.strip().lower()
    if policy_label in {"auto", "adaptive", "long_flat", "long_only"}:
        return np.where(cls_pred == 1, 1.0, 0.0)
    if policy_label == "short_only":
        return np.where(cls_pred == 1, 0.0, -1.0)
    return np.where(cls_pred == 1, 1.0, -1.0)


def _returns_objective_score(
    *,
    execution_metrics: dict[str, float],
    paper_metrics: dict[str, float],
    baseline_execution_metrics: dict[str, float],
    baseline_paper_metrics: dict[str, float],
    min_take_rate: float,
    take_rate: float,
    max_turnover_per_step: float,
    max_drawdown_limit: float,
) -> float:
    """Score financial quality relative to baseline without relaxing strict promotion rules."""
    model_net = float(execution_metrics.get("net_mean_return", math.nan))
    base_net = float(baseline_execution_metrics.get("net_mean_return", math.nan))
    model_sharpe = float(execution_metrics.get("sharpe_net", math.nan))
    base_sharpe = float(baseline_execution_metrics.get("sharpe_net", math.nan))
    model_turnover = float(execution_metrics.get("turnover_rate", math.nan))
    model_total = float(paper_metrics.get("total_return", math.nan))
    base_total = float(baseline_paper_metrics.get("total_return", math.nan))
    model_drawdown = float(paper_metrics.get("max_drawdown", math.nan))
    base_drawdown = float(baseline_paper_metrics.get("max_drawdown", math.nan))

    if not math.isfinite(model_net):
        model_net = -1.0
    if not math.isfinite(base_net):
        base_net = 0.0
    if not math.isfinite(model_sharpe):
        model_sharpe = -5.0
    if not math.isfinite(base_sharpe):
        base_sharpe = 0.0
    if not math.isfinite(model_total):
        model_total = -1.0
    if not math.isfinite(base_total):
        base_total = 0.0
    if not math.isfinite(model_drawdown):
        model_drawdown = -1.0
    if not math.isfinite(base_drawdown):
        base_drawdown = max_drawdown_limit
    if not math.isfinite(model_turnover):
        model_turnover = 1.0

    delta_net = model_net - base_net
    delta_sharpe = model_sharpe - base_sharpe
    delta_total = model_total - base_total
    drawdown_floor = max(float(max_drawdown_limit), float(base_drawdown))
    drawdown_margin = model_drawdown - drawdown_floor

    net_term = math.tanh(delta_net * 6000.0)
    sharpe_term = math.tanh(delta_sharpe / 2.0)
    total_term = math.tanh(delta_total / 2.0)
    drawdown_term = math.tanh(drawdown_margin / 0.12)

    take_penalty = max(0.0, float(min_take_rate) - float(take_rate))
    turnover_penalty = max(0.0, float(model_turnover) - float(max_turnover_per_step))

    return (
        (0.50 * net_term)
        + (0.22 * sharpe_term)
        + (0.18 * total_term)
        + (0.10 * drawdown_term)
        - (0.28 * take_penalty)
        - (0.20 * turnover_penalty)
    )


def _returns_tuning_threshold_candidates(prob: np.ndarray) -> np.ndarray:
    """Generate bounded decision-threshold candidates for returns tuning."""
    clipped = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    quantiles = np.quantile(clipped, np.linspace(0.10, 0.90, 9))
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.30, 0.70, 9),
                quantiles,
                np.array([0.25, 0.35, 0.50, 0.65, 0.75]),
            ]
        )
    )


def _baseline_strategy_book(
    split: SplitData,
    horizon: str,
    *,
    partition: str = "test",
) -> dict[str, dict[str, dict[str, float] | np.ndarray]]:
    """Evaluate baseline strategies on the requested split partition."""
    settings = get_settings()
    features, current_close, target_close = _baseline_partition_arrays(split, partition)
    vol = _frame_column(features, "volatility_20", default=0.0)
    vol_scale = _vol_target_position_scale(
        vol,
        horizon=horizon,
        target_annual_vol=float(settings.position_target_annual_vol),
        max_leverage=float(settings.position_max_leverage),
    )
    ret_1 = _frame_column(features, "ret_1", default=0.0)
    kalman_trend = _frame_column(features, "kalman_trend_5", default=0.0)
    always_long_signal = np.ones(len(features), dtype=float)
    if bool(settings.baseline_risk_matched):
        always_long_signal = np.asarray(vol_scale, dtype=float)
    signals: dict[str, np.ndarray] = {
        "always_long": always_long_signal,
        "vol_targeted_momentum": np.sign(ret_1),
        "vol_targeted_mean_reversion": -np.sign(ret_1),
        "vol_targeted_kalman_trend": np.sign(kalman_trend),
    }
    payload: dict[str, dict[str, dict[str, float] | np.ndarray]] = {}
    for name, base_signal in signals.items():
        position = np.asarray(base_signal, dtype=float) * vol_scale
        direction_up = (position >= 0).astype(int)
        execution = execution_aware_metrics(
            current_close=current_close,
            target_close=target_close,
            direction_up=direction_up,
            horizon=horizon,
            fee_bps=float(settings.execution_fee_bps),
            slippage_bps=float(settings.execution_slippage_bps),
            max_turnover_per_step=float(settings.execution_max_turnover_per_step),
            position_signal=position,
        )
        paper = paper_trading_metrics(
            current_close=current_close,
            target_close=target_close,
            direction_up=direction_up,
            fee_bps=float(settings.execution_fee_bps),
            slippage_bps=float(settings.execution_slippage_bps),
            max_turnover_per_step=float(settings.execution_max_turnover_per_step),
            initial_capital=float(settings.paper_trade_initial_capital),
            position_signal=position,
        )
        payload[name] = {"execution": execution, "paper": paper, "position_signal": position}
    return payload


def _best_strategy_name(strategy_book: dict[str, dict[str, dict[str, float] | np.ndarray]]) -> str:
    """Pick highest-scoring baseline strategy within a provided strategy book."""
    best_name = "always_long"
    best_key = (-math.inf, -math.inf, -math.inf)
    for name, item in strategy_book.items():
        execution = item.get("execution", {})
        paper = item.get("paper", {})
        if not isinstance(execution, dict) or not isinstance(paper, dict):
            continue
        net = float(execution.get("net_mean_return", math.nan))
        sharpe = float(execution.get("sharpe_net", math.nan))
        drawdown = float(paper.get("max_drawdown", -1.0))
        if not math.isfinite(net):
            continue
        if not math.isfinite(sharpe):
            sharpe = -math.inf
        if not math.isfinite(drawdown):
            drawdown = -1.0
        key = (net, sharpe, drawdown)
        if key > best_key:
            best_key = key
            best_name = name
    return best_name


def _select_financial_baseline(
    strategy_book_test: dict[str, dict[str, dict[str, float] | np.ndarray]],
    strategy_book_val: dict[str, dict[str, dict[str, float] | np.ndarray]] | None = None,
) -> tuple[str, dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Select strict baseline strategy with configurable non-oracle policy."""
    settings = get_settings()
    mode = settings.financial_baseline_selection_mode.strip().lower()
    preferred = settings.financial_baseline_strategy.strip().lower()

    selected_name = "always_long"
    if mode in {"best_validation", "best_val"} and isinstance(strategy_book_val, dict) and strategy_book_val:
        selected_name = _best_strategy_name(strategy_book_val)
    elif mode in {"best_expost", "best", "best_test"}:
        selected_name = _best_strategy_name(strategy_book_test)
    else:
        if preferred in strategy_book_test:
            selected_name = preferred
        elif "always_long" in strategy_book_test:
            selected_name = "always_long"
        elif strategy_book_test:
            selected_name = sorted(strategy_book_test.keys())[0]

    test_item = strategy_book_test.get(selected_name, {})
    if not isinstance(test_item, dict):
        test_item = {}
    val_item = strategy_book_val.get(selected_name, {}) if isinstance(strategy_book_val, dict) else {}
    if not isinstance(val_item, dict):
        val_item = {}

    test_exec = test_item.get("execution", {})
    test_paper = test_item.get("paper", {})
    if not isinstance(test_exec, dict):
        test_exec = {}
    if not isinstance(test_paper, dict):
        test_paper = {}

    val_exec = val_item.get("execution", {})
    val_paper = val_item.get("paper", {})
    if not isinstance(val_exec, dict):
        val_exec = test_exec
    if not isinstance(val_paper, dict):
        val_paper = test_paper

    return selected_name, test_exec, test_paper, val_exec, val_paper


def _returns_tuning_policies() -> list[str]:
    """Resolve tuning policy list from settings."""
    settings = get_settings()
    configured = [item.strip().lower() for item in settings.returns_tuning_policies.split(",") if item.strip()]
    defaults = ["long_flat", "long_short", "long_only"]
    if not configured:
        return defaults
    valid = [item for item in configured if item in {"long_flat", "long_only", "long_short", "short_only"}]
    return valid or defaults


def _returns_tuning_position_scale_candidates() -> list[float]:
    """Resolve position-scale candidates for returns tuning."""
    settings = get_settings()
    raw_values = [item.strip() for item in settings.returns_tuning_position_scale_candidates.split(",") if item.strip()]
    parsed: list[float] = []
    for item in raw_values:
        try:
            value = float(item)
        except ValueError:
            continue
        if value <= 0:
            continue
        parsed.append(float(np.clip(value, 0.10, 3.0)))
    if not parsed:
        parsed = [0.5, 0.75, 1.0, 1.25, 1.5]
    return sorted(set(parsed))


def _tune_returns_profile(
    *,
    horizon: str,
    cls_prob_val: np.ndarray,
    reg_pred_val: np.ndarray,
    current_close_val: np.ndarray,
    target_close_val: np.ndarray,
    vol_scale_val: np.ndarray,
    conformal_q_abs_usd: float,
    baseline_execution_metrics: dict[str, float],
    baseline_paper_metrics: dict[str, float],
    default_threshold: float,
    default_policy: str,
    default_edge_threshold_scale: float,
    default_position_scale: float,
) -> ReturnsTuningResult:
    """Jointly tune threshold, edge scale, and direction policy for financial objectives."""
    settings = get_settings()
    threshold_candidates = _returns_tuning_threshold_candidates(cls_prob_val)
    edge_scale_candidates = np.array([0.35, 0.50, 0.70, 0.85, 1.00, 1.20], dtype=float)
    policy_default = default_policy.strip().lower()
    if policy_default in {"", "auto", "adaptive"}:
        policy_candidates = _returns_tuning_policies()
    else:
        policy_candidates = [policy_default]
        for candidate in _returns_tuning_policies():
            if candidate not in policy_candidates:
                policy_candidates.append(candidate)

    best = ReturnsTuningResult(
        decision_threshold=float(default_threshold),
        edge_threshold_scale=float(default_edge_threshold_scale),
        direction_policy=(policy_candidates[0] if policy_candidates else "long_flat"),
        position_scale=float(np.clip(default_position_scale, 0.10, 3.0)),
        objective=-math.inf,
        diagnostics={},
    )
    diagnostics_rows: list[dict[str, object]] = []
    min_take_rate = float(np.clip(settings.edge_action_min_take_rate, 0.0, 0.5))
    position_scale_candidates = _returns_tuning_position_scale_candidates()

    for threshold in threshold_candidates:
        cls_pred_val = (np.asarray(cls_prob_val, dtype=float) >= float(threshold)).astype(int)
        for edge_scale in edge_scale_candidates:
            edge_mask_val = _edge_action_mask(
                cls_prob=cls_prob_val,
                predicted_close=reg_pred_val,
                current_close=current_close_val,
                conformal_q_abs_usd=conformal_q_abs_usd,
                fee_bps=float(settings.execution_fee_bps),
                slippage_bps=float(settings.execution_slippage_bps),
                uncertainty_buffer_mult=float(settings.trade_edge_uncertainty_buffer_mult),
                threshold_scale=float(edge_scale),
                min_take_rate=min_take_rate,
            )
            take_rate = float(np.mean(edge_mask_val.astype(float))) if edge_mask_val.size else 0.0
            for direction_policy in policy_candidates:
                signed_direction = _signed_direction_from_policy(cls_pred_val, direction_policy)
                for position_scale in position_scale_candidates:
                    positions = signed_direction * edge_mask_val.astype(float) * np.asarray(vol_scale_val, dtype=float) * float(position_scale)
                    execution = execution_aware_metrics(
                        current_close=current_close_val,
                        target_close=target_close_val,
                        direction_up=cls_pred_val,
                        horizon=horizon,
                        fee_bps=float(settings.execution_fee_bps),
                        slippage_bps=float(settings.execution_slippage_bps),
                        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
                        position_signal=positions,
                    )
                    paper = paper_trading_metrics(
                        current_close=current_close_val,
                        target_close=target_close_val,
                        direction_up=cls_pred_val,
                        fee_bps=float(settings.execution_fee_bps),
                        slippage_bps=float(settings.execution_slippage_bps),
                        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
                        initial_capital=float(settings.paper_trade_initial_capital),
                        position_signal=positions,
                    )
                    objective = _returns_objective_score(
                        execution_metrics=execution,
                        paper_metrics=paper,
                        baseline_execution_metrics=baseline_execution_metrics,
                        baseline_paper_metrics=baseline_paper_metrics,
                        min_take_rate=min_take_rate,
                        take_rate=take_rate,
                        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
                        max_drawdown_limit=float(settings.promotion_max_drawdown_limit),
                    )
                    diagnostics_rows.append(
                        {
                            "decision_threshold": float(threshold),
                            "edge_threshold_scale": float(edge_scale),
                            "direction_policy": direction_policy,
                            "position_scale": float(position_scale),
                            "take_rate": take_rate,
                            "objective": float(objective),
                            "net_mean_return": float(execution.get("net_mean_return", math.nan)),
                            "baseline_net_mean_return": float(baseline_execution_metrics.get("net_mean_return", math.nan)),
                            "sharpe_net": float(execution.get("sharpe_net", math.nan)),
                            "baseline_sharpe_net": float(baseline_execution_metrics.get("sharpe_net", math.nan)),
                            "total_return": float(paper.get("total_return", math.nan)),
                            "baseline_total_return": float(baseline_paper_metrics.get("total_return", math.nan)),
                        }
                    )
                    if objective > best.objective:
                        best = ReturnsTuningResult(
                            decision_threshold=float(threshold),
                            edge_threshold_scale=float(edge_scale),
                            direction_policy=direction_policy,
                            position_scale=float(position_scale),
                            objective=float(objective),
                            diagnostics={
                                "take_rate": take_rate,
                                "execution": execution,
                                "paper": paper,
                            },
                        )

    top_rows = sorted(diagnostics_rows, key=lambda item: float(item.get("objective", -math.inf)), reverse=True)[:12]
    return ReturnsTuningResult(
        decision_threshold=float(best.decision_threshold),
        edge_threshold_scale=float(best.edge_threshold_scale),
        direction_policy=str(best.direction_policy),
        position_scale=float(best.position_scale),
        objective=float(best.objective),
        diagnostics={
            **best.diagnostics,
            "top_candidates": top_rows,
        },
    )


def evaluate_symbol_horizon(
    symbol: str,
    spec: HorizonSpec,
    model_version: str,
    models_root: Path,
    write_artifacts: bool = True,
    as_of_cutoff_by_granularity: dict[str, int] | None = None,
    sync_row_depth_by_granularity: dict[str, int] | None = None,
) -> dict[str, object]:
    """Evaluate symbol horizon."""
    min_samples = min_samples_for_horizon(spec.label)
    resolved = resolve_horizon_data(
        symbol,
        spec,
        min_samples=min_samples,
        as_of_cutoff_by_granularity=as_of_cutoff_by_granularity,
        sync_row_depth_by_granularity=sync_row_depth_by_granularity,
    )
    if resolved is None:
        return {
            "symbol": symbol,
            "horizon": spec.label,
            "status": "insufficient_data",
            "reason": "no_granularity_meets_minimum",
            "required_min_samples": min_samples,
        }

    granularity = resolved.granularity
    steps = resolved.steps_ahead
    frame = resolved.frame
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

    cls_best = _select_best_classifier(split, spec.label)
    reg_best = _select_best_regressor(split, spec.label)
    settings = get_settings()
    regime_cls_models: dict[str, Pipeline] = {}
    regime_reg_models: dict[str, Pipeline] = {}
    if settings.regime_models_enabled:
        regime_cls_models, regime_reg_models = _train_regime_models(split, spec.label)

    reg_blend_alpha = float(np.clip(reg_best.regression_blend_alpha, 0.0, 1.0))
    val_cls_prob_pretilt, val_reg_pred_pretilt = _predict_outputs_with_regime(
        x=split.x_val,
        current_close=split.current_close_val,
        regime_ids=split.regime_val,
        horizon=spec.label,
        classifier_model=cls_best.model,
        regressor_model=reg_best.model,
        regression_target=reg_best.regression_target,
        regression_blend_alpha=reg_blend_alpha,
        directional_tilt_gamma=0.0,
        regime_cls_models=regime_cls_models,
        regime_reg_models=regime_reg_models,
    )
    val_rmse_before_tilt = float(math.sqrt(mean_squared_error(split.target_close_val, val_reg_pred_pretilt)))
    directional_tilt_gamma, val_rmse_after_tilt = _best_directional_tilt_gamma(
        y_true=split.target_close_val,
        current_close=split.current_close_val,
        base_pred_close=val_reg_pred_pretilt,
        prob_up=val_cls_prob_pretilt,
        volatility_20=_frame_column(split.x_val, "volatility_20", default=0.0),
        horizon=spec.label,
    )
    val_cls_prob, val_reg_pred = _predict_outputs_with_regime(
        x=split.x_val,
        current_close=split.current_close_val,
        regime_ids=split.regime_val,
        horizon=spec.label,
        classifier_model=cls_best.model,
        regressor_model=reg_best.model,
        regression_target=reg_best.regression_target,
        regression_blend_alpha=reg_blend_alpha,
        directional_tilt_gamma=directional_tilt_gamma,
        regime_cls_models=regime_cls_models,
        regime_reg_models=regime_reg_models,
    )

    conformal_q_abs_usd, conformal_val_coverage = _conformal_quantile(
        y_true=split.target_close_val,
        y_pred=val_reg_pred,
        alpha=float(settings.conformal_alpha),
    )

    strategy_baselines_val = _baseline_strategy_book(split, spec.label, partition="val")
    strategy_baselines = _baseline_strategy_book(split, spec.label, partition="test")
    (
        financial_baseline_name,
        baseline_execution_metrics,
        baseline_paper_trading,
        baseline_execution_metrics_val,
        baseline_paper_trading_val,
    ) = _select_financial_baseline(strategy_baselines, strategy_baselines_val)

    decision_threshold = float(np.clip(cls_best.decision_threshold, 0.02, 0.98))
    edge_threshold_scale = float(np.clip(settings.edge_action_threshold_scale, 0.05, 3.0))
    position_scale = 1.0
    direction_policy = settings.trade_direction_policy.strip().lower() or "auto"
    returns_tuning: ReturnsTuningResult | None = None
    if bool(settings.returns_tuning_enabled):
        vol_scale_val = _vol_target_position_scale(
            _frame_column(split.x_val, "volatility_20", default=0.0),
            horizon=spec.label,
            target_annual_vol=float(settings.position_target_annual_vol),
            max_leverage=float(settings.position_max_leverage),
        )
        returns_tuning = _tune_returns_profile(
            horizon=spec.label,
            cls_prob_val=val_cls_prob,
            reg_pred_val=val_reg_pred,
            current_close_val=split.current_close_val,
            target_close_val=split.target_close_val,
            vol_scale_val=vol_scale_val,
            conformal_q_abs_usd=conformal_q_abs_usd,
            baseline_execution_metrics=baseline_execution_metrics_val,
            baseline_paper_metrics=baseline_paper_trading_val,
            default_threshold=decision_threshold,
            default_policy=direction_policy,
            default_edge_threshold_scale=edge_threshold_scale,
            default_position_scale=position_scale,
        )
        if math.isfinite(float(returns_tuning.objective)):
            decision_threshold = float(np.clip(returns_tuning.decision_threshold, 0.02, 0.98))
            edge_threshold_scale = float(np.clip(returns_tuning.edge_threshold_scale, 0.05, 3.0))
            position_scale = float(np.clip(returns_tuning.position_scale, 0.10, 3.0))
            direction_policy = returns_tuning.direction_policy

    cls_prob, reg_pred = _predict_outputs_with_regime(
        x=split.x_test,
        current_close=split.current_close_test,
        regime_ids=split.regime_test,
        horizon=spec.label,
        classifier_model=cls_best.model,
        regressor_model=reg_best.model,
        regression_target=reg_best.regression_target,
        regression_blend_alpha=reg_blend_alpha,
        directional_tilt_gamma=directional_tilt_gamma,
        regime_cls_models=regime_cls_models,
        regime_reg_models=regime_reg_models,
    )
    cls_pred = (cls_prob >= decision_threshold).astype(int)

    cls_prob_train, reg_pred_train = _predict_outputs_with_regime(
        x=split.x_train,
        current_close=split.current_close_train,
        regime_ids=split.regime_train,
        horizon=spec.label,
        classifier_model=cls_best.model,
        regressor_model=reg_best.model,
        regression_target=reg_best.regression_target,
        regression_blend_alpha=reg_blend_alpha,
        directional_tilt_gamma=directional_tilt_gamma,
        regime_cls_models=regime_cls_models,
        regime_reg_models=regime_reg_models,
    )

    meta_result = MetaLabelingResult(
        enabled=False,
        model=None,
        threshold=0.5,
        min_take_rate=float(settings.meta_label_min_take_rate),
        val_take_rate=1.0,
        val_net_mean_return=math.nan,
        val_max_drawdown=math.nan,
        reason="meta_labeling_disabled",
    )
    if bool(settings.meta_labeling_enabled):
        meta_result = _train_meta_labeler(
            split=split,
            horizon=spec.label,
            decision_threshold=float(decision_threshold),
            cls_prob_train=cls_prob_train,
            cls_prob_val=val_cls_prob,
            reg_pred_train=reg_pred_train,
            reg_pred_val=val_reg_pred,
        )

    if meta_result.enabled and meta_result.model is not None:
        x_test_meta = _meta_feature_frame(
            x=split.x_test,
            prob_up=cls_prob,
            decision_threshold=float(decision_threshold),
            predicted_close=reg_pred,
            current_close=split.current_close_test,
        )
        meta_prob_test = np.asarray(meta_result.model.predict_proba(x_test_meta)[:, 1], dtype=float)
        action_mask_test = meta_prob_test >= float(meta_result.threshold)
    else:
        meta_prob_test = np.ones(len(split.x_test), dtype=float)
        action_mask_test = np.ones(len(split.x_test), dtype=bool)

    conformal_test_coverage = (
        float(np.mean(np.abs(split.target_close_test - reg_pred) <= conformal_q_abs_usd))
        if conformal_q_abs_usd > 0
        else 0.0
    )

    metrics = _metrics(
        y_true_cls=split.y_test_cls,
        y_pred_cls=cls_pred,
        y_proba_cls=cls_prob,
        y_true_reg=split.target_close_test,
        y_pred_reg=reg_pred,
    )
    metric_confidence_intervals = bootstrap_metric_confidence_intervals(
        y_true_cls=split.y_test_cls,
        y_pred_cls=cls_pred,
        y_true_reg=split.target_close_test,
        y_pred_reg=reg_pred,
        n_bootstrap=max(int(settings.metric_ci_bootstrap_samples), 80),
        confidence=float(settings.metric_ci_level),
        random_seed=42,
    )
    confidence_slice = _confidence_slice_metrics(
        y_true_cls=split.y_test_cls,
        y_pred_cls=cls_pred,
        y_proba_cls=cls_prob,
        threshold=float(settings.high_confidence_threshold),
    )
    edge_action_mask = _edge_action_mask(
        cls_prob=cls_prob,
        predicted_close=reg_pred,
        current_close=split.current_close_test,
        conformal_q_abs_usd=conformal_q_abs_usd,
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        uncertainty_buffer_mult=float(settings.trade_edge_uncertainty_buffer_mult),
        threshold_scale=edge_threshold_scale,
    )
    vol_scale = _vol_target_position_scale(
        _frame_column(split.x_test, "volatility_20", default=0.0),
        horizon=spec.label,
        target_annual_vol=float(settings.position_target_annual_vol),
        max_leverage=float(settings.position_max_leverage),
    )
    vol_scale = np.asarray(vol_scale, dtype=float) * float(position_scale)
    signed_direction = _signed_direction_from_policy(cls_pred, direction_policy)
    signed_positions_no_meta = signed_direction * edge_action_mask.astype(float) * vol_scale
    signed_positions_test = signed_direction * action_mask_test.astype(float) * edge_action_mask.astype(float) * vol_scale

    regime_breakdown = _regime_metrics(
        split=split,
        y_pred_cls=cls_pred,
        horizon=spec.label,
        position_signal=signed_positions_test,
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
        initial_capital=float(settings.paper_trade_initial_capital),
    )
    alpha_signal_diag = _alpha_signal_diagnostics(split, spec.label)
    baseline = _baseline_metrics(
        y_true_cls=split.y_test_cls,
        y_true_reg=split.target_close_test,
        current_close=split.current_close_test,
    )
    execution_metrics = execution_aware_metrics(
        current_close=split.current_close_test,
        target_close=split.target_close_test,
        direction_up=cls_pred,
        horizon=spec.label,
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
        position_signal=signed_positions_test,
    )
    paper_trading = paper_trading_metrics(
        current_close=split.current_close_test,
        target_close=split.target_close_test,
        direction_up=cls_pred,
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
        initial_capital=float(settings.paper_trade_initial_capital),
        position_signal=signed_positions_test,
    )
    execution_stress = execution_stress_metrics(
        current_close=split.current_close_test,
        target_close=split.target_close_test,
        direction_up=cls_pred,
        horizon=spec.label,
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
        initial_capital=float(settings.paper_trade_initial_capital),
        position_signal=signed_positions_test,
    )
    no_meta_execution_metrics = execution_aware_metrics(
        current_close=split.current_close_test,
        target_close=split.target_close_test,
        direction_up=cls_pred,
        horizon=spec.label,
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
        position_signal=signed_positions_no_meta,
    )
    no_meta_paper_trading = paper_trading_metrics(
        current_close=split.current_close_test,
        target_close=split.target_close_test,
        direction_up=cls_pred,
        fee_bps=float(settings.execution_fee_bps),
        slippage_bps=float(settings.execution_slippage_bps),
        max_turnover_per_step=float(settings.execution_max_turnover_per_step),
        initial_capital=float(settings.paper_trade_initial_capital),
        position_signal=signed_positions_no_meta,
    )
    walk_forward_gate = strict_gate_walk_forward_diagnostic(
        y_true_cls=split.y_test_cls,
        y_prob_cls=cls_prob,
        y_true_reg=split.target_close_test,
        y_pred_reg=reg_pred,
        baseline_reg=split.current_close_test,
        decision_threshold=float(decision_threshold),
        folds=max(int(settings.walk_forward_gate_folds), 2),
    )
    walk_forward_mode = settings.walk_forward_gate_mode.strip().lower()
    walk_forward_enforced = walk_forward_mode == "strict"
    martingale_diag = _martingale_residual_diagnostic(split.target_close_test, reg_pred)
    martingale_enforced = settings.martingale_gate_mode.strip().lower() == "strict"
    walk_forward_pass = True
    leakage_pass = bool(split.leakage_diagnostic.get("pass", False))
    if walk_forward_enforced:
        walk_forward_pass = bool(walk_forward_gate.get("enabled")) and bool(walk_forward_gate.get("strict_pass_all_folds"))

    model_net_mean_return = float(execution_metrics.get("net_mean_return", math.nan))
    baseline_net_mean_return = float(baseline_execution_metrics.get("net_mean_return", math.nan))
    model_sharpe_net = float(execution_metrics.get("sharpe_net", math.nan))
    baseline_sharpe_net = float(baseline_execution_metrics.get("sharpe_net", math.nan))
    model_total_return = float(paper_trading.get("total_return", math.nan))
    baseline_total_return = float(baseline_paper_trading.get("total_return", math.nan))

    max_drawdown_limit = float(settings.promotion_max_drawdown_limit)
    model_max_drawdown = float(paper_trading.get("max_drawdown", math.nan))
    baseline_max_drawdown = float(baseline_paper_trading.get("max_drawdown", math.nan))
    drawdown_floor = max_drawdown_limit
    if math.isfinite(baseline_max_drawdown):
        drawdown_floor = max(drawdown_floor, baseline_max_drawdown)
    drawdown_gate_pass = math.isfinite(model_max_drawdown) and (model_max_drawdown >= drawdown_floor)

    returns_gate_vs_baseline = (
        math.isfinite(model_net_mean_return)
        and math.isfinite(baseline_net_mean_return)
        and (model_net_mean_return > baseline_net_mean_return)
    )
    returns_gate_positive = math.isfinite(model_net_mean_return) and (model_net_mean_return > 0.0)
    returns_gate_pass = returns_gate_vs_baseline
    if bool(settings.promotion_require_positive_net_return):
        returns_gate_pass = returns_gate_pass and returns_gate_positive

    sharpe_gate_pass = True
    if bool(settings.promotion_require_sharpe_above_baseline):
        sharpe_gate_pass = (
            math.isfinite(model_sharpe_net)
            and math.isfinite(baseline_sharpe_net)
            and (model_sharpe_net > baseline_sharpe_net)
            and (model_sharpe_net >= float(settings.promotion_min_sharpe_net))
        )

    total_return_gate_pass = True
    if bool(settings.promotion_require_total_return_above_baseline):
        total_return_gate_pass = (
            math.isfinite(model_total_return)
            and math.isfinite(baseline_total_return)
            and (model_total_return > baseline_total_return)
        )

    classification_gate_pass = (metrics["f1"] > baseline["f1"]) and (metrics["accuracy"] > baseline["accuracy"])
    regression_gate_pass = metrics["rmse"] < baseline["rmse"]
    require_classification = bool(settings.promotion_require_classification_edge)
    require_regression = bool(settings.promotion_require_regression_edge)
    gate_mode = settings.promotion_gate_mode.strip().lower()
    returns_first_mode = gate_mode in {"returns_first", "returns-first", "financial"}

    if returns_first_mode:
        promotion_pass = (
            returns_gate_pass
            and sharpe_gate_pass
            and total_return_gate_pass
            and drawdown_gate_pass
            and leakage_pass
            and walk_forward_pass
            and (bool(martingale_diag["pass"]) if martingale_enforced else True)
            and ((classification_gate_pass) if require_classification else True)
            and ((regression_gate_pass) if require_regression else True)
        )
    else:
        promotion_pass = (
            classification_gate_pass
            and regression_gate_pass
            and returns_gate_pass
            and drawdown_gate_pass
            and leakage_pass
            and walk_forward_pass
            and (bool(martingale_diag["pass"]) if martingale_enforced else True)
        )

    failed_reasons: list[str] = []
    if require_classification and not (metrics["f1"] > baseline["f1"]):
        failed_reasons.append("classification_f1_not_above_baseline")
    if require_classification and not (metrics["accuracy"] > baseline["accuracy"]):
        failed_reasons.append("classification_accuracy_not_above_baseline")
    if require_regression and not (metrics["rmse"] < baseline["rmse"]):
        failed_reasons.append("regression_rmse_not_below_baseline")
    if not returns_gate_vs_baseline:
        failed_reasons.append("execution_net_mean_return_not_above_baseline")
    if bool(settings.promotion_require_positive_net_return) and not returns_gate_positive:
        failed_reasons.append("execution_net_mean_return_not_positive")
    if bool(settings.promotion_require_sharpe_above_baseline) and not sharpe_gate_pass:
        failed_reasons.append("execution_sharpe_not_above_baseline")
    if bool(settings.promotion_require_total_return_above_baseline) and not total_return_gate_pass:
        failed_reasons.append("paper_trading_total_return_not_above_baseline")
    if not drawdown_gate_pass:
        failed_reasons.append("paper_trading_max_drawdown_below_limit")
    if not leakage_pass:
        failed_reasons.append("data_leakage_check_not_passed")
    if walk_forward_enforced and not walk_forward_pass:
        failed_reasons.append("walk_forward_gate_not_passed")
    if martingale_enforced and (not bool(martingale_diag["pass"])):
        failed_reasons.append("martingale_residual_autocorrelation_too_high")

    calibration = _build_calibration_payload(split.y_test_cls, cls_prob)
    calibration["decision_threshold"] = float(decision_threshold)
    calibration["trade_direction_policy"] = str(direction_policy)
    calibration["edge_threshold_scale"] = float(edge_threshold_scale)
    calibration["position_scale"] = float(position_scale)
    calibration["conformal_alpha"] = float(np.clip(float(settings.conformal_alpha), 0.01, 0.40))
    calibration["conformal_q_abs_usd"] = float(conformal_q_abs_usd)
    calibration["conformal_val_coverage"] = float(conformal_val_coverage)
    if meta_result.enabled:
        calibration["meta_decision_threshold"] = float(meta_result.threshold)

    symbol_dir = models_root / model_version / symbol
    artifact_paths = {
        "classification": symbol_dir / f"cls_{spec.label}.joblib",
        "regression": symbol_dir / f"reg_{spec.label}.joblib",
        "calibration": symbol_dir / f"calibration_{spec.label}.json",
        "metrics": symbol_dir / f"metrics_{spec.label}.json",
    }
    meta_artifact_path = symbol_dir / f"meta_{spec.label}.joblib"
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
        "rows_input_before_feature_pipeline": int(len(frame)),
        "archive_rows_available_before_horizon_window": int(resolved.archive_rows_before_window),
        "rows_test": int(len(split.x_test)),
        "as_of_cutoff_time": (
            int(as_of_cutoff_by_granularity.get(granularity))
            if isinstance(as_of_cutoff_by_granularity, dict) and granularity in as_of_cutoff_by_granularity
            else None
        ),
        "sync_row_depth": (
            int(sync_row_depth_by_granularity.get(granularity))
            if isinstance(sync_row_depth_by_granularity, dict) and granularity in sync_row_depth_by_granularity
            else None
        ),
        "train_start_time": int(frame["start_time"].iloc[0]) if len(frame) else None,
        "train_end_time": int(frame["start_time"].iloc[-1]) if len(frame) else None,
        "archive_start_time_before_horizon_window": resolved.archive_start_time,
        "archive_end_time_before_horizon_window": resolved.archive_end_time,
        "horizon_training_window": {
            "enabled": bool(settings.train_horizon_windowing_enabled),
            "rows_limit": resolved.horizon_window_rows_limit,
            "trim_applied": bool(resolved.horizon_window_trim_applied),
            "archive_rows_before_window": int(resolved.archive_rows_before_window),
            "rows_used_after_window": int(len(frame)),
        },
        "selected_models": {
            "classifier": cls_best.model_name,
            "regressor": reg_best.model_name,
            "meta_labeler": ("logistic_regression_meta" if meta_result.enabled else "disabled"),
            "classifier_down_weight_boost": float(cls_best.class_down_weight_boost),
            "classifier_threshold_tuning": cls_best.threshold_tuning or {},
            "classifier_stability": cls_best.stability or {},
            "decision_threshold_runtime": float(decision_threshold),
            "trade_direction_policy_runtime": str(direction_policy),
            "edge_threshold_scale_runtime": float(edge_threshold_scale),
            "position_scale_runtime": float(position_scale),
            "regression_target": reg_best.regression_target,
            "regression_blend_alpha": reg_blend_alpha,
            "regression_stability": reg_best.stability or {},
            "directional_tilt_gamma": float(directional_tilt_gamma),
        },
        "feature_columns": split.feature_columns,
        "alpha_kill_diagnostics": split.alpha_kill_diagnostics,
        "metrics": metrics,
        "baseline": baseline,
        "near_pass_delta": _near_pass_delta(metrics, baseline),
        "martingale_diagnostic": martingale_diag,
        "martingale_enforced": martingale_enforced,
        "regression_target": reg_best.regression_target,
        "regression_output_transform": (
            "predicted_close = current_close * exp(clipped_log_return)"
            if reg_best.regression_target == "log_return"
            else "predicted_close = current_close + clipped_residual"
        ),
        "regression_blend_alpha": reg_blend_alpha,
        "directional_tilt_gamma": float(directional_tilt_gamma),
        "validation_rmse_before_tilt": val_rmse_before_tilt,
        "validation_rmse_after_tilt": val_rmse_after_tilt,
        "classification_label_mode": settings.classification_label_mode,
        "triple_barrier_sigma_mult": float(settings.triple_barrier_sigma_mult),
        "regime_models_enabled": bool(settings.regime_models_enabled),
        "log_return_clip": float(HORIZON_LOG_RETURN_CLIP.get(spec.label, 1.0)),
        "confidence_slice": confidence_slice,
        "regime_breakdown": regime_breakdown,
        "alpha_signal_diagnostics": alpha_signal_diag,
        "metric_confidence_intervals": metric_confidence_intervals,
        "data_leakage_checks": split.leakage_diagnostic,
        "execution_aware_metrics": execution_metrics,
        "execution_stress_metrics": execution_stress,
        "baseline_execution_aware_metrics": baseline_execution_metrics,
        "paper_trading_metrics": paper_trading,
        "baseline_paper_trading_metrics": baseline_paper_trading,
        "financial_baseline_strategy": financial_baseline_name,
        "financial_baseline_selection_mode": settings.financial_baseline_selection_mode,
        "financial_baseline_candidates": {
            name: {
                "execution": (item.get("execution") if isinstance(item, dict) else {}),
                "paper": (item.get("paper") if isinstance(item, dict) else {}),
            }
            for name, item in strategy_baselines.items()
        },
        "financial_baseline_candidates_validation": {
            name: {
                "execution": (item.get("execution") if isinstance(item, dict) else {}),
                "paper": (item.get("paper") if isinstance(item, dict) else {}),
            }
            for name, item in strategy_baselines_val.items()
        },
        "returns_tuning": (
            {
                "enabled": True,
                "objective": float(returns_tuning.objective),
                "decision_threshold": float(returns_tuning.decision_threshold),
                "edge_threshold_scale": float(returns_tuning.edge_threshold_scale),
                "trade_direction_policy": returns_tuning.direction_policy,
                "position_scale": float(returns_tuning.position_scale),
                "diagnostics": returns_tuning.diagnostics,
            }
            if returns_tuning is not None
            else {"enabled": False}
        ),
        "ablation_metrics": {
            "with_meta_abstention": {
                "execution": execution_metrics,
                "paper": paper_trading,
            },
            "without_meta_abstention": {
                "execution": no_meta_execution_metrics,
                "paper": no_meta_paper_trading,
            },
            "edge_filter_take_rate": float(np.mean(edge_action_mask.astype(float))) if len(edge_action_mask) else 0.0,
        },
        "walk_forward_gate": walk_forward_gate,
        "walk_forward_mode": walk_forward_mode,
        "walk_forward_enforced": walk_forward_enforced,
        "meta_labeling": {
            "enabled": bool(meta_result.enabled),
            "reason": meta_result.reason,
            "decision_threshold": float(meta_result.threshold),
            "min_take_rate": float(meta_result.min_take_rate),
            "val_take_rate": float(meta_result.val_take_rate),
            "val_net_mean_return": float(meta_result.val_net_mean_return),
            "val_max_drawdown": float(meta_result.val_max_drawdown),
            "test_take_rate": float(np.mean(action_mask_test.astype(float))) if len(action_mask_test) else 0.0,
        },
        "conformal_interval": {
            "alpha": float(np.clip(float(settings.conformal_alpha), 0.01, 0.40)),
            "q_abs_usd": float(conformal_q_abs_usd),
            "val_coverage": float(conformal_val_coverage),
            "test_coverage": float(conformal_test_coverage),
        },
        "top_features": {
            "classifier": _extract_top_feature_importance(cls_best.model, split.feature_columns),
            "regressor": _extract_top_feature_importance(reg_best.model, split.feature_columns),
        },
        "promotion_gate": {
            "passed": promotion_pass,
            "failed_reasons": failed_reasons,
            "mode": settings.promotion_gate_mode,
            "rules": {
                "classification": ["f1 > baseline.f1", "accuracy > baseline.accuracy (optional)"],
                "regression": ["rmse < baseline.rmse (optional)"],
                "execution": [
                    "net_mean_return > selected_financial_baseline.net_mean_return",
                    "net_mean_return > 0 (if configured)",
                    "sharpe_net > selected_financial_baseline.sharpe_net (if configured)",
                    "total_return > selected_financial_baseline.total_return (if configured)",
                ],
                "risk": ["max_drawdown >= max(PROMOTION_MAX_DRAWDOWN_LIMIT, baseline.max_drawdown)"],
                "leakage": ["no train/val/test overlap with purge gap >= steps_ahead"],
                "walk_forward": ["strict_pass_all_folds == true (strict mode only)"],
                "stochastic": ["abs(residual_acf1) <= 0.10 (strict mode only)"],
            },
        },
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
        "meta_artifact_path": str(meta_artifact_path) if meta_result.enabled else None,
        "regime_artifact_paths": {
            "classification": {key: str(path) for key, path in regime_artifact_paths["classification"].items()},
            "regression": {key: str(path) for key, path in regime_artifact_paths["regression"].items()},
        },
    }

    if write_artifacts:
        # Artifacts are versioned by symbol/horizon for deterministic inference and rollbacks.
        symbol_dir.mkdir(parents=True, exist_ok=True)
        dump(cls_best.model, artifact_paths["classification"])
        dump(reg_best.model, artifact_paths["regression"])
        if meta_result.enabled and meta_result.model is not None:
            dump(meta_result.model, meta_artifact_path)
        for name, model in regime_cls_models.items():
            dump(model, regime_artifact_paths["classification"][name])
        for name, model in regime_reg_models.items():
            dump(model, regime_artifact_paths["regression"][name])
        artifact_paths["calibration"].write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        artifact_paths["metrics"].write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metrics_payload["calibration"] = calibration
    return metrics_payload


def training_universe_snapshot(symbols: list[str]) -> dict[str, object]:
    """Compute training universe snapshot."""
    return {
        "count": len(symbols),
        "symbols": sorted(symbols),
    }


def model_manifest_payload(model_version: str, universe: list[str], symbol_horizon_results: dict[str, dict[str, object]]) -> dict[str, object]:
    """Compute model manifest payload."""
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
    """Compute all horizon specs."""
    return list(HORIZON_SPECS)


def as_json(data: object) -> str:
    """Return JSON."""
    return json.dumps(data, indent=2, sort_keys=False)


def as_plain_dict(obj: object) -> dict[str, object]:
    """Return plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)  # type: ignore[arg-type]
    raise TypeError("Object is not a dataclass instance")
