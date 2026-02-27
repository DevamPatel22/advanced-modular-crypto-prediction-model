from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_VERSION = "v2"

FEATURE_COLUMNS = [
    "ret_1",
    "ret_2",
    "ret_5",
    "ret_15",
    "ret_60",
    "log_ret_1",
    "ma_10_ratio",
    "ma_30_ratio",
    "ma_90_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "ema_8_ratio",
    "ema_21_ratio",
    "volatility_20",
    "volatility_5",
    "volatility_60",
    "volume_z_20",
    "volume_z_5",
    "rsi_14",
    "range_ratio",
    "hl_spread",
    "body_ratio",
    "close_position",
    "trend_slope_30",
    "mu_20",
    "sigma_20",
    "gbm_expected_ret_1",
    "gbm_var_1",
    "shock_z_20",
    "markov_prob_down",
    "markov_prob_flat",
    "markov_prob_up",
]

SHORT_HORIZONS = {"5m", "1h", "3h", "6h", "12h"}
LONG_HORIZONS = {"1d", "1w", "1mo", "3mo"}

SHORT_FEATURE_COLUMNS = [
    "ret_1",
    "ret_2",
    "ret_5",
    "ret_15",
    "log_ret_1",
    "ema_8_ratio",
    "ema_12_ratio",
    "ema_21_ratio",
    "volatility_5",
    "volatility_20",
    "volume_z_5",
    "volume_z_20",
    "rsi_14",
    "range_ratio",
    "hl_spread",
    "body_ratio",
    "close_position",
    "shock_z_20",
    "markov_prob_down",
    "markov_prob_flat",
    "markov_prob_up",
]

LONG_FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "log_ret_1",
    "ma_10_ratio",
    "ma_30_ratio",
    "ma_90_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "volatility_20",
    "volatility_60",
    "volume_z_20",
    "rsi_14",
    "trend_slope_30",
    "mu_20",
    "sigma_20",
    "gbm_expected_ret_1",
    "gbm_var_1",
    "shock_z_20",
    "markov_prob_down",
    "markov_prob_flat",
    "markov_prob_up",
]


@dataclass(frozen=True)
class HorizonSpec:
    label: str
    candidates: list[tuple[str, int]]


HORIZON_SPECS: list[HorizonSpec] = [
    HorizonSpec("5m", [("1m", 5), ("5m", 1)]),
    HorizonSpec("1h", [("1h", 1), ("15m", 4), ("5m", 12), ("1m", 60)]),
    HorizonSpec("3h", [("1h", 3), ("15m", 12), ("5m", 36), ("1m", 180)]),
    HorizonSpec("6h", [("1h", 6), ("15m", 24), ("5m", 72)]),
    HorizonSpec("12h", [("1h", 12), ("15m", 48), ("5m", 144)]),
    HorizonSpec("1d", [("1h", 24), ("6h", 4), ("1d", 1)]),
    HorizonSpec("1w", [("1d", 7), ("6h", 28), ("1h", 168)]),
    HorizonSpec("1mo", [("1d", 30), ("6h", 120), ("1h", 720)]),
    HorizonSpec("3mo", [("1d", 90), ("6h", 360), ("1h", 2160)]),
]

HORIZON_TO_DATA_WINDOW: dict[str, tuple[str, int]] = {
    spec.label: spec.candidates[0] for spec in HORIZON_SPECS
}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _encode_state(ret_1: pd.Series, flat_threshold: float = 0.0015) -> pd.Series:
    # -1 => down, 0 => flat, +1 => up
    states = pd.Series(np.where(ret_1 > flat_threshold, 1, np.where(ret_1 < -flat_threshold, -1, 0)), index=ret_1.index)
    return states


def _markov_transition_features(states: pd.Series) -> pd.DataFrame:
    # Online 3-state transition estimator using only past transitions.
    mapping = {-1: 0, 0: 1, 1: 2}
    encoded = states.map(mapping).to_numpy(dtype=float)
    n = len(encoded)
    probs = np.full((n, 3), np.nan, dtype=float)
    counts = np.zeros((3, 3), dtype=float)

    for t in range(1, n):
        prev_state = encoded[t - 1]
        curr_state = encoded[t]
        if np.isnan(prev_state) or np.isnan(curr_state):
            continue

        counts[int(prev_state), int(curr_state)] += 1.0
        row = counts[int(curr_state)]
        total = row.sum()
        if total > 0:
            probs[t, :] = row / total

    return pd.DataFrame(
        {
            "markov_prob_down": probs[:, 0],
            "markov_prob_flat": probs[:, 1],
            "markov_prob_up": probs[:, 2],
        },
        index=states.index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    # Technical/momentum/volatility/volume feature set.
    frame["ret_1"] = frame["close"].pct_change(1)
    frame["ret_2"] = frame["close"].pct_change(2)
    frame["ret_5"] = frame["close"].pct_change(5)
    frame["ret_15"] = frame["close"].pct_change(15)
    frame["ret_60"] = frame["close"].pct_change(60)
    frame["log_ret_1"] = np.log(frame["close"]).diff(1)

    frame["ma_10"] = frame["close"].rolling(10).mean()
    frame["ma_30"] = frame["close"].rolling(30).mean()
    frame["ma_90"] = frame["close"].rolling(90).mean()
    frame["ema_8"] = frame["close"].ewm(span=8, adjust=False).mean()
    frame["ema_12"] = frame["close"].ewm(span=12, adjust=False).mean()
    frame["ema_21"] = frame["close"].ewm(span=21, adjust=False).mean()
    frame["ema_26"] = frame["close"].ewm(span=26, adjust=False).mean()

    frame["ma_10_ratio"] = frame["close"] / frame["ma_10"] - 1
    frame["ma_30_ratio"] = frame["close"] / frame["ma_30"] - 1
    frame["ma_90_ratio"] = frame["close"] / frame["ma_90"] - 1
    frame["ema_8_ratio"] = frame["close"] / frame["ema_8"] - 1
    frame["ema_12_ratio"] = frame["close"] / frame["ema_12"] - 1
    frame["ema_21_ratio"] = frame["close"] / frame["ema_21"] - 1
    frame["ema_26_ratio"] = frame["close"] / frame["ema_26"] - 1

    frame["volatility_5"] = frame["log_ret_1"].rolling(5).std()
    frame["volatility_20"] = frame["log_ret_1"].rolling(20).std()
    frame["volatility_60"] = frame["log_ret_1"].rolling(60).std()
    frame["mu_20"] = frame["log_ret_1"].rolling(20).mean()
    frame["sigma_20"] = frame["log_ret_1"].rolling(20).std()
    frame["gbm_expected_ret_1"] = np.exp(frame["mu_20"] + 0.5 * np.square(frame["sigma_20"])) - 1
    frame["gbm_var_1"] = (np.exp(np.square(frame["sigma_20"])) - 1) * np.exp(
        2 * frame["mu_20"] + np.square(frame["sigma_20"])
    )
    frame["shock_z_20"] = (frame["log_ret_1"] - frame["mu_20"]) / (frame["sigma_20"] + 1e-12)

    rolling_volume_mean = frame["volume"].rolling(20).mean()
    rolling_volume_std = frame["volume"].rolling(20).std()
    rolling_volume_mean_5 = frame["volume"].rolling(5).mean()
    rolling_volume_std_5 = frame["volume"].rolling(5).std()
    frame["volume_z_5"] = (frame["volume"] - rolling_volume_mean_5) / (rolling_volume_std_5 + 1e-12)
    frame["volume_z_20"] = (frame["volume"] - rolling_volume_mean) / (rolling_volume_std + 1e-12)
    frame["rsi_14"] = _rsi(frame["close"], 14)
    frame["hl_spread"] = (frame["high"] - frame["low"]) / (frame["close"] + 1e-12)
    frame["range_ratio"] = (frame["close"] - frame["low"]) / ((frame["high"] - frame["low"]) + 1e-12)
    frame["body_ratio"] = (frame["close"] - frame["open"]) / (frame["open"] + 1e-12)
    frame["close_position"] = (frame["close"] - frame["low"]) / ((frame["high"] - frame["low"]) + 1e-12)
    frame["trend_slope_30"] = frame["close"].diff(30) / (frame["close"].shift(30) + 1e-12)
    states = _encode_state(frame["ret_1"])
    frame = frame.join(_markov_transition_features(states))

    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Deterministic NaN handling to keep feature matrix stable.
    for column in FEATURE_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype(float)

    return frame


def feature_columns_for_horizon(horizon: str) -> list[str]:
    if horizon in SHORT_HORIZONS:
        return SHORT_FEATURE_COLUMNS
    if horizon in LONG_HORIZONS:
        return LONG_FEATURE_COLUMNS
    return FEATURE_COLUMNS
