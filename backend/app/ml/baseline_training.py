from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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


@dataclass(frozen=True)
class HorizonSpec:
    label: str
    candidates: list[tuple[str, int]]


HORIZON_SPECS: list[HorizonSpec] = [
    HorizonSpec("5m", [("1m", 5), ("5m", 1)]),
    HorizonSpec("1h", [("1h", 1), ("15m", 4), ("5m", 12), ("1m", 60)]),
    HorizonSpec("6h", [("1h", 6), ("15m", 24), ("5m", 72)]),
    HorizonSpec("12h", [("1h", 12), ("15m", 48), ("5m", 144)]),
    HorizonSpec("1d", [("1h", 24), ("6h", 4), ("1d", 1)]),
    HorizonSpec("1w", [("1d", 7), ("6h", 28), ("1h", 168)]),
    HorizonSpec("1mo", [("1d", 30), ("6h", 120), ("1h", 720)]),
    HorizonSpec("3mo", [("1d", 90), ("6h", 360), ("1h", 2160)]),
]

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_15",
    "log_ret_1",
    "ma_10_ratio",
    "ma_30_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "volatility_20",
    "volume_z_20",
    "rsi_14",
    "range_ratio",
    "hl_spread",
]


def _database_path() -> Path:
    settings = get_settings()
    return Path(settings.market_data_sqlite_path)


def _connect() -> sqlite3.Connection:
    db_path = _database_path()
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def list_symbols_with_granularity(granularity: str, min_rows: int = 200) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT symbol
            FROM candles
            WHERE granularity = ?
            GROUP BY symbol
            HAVING COUNT(*) >= ?
            ORDER BY symbol ASC
            """,
            (granularity, min_rows),
        ).fetchall()
    return [str(row["symbol"]) for row in rows]


def load_candles(symbol: str, granularity: str) -> pd.DataFrame:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT start_time, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND granularity = ?
            ORDER BY start_time ASC
            """,
            (symbol, granularity),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["start_time", "open", "high", "low", "close", "volume"])

    frame = pd.DataFrame([dict(row) for row in rows])
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return frame


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["ret_1"] = frame["close"].pct_change(1)
    frame["ret_5"] = frame["close"].pct_change(5)
    frame["ret_15"] = frame["close"].pct_change(15)
    frame["log_ret_1"] = np.log(frame["close"]).diff(1)

    frame["ma_10"] = frame["close"].rolling(10).mean()
    frame["ma_30"] = frame["close"].rolling(30).mean()
    frame["ema_12"] = frame["close"].ewm(span=12, adjust=False).mean()
    frame["ema_26"] = frame["close"].ewm(span=26, adjust=False).mean()

    frame["ma_10_ratio"] = frame["close"] / frame["ma_10"] - 1
    frame["ma_30_ratio"] = frame["close"] / frame["ma_30"] - 1
    frame["ema_12_ratio"] = frame["close"] / frame["ema_12"] - 1
    frame["ema_26_ratio"] = frame["close"] / frame["ema_26"] - 1

    frame["volatility_20"] = frame["log_ret_1"].rolling(20).std()
    rolling_volume_mean = frame["volume"].rolling(20).mean()
    rolling_volume_std = frame["volume"].rolling(20).std()
    frame["volume_z_20"] = (frame["volume"] - rolling_volume_mean) / (rolling_volume_std + 1e-12)
    frame["rsi_14"] = _rsi(frame["close"], 14)
    frame["hl_spread"] = (frame["high"] - frame["low"]) / (frame["close"] + 1e-12)
    frame["range_ratio"] = (frame["close"] - frame["low"]) / ((frame["high"] - frame["low"]) + 1e-12)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _split_indices(length: int) -> tuple[int, int]:
    train_end = int(length * 0.7)
    val_end = int(length * 0.85)
    return train_end, val_end


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-12, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def _evaluate_horizon(symbol: str, spec: HorizonSpec) -> dict[str, object]:
    min_samples = 320
    lookback = 40

    chosen_granularity = None
    chosen_steps = None
    chosen_df = pd.DataFrame()

    for granularity, steps in spec.candidates:
        candidate = load_candles(symbol, granularity)
        if len(candidate) >= min_samples + steps + lookback:
            chosen_granularity = granularity
            chosen_steps = steps
            chosen_df = candidate
            break

    if chosen_granularity is None or chosen_steps is None:
        return {
            "horizon": spec.label,
            "status": "insufficient_data",
            "symbol": symbol,
            "required_min_rows": min_samples,
        }

    enriched = build_features(chosen_df)
    enriched["target_close"] = enriched["close"].shift(-chosen_steps)
    enriched["target_up"] = (enriched["target_close"] > enriched["close"]).astype(int)
    enriched = enriched.dropna(subset=FEATURE_COLUMNS + ["target_close", "target_up"]).reset_index(drop=True)

    if len(enriched) < min_samples:
        return {
            "horizon": spec.label,
            "status": "insufficient_data_after_features",
            "symbol": symbol,
            "rows": int(len(enriched)),
        }

    train_end, val_end = _split_indices(len(enriched))
    train_df = enriched.iloc[:train_end]
    val_df = enriched.iloc[train_end:val_end]
    test_df = enriched.iloc[val_end:]

    if len(test_df) < 30 or train_df["target_up"].nunique() < 2:
        return {
            "horizon": spec.label,
            "status": "insufficient_split_data",
            "symbol": symbol,
        }

    x_train = train_df[FEATURE_COLUMNS]
    y_train_cls = train_df["target_up"]
    y_train_reg = train_df["target_close"]

    x_test = test_df[FEATURE_COLUMNS]
    y_test_cls = test_df["target_up"].to_numpy(dtype=float)
    y_test_reg = test_df["target_close"].to_numpy(dtype=float)

    cls_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)),
        ]
    )
    cls_model.fit(x_train, y_train_cls)

    y_pred_cls = cls_model.predict(x_test)
    y_pred_proba = cls_model.predict_proba(x_test)[:, 1]

    reg_model = Pipeline(
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
    )
    reg_model.fit(x_train, y_train_reg)
    y_pred_reg = reg_model.predict(x_test)

    baseline_up = np.ones_like(y_test_cls)

    metrics = {
        "accuracy": float(accuracy_score(y_test_cls, y_pred_cls)),
        "precision": float(precision_score(y_test_cls, y_pred_cls, zero_division=0)),
        "recall": float(recall_score(y_test_cls, y_pred_cls, zero_division=0)),
        "f1": float(f1_score(y_test_cls, y_pred_cls, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_cls, y_pred_proba)) if len(np.unique(y_test_cls)) > 1 else math.nan,
        "baseline_up_accuracy": float(accuracy_score(y_test_cls, baseline_up)),
        "mae": float(mean_absolute_error(y_test_reg, y_pred_reg)),
        "rmse": float(math.sqrt(mean_squared_error(y_test_reg, y_pred_reg))),
        "mape": _mape(y_test_reg, y_pred_reg),
    }

    return {
        "horizon": spec.label,
        "status": "ok",
        "symbol": symbol,
        "granularity": chosen_granularity,
        "steps_ahead": chosen_steps,
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "metrics": metrics,
    }


def run_baseline_report(symbol: str, output_path: Path | None = None) -> dict[str, object]:
    settings = get_settings()
    normalized_symbol = symbol.upper().strip()

    report: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "symbol": normalized_symbol,
        "database": str(_database_path()),
        "horizons": [],
    }

    horizon_entries: list[dict[str, object]] = []
    for spec in HORIZON_SPECS:
        horizon_entries.append(_evaluate_horizon(normalized_symbol, spec))

    report["horizons"] = horizon_entries

    if output_path is None:
        output_path = Path("reports") / f"baseline_report_{normalized_symbol.lower().replace('-', '_')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

