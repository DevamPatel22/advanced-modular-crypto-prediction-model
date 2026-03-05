"""Reusable evaluation utilities for walk-forward, costs, and risk diagnostics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error


def _periods_per_year_from_horizon(horizon: str) -> float:
    """Internal helper to compute periods per year from horizon."""
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


def walk_forward_splits(
    n_samples: int,
    folds: int = 4,
    min_train_fraction: float = 0.55,
    min_test_size: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Compute walk forward splits."""
    if n_samples <= 0:
        return []
    folds = max(int(folds), 1)
    min_train_fraction = float(np.clip(min_train_fraction, 0.25, 0.85))
    min_test_size = max(int(min_test_size), 20)
    # Expanding-window walk-forward: train grows over time, test moves forward chronologically.
    train_end_start = max(int(n_samples * min_train_fraction), min_test_size * 2)
    if train_end_start >= n_samples - min_test_size:
        return []
    remaining = n_samples - train_end_start
    step = max(remaining // folds, min_test_size)
    splits: list[tuple[int, int, int, int]] = []
    train_end = train_end_start
    while train_end + min_test_size <= n_samples and len(splits) < folds:
        test_end = min(train_end + step, n_samples)
        if test_end - train_end < min_test_size:
            break
        splits.append((0, train_end, train_end, test_end))
        train_end = test_end
    return splits


def _threshold_grid(y_prob: np.ndarray) -> np.ndarray:
    """Internal helper to compute threshold grid."""
    quantiles = np.quantile(y_prob, np.linspace(0.05, 0.95, 19)) if y_prob.size else np.array([0.5], dtype=float)
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.20, 0.90, 71),
                quantiles,
                np.array([0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.82], dtype=float),
            ]
        )
    )
    return np.clip(grid, 0.01, 0.99)


def choose_threshold_walk_forward(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    folds: int = 4,
) -> tuple[float, dict[str, object]]:
    """Choose threshold walk forward."""
    labels = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if labels.size == 0 or labels.size != prob.size:
        return 0.5, {"mode": "fallback_invalid_input"}

    splits = walk_forward_splits(n_samples=labels.size, folds=folds)
    if not splits:
        return 0.5, {"mode": "fallback_insufficient_samples", "sample_count": int(labels.size)}

    thresholds = _threshold_grid(prob)
    rows: list[tuple[float, float, float, float]] = []
    fold_payload: list[dict[str, float]] = []
    for threshold in thresholds:
        # Score candidate threshold by baseline-relative margins across folds.
        margins: list[float] = []
        f1_deltas: list[float] = []
        acc_deltas: list[float] = []
        for _train_start, _train_end, test_start, test_end in splits:
            y_fold = labels[test_start:test_end]
            p_fold = prob[test_start:test_end]
            y_pred = (p_fold >= threshold).astype(int)
            baseline = np.ones_like(y_fold)
            score_f1 = float(f1_score(y_fold, y_pred, zero_division=0))
            score_acc = float(accuracy_score(y_fold, y_pred))
            baseline_f1 = float(f1_score(y_fold, baseline, zero_division=0))
            baseline_acc = float(accuracy_score(y_fold, baseline))
            delta_f1 = score_f1 - baseline_f1
            delta_acc = score_acc - baseline_acc
            f1_deltas.append(delta_f1)
            acc_deltas.append(delta_acc)
            margins.append(min(delta_f1, delta_acc))
        med_margin = float(np.median(margins))
        med_f1 = float(np.median(f1_deltas))
        med_acc = float(np.median(acc_deltas))
        score = med_margin + (0.45 * med_f1) + (0.25 * med_acc)
        rows.append((float(threshold), score, med_f1, med_acc))

    best_threshold, best_score, best_f1_delta, best_acc_delta = max(rows, key=lambda item: (item[1], item[2], item[3]))

    for fold_idx, (_train_start, _train_end, test_start, test_end) in enumerate(splits, start=1):
        y_fold = labels[test_start:test_end]
        p_fold = prob[test_start:test_end]
        y_pred = (p_fold >= best_threshold).astype(int)
        baseline = np.ones_like(y_fold)
        fold_payload.append(
            {
                "fold": float(fold_idx),
                "f1_delta": float(f1_score(y_fold, y_pred, zero_division=0) - f1_score(y_fold, baseline, zero_division=0)),
                "accuracy_delta": float(accuracy_score(y_fold, y_pred) - accuracy_score(y_fold, baseline)),
                "test_size": float(len(y_fold)),
            }
        )

    return (
        float(best_threshold),
        {
            "mode": "walk_forward",
            "fold_count": len(splits),
            "score": best_score,
            "median_f1_delta": best_f1_delta,
            "median_accuracy_delta": best_acc_delta,
            "folds": fold_payload,
        },
    )


def execution_aware_metrics(
    *,
    current_close: np.ndarray,
    target_close: np.ndarray,
    direction_up: np.ndarray,
    horizon: str,
    fee_bps: float,
    slippage_bps: float,
    max_turnover_per_step: float,
) -> dict[str, float]:
    """Compute execution aware metrics."""
    current = np.asarray(current_close, dtype=float)
    target = np.asarray(target_close, dtype=float)
    signal = np.asarray(direction_up, dtype=int)
    if current.size == 0 or current.size != target.size or signal.size != current.size:
        return {
            "gross_mean_return": math.nan,
            "net_mean_return": math.nan,
            "turnover_rate": math.nan,
            "sharpe_gross": math.nan,
            "sharpe_net": math.nan,
            "max_drawdown_net": math.nan,
            "var95_net": math.nan,
            "cvar95_net": math.nan,
        }

    raw_returns = (target / np.clip(current, 1e-12, None)) - 1.0
    desired_positions = np.where(signal == 1, 1.0, -1.0)
    # Turnover cap is converted into max position delta per step.
    max_turnover_per_step = float(np.clip(max_turnover_per_step, 0.0, 1.0))
    max_position_delta = max_turnover_per_step * 2.0
    positions = np.zeros_like(desired_positions, dtype=float)
    for idx, desired in enumerate(desired_positions):
        if idx == 0:
            positions[idx] = float(np.clip(desired, -max_position_delta, max_position_delta))
            continue
        delta = desired - positions[idx - 1]
        positions[idx] = positions[idx - 1] + float(np.clip(delta, -max_position_delta, max_position_delta))
    gross = positions * raw_returns

    prev_positions = np.concatenate(([0.0], positions[:-1]))
    turnover = np.abs(positions - prev_positions) / 2.0
    turnover_rate = float(np.mean(turnover))
    trading_cost = turnover * ((float(fee_bps) + float(slippage_bps)) / 10000.0)
    net = gross - trading_cost

    def _sharpe(values: np.ndarray) -> float:
        """Internal helper to compute sharpe."""
        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if sigma <= 1e-12:
            return 0.0
        return float((mu / sigma) * math.sqrt(max(_periods_per_year_from_horizon(horizon), 1.0)))

    equity = np.cumprod(1.0 + net)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity / np.clip(running_max, 1e-12, None)) - 1.0

    q95 = float(np.quantile(net, 0.05))
    left_tail = net[net <= q95]
    cvar95 = float(np.mean(left_tail)) if left_tail.size else q95

    return {
        "gross_mean_return": float(np.mean(gross)),
        "net_mean_return": float(np.mean(net)),
        "turnover_rate": turnover_rate,
        "sharpe_gross": _sharpe(gross),
        "sharpe_net": _sharpe(net),
        "max_drawdown_net": float(np.min(drawdown)),
        "var95_net": q95,
        "cvar95_net": cvar95,
    }


def strict_gate_walk_forward_diagnostic(
    *,
    y_true_cls: np.ndarray,
    y_prob_cls: np.ndarray,
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    baseline_reg: np.ndarray,
    decision_threshold: float,
    folds: int,
) -> dict[str, object]:
    """Compute strict gate walk forward diagnostic."""
    y_cls = np.asarray(y_true_cls, dtype=int)
    y_prob = np.asarray(y_prob_cls, dtype=float)
    y_reg = np.asarray(y_true_reg, dtype=float)
    y_reg_pred = np.asarray(y_pred_reg, dtype=float)
    y_reg_base = np.asarray(baseline_reg, dtype=float)
    if y_cls.size == 0 or y_cls.size != y_prob.size or y_reg.size != y_reg_pred.size or y_reg.size != y_cls.size:
        return {
            "enabled": False,
            "reason": "invalid_inputs",
        }

    splits = walk_forward_splits(n_samples=y_cls.size, folds=folds, min_train_fraction=0.5, min_test_size=24)
    if not splits:
        return {
            "enabled": False,
            "reason": "insufficient_samples",
            "sample_count": int(y_cls.size),
        }

    fold_rows: list[dict[str, float]] = []
    fold_pass_flags: list[bool] = []
    for fold_idx, (_train_start, _train_end, test_start, test_end) in enumerate(splits, start=1):
        # Fold must beat baseline on both classification and regression dimensions.
        y_cls_fold = y_cls[test_start:test_end]
        y_prob_fold = y_prob[test_start:test_end]
        y_reg_fold = y_reg[test_start:test_end]
        y_reg_pred_fold = y_reg_pred[test_start:test_end]
        y_reg_base_fold = y_reg_base[test_start:test_end]

        y_pred_fold = (y_prob_fold >= float(decision_threshold)).astype(int)
        cls_baseline = np.ones_like(y_cls_fold)
        f1_model = float(f1_score(y_cls_fold, y_pred_fold, zero_division=0))
        f1_base = float(f1_score(y_cls_fold, cls_baseline, zero_division=0))
        acc_model = float(accuracy_score(y_cls_fold, y_pred_fold))
        acc_base = float(accuracy_score(y_cls_fold, cls_baseline))
        rmse_model = float(math.sqrt(mean_squared_error(y_reg_fold, y_reg_pred_fold)))
        rmse_base = float(math.sqrt(mean_squared_error(y_reg_fold, y_reg_base_fold)))

        pass_fold = (f1_model > f1_base) and (acc_model > acc_base) and (rmse_model < rmse_base)
        fold_pass_flags.append(pass_fold)
        fold_rows.append(
            {
                "fold": float(fold_idx),
                "f1_delta": f1_model - f1_base,
                "accuracy_delta": acc_model - acc_base,
                "rmse_delta": rmse_base - rmse_model,
                "strict_pass": float(1.0 if pass_fold else 0.0),
            }
        )

    f1_delta_med = float(np.median([row["f1_delta"] for row in fold_rows]))
    acc_delta_med = float(np.median([row["accuracy_delta"] for row in fold_rows]))
    rmse_delta_med = float(np.median([row["rmse_delta"] for row in fold_rows]))
    pass_rate = float(np.mean([1.0 if item else 0.0 for item in fold_pass_flags]))
    strict_pass_all = all(fold_pass_flags)

    return {
        "enabled": True,
        "fold_count": len(splits),
        "decision_threshold": float(decision_threshold),
        "median_f1_delta": f1_delta_med,
        "median_accuracy_delta": acc_delta_med,
        "median_rmse_delta": rmse_delta_med,
        "strict_pass_rate": pass_rate,
        "strict_pass_all_folds": strict_pass_all,
        "folds": fold_rows,
    }


def bootstrap_metric_confidence_intervals(
    *,
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    n_bootstrap: int = 400,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap metric confidence intervals."""
    true_cls = np.asarray(y_true_cls, dtype=int)
    pred_cls = np.asarray(y_pred_cls, dtype=int)
    true_reg = np.asarray(y_true_reg, dtype=float)
    pred_reg = np.asarray(y_pred_reg, dtype=float)
    n = int(true_cls.size)
    if n < 40 or n != pred_cls.size or n != true_reg.size or n != pred_reg.size:
        return {
            "enabled": False,
            "reason": "insufficient_or_mismatched_samples",
            "sample_count": float(n),
        }

    n_bootstrap = max(int(n_bootstrap), 80)
    confidence = float(np.clip(confidence, 0.80, 0.99))
    alpha = (1.0 - confidence) / 2.0
    lower_q = alpha
    upper_q = 1.0 - alpha
    rng = np.random.default_rng(int(random_seed))

    acc_rows: list[float] = []
    f1_rows: list[float] = []
    rmse_rows: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t_cls = true_cls[idx]
        p_cls = pred_cls[idx]
        t_reg = true_reg[idx]
        p_reg = pred_reg[idx]
        acc_rows.append(float(accuracy_score(t_cls, p_cls)))
        f1_rows.append(float(f1_score(t_cls, p_cls, zero_division=0)))
        rmse_rows.append(float(math.sqrt(mean_squared_error(t_reg, p_reg))))

    def _summary(values: list[float]) -> dict[str, float]:
        """Internal helper to compute summary."""
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "lower": float(np.quantile(arr, lower_q)),
            "upper": float(np.quantile(arr, upper_q)),
        }

    return {
        "enabled": True,
        "confidence_level": confidence,
        "n_bootstrap": float(n_bootstrap),
        "sample_count": float(n),
        "accuracy": _summary(acc_rows),
        "f1": _summary(f1_rows),
        "rmse": _summary(rmse_rows),
    }


def paper_trading_metrics(
    *,
    current_close: np.ndarray,
    target_close: np.ndarray,
    direction_up: np.ndarray,
    fee_bps: float,
    slippage_bps: float,
    max_turnover_per_step: float,
    initial_capital: float = 10_000.0,
) -> dict[str, float]:
    """Compute paper trading metrics."""
    current = np.asarray(current_close, dtype=float)
    target = np.asarray(target_close, dtype=float)
    signal = np.asarray(direction_up, dtype=int)
    if current.size == 0 or current.size != target.size or signal.size != current.size:
        return {
            "enabled": False,
            "reason": "invalid_inputs",
        }

    start_capital = max(float(initial_capital), 100.0)
    max_turnover_per_step = float(np.clip(max_turnover_per_step, 0.0, 1.0))
    max_position_delta = max_turnover_per_step * 2.0
    desired_positions = np.where(signal == 1, 1.0, -1.0).astype(float)
    positions = np.zeros_like(desired_positions, dtype=float)
    for idx, desired in enumerate(desired_positions):
        if idx == 0:
            positions[idx] = float(np.clip(desired, -max_position_delta, max_position_delta))
            continue
        delta = desired - positions[idx - 1]
        positions[idx] = positions[idx - 1] + float(np.clip(delta, -max_position_delta, max_position_delta))

    realized_returns = (target / np.clip(current, 1e-12, None)) - 1.0
    gross = positions * realized_returns
    prev_positions = np.concatenate(([0.0], positions[:-1]))
    turnover = np.abs(positions - prev_positions) / 2.0
    costs = turnover * ((float(fee_bps) + float(slippage_bps)) / 10000.0)
    net = gross - costs

    equity = np.empty_like(net, dtype=float)
    running = start_capital
    for idx, step_ret in enumerate(net):
        running = running * max(1.0 + float(step_ret), 1e-6)
        equity[idx] = running

    peaks = np.maximum.accumulate(equity)
    drawdown = (equity / np.clip(peaks, 1e-12, None)) - 1.0
    best_equity = float(np.max(equity))
    worst_equity = float(np.min(equity))
    final_equity = float(equity[-1]) if equity.size else start_capital
    total_return = (final_equity / start_capital) - 1.0
    win_rate = float(np.mean(net > 0.0)) if net.size else 0.0
    q95 = float(np.quantile(net, 0.05))
    left_tail = net[net <= q95]
    cvar95 = float(np.mean(left_tail)) if left_tail.size else q95

    return {
        "enabled": True,
        "initial_capital": start_capital,
        "final_equity": final_equity,
        "best_equity": best_equity,
        "worst_equity": worst_equity,
        "total_return": float(total_return),
        "max_drawdown": float(np.min(drawdown)) if drawdown.size else 0.0,
        "win_rate": win_rate,
        "mean_step_return": float(np.mean(net)) if net.size else 0.0,
        "turnover_rate": float(np.mean(turnover)) if turnover.size else 0.0,
        "var95": q95,
        "cvar95": cvar95,
    }
