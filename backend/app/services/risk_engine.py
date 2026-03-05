from __future__ import annotations

import math

import numpy as np


def portfolio_risk_snapshot(
    returns: list[float],
    confidence_level: float = 0.95,
    annualization_factor: float = 365.0,
) -> dict[str, float]:
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        return {
            "sample_count": 0,
            "mean_return": math.nan,
            "volatility": math.nan,
            "sharpe": math.nan,
            "var": math.nan,
            "cvar": math.nan,
            "max_drawdown": math.nan,
        }

    mean_return = float(np.mean(values))
    volatility = float(np.std(values))
    if volatility <= 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((mean_return / volatility) * math.sqrt(max(float(annualization_factor), 1.0)))

    left_tail_q = float(np.clip(1.0 - confidence_level, 0.001, 0.20))
    var = float(np.quantile(values, left_tail_q))
    tail = values[values <= var]
    cvar = float(np.mean(tail)) if tail.size else var

    equity = np.cumprod(1.0 + values)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity / np.clip(running_max, 1e-12, None)) - 1.0
    max_drawdown = float(np.min(drawdown))

    return {
        "sample_count": int(values.size),
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "var": var,
        "cvar": cvar,
        "max_drawdown": max_drawdown,
    }


def apply_risk_limits(
    proposed_weights: dict[str, float],
    current_weights: dict[str, float],
    *,
    max_position_abs: float,
    max_gross_exposure: float,
    max_turnover: float,
) -> dict[str, object]:
    cleaned: dict[str, float] = {}
    breached_limits: list[str] = []

    for symbol, weight in proposed_weights.items():
        bounded = float(np.clip(float(weight), -max_position_abs, max_position_abs))
        if not math.isclose(float(weight), bounded, rel_tol=1e-9, abs_tol=1e-9):
            breached_limits.append(f"position_cap:{symbol}")
        cleaned[symbol] = bounded

    gross_exposure = float(sum(abs(value) for value in cleaned.values()))
    if gross_exposure > max_gross_exposure and gross_exposure > 1e-12:
        scale = float(max_gross_exposure / gross_exposure)
        cleaned = {symbol: weight * scale for symbol, weight in cleaned.items()}
        gross_exposure = float(sum(abs(value) for value in cleaned.values()))
        breached_limits.append("gross_exposure_cap")

    symbols = set(cleaned) | set(current_weights)
    turnover = float(sum(abs(cleaned.get(symbol, 0.0) - float(current_weights.get(symbol, 0.0))) for symbol in symbols))
    if turnover > max_turnover and turnover > 1e-12:
        scale = float(max_turnover / turnover)
        adjusted: dict[str, float] = {}
        for symbol in symbols:
            current = float(current_weights.get(symbol, 0.0))
            target = float(cleaned.get(symbol, 0.0))
            adjusted[symbol] = current + ((target - current) * scale)
        cleaned = adjusted
        turnover = float(sum(abs(cleaned.get(symbol, 0.0) - float(current_weights.get(symbol, 0.0))) for symbol in symbols))
        breached_limits.append("turnover_cap")

    return {
        "accepted_weights": cleaned,
        "gross_exposure": gross_exposure,
        "turnover": turnover,
        "breached_limits": sorted(set(breached_limits)),
    }
