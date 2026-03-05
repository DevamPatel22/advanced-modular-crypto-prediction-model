# Model Risk and Failure Modes

This document records where the model is strong, where it fails, and which safeguards are active.

## What Works Best

- High-liquidity symbols with deeper history (`BTC-USD`, `ETH-USD`, `SOL-USD`).
- Short and medium horizons where microstructure and momentum features are denser (`5m` to `12h`).
- Regime-consistent windows (stable trend or stable range) where threshold calibration remains stable.

## Known Failure Modes

1. Thin-history symbols or sparse granularities
- Failure: unstable thresholds, high variance RMSE, low promotion pass rate.
- Mitigation: data-quality gate blocks retrain/promotion when row depth/freshness/gap checks fail.

2. Distribution shift / regime break
- Failure: rapid volatility regime change degrades F1 and directional confidence.
- Mitigation: regime-aware features, walk-forward diagnostics, abstain-to-fallback when confidence is too low.

3. Cost-insensitive apparent edge
- Failure: pre-cost signal looks good but collapses after fees/slippage.
- Mitigation: execution-aware metrics and paper-trading metrics are persisted for every symbol+horizon.

4. Boundary leakage risk in time splits
- Failure: optimistic validation from training labels that overlap future windows.
- Mitigation: purged train/val/test split with explicit leakage diagnostics (`required_gap_rows >= steps_ahead`).

5. Overfitting on narrow windows
- Failure: candidate model beats baseline in one slice, fails out-of-sample.
- Mitigation: strict baseline gate, optional strict walk-forward fold gate, and fallback for non-promoted pairs.

6. Source degradation and stale cache fallback
- Failure: degraded source quality causes stale input and unstable signals.
- Mitigation: source-health telemetry, quality-gated retrain, auto rollback guard tooling.

## Hard Production Gates

A symbol+horizon is promoted only if all required checks pass:

- `f1 > always-up baseline f1`
- `accuracy > always-up baseline accuracy`
- `rmse < persistence baseline rmse`
- leakage diagnostic pass (purged split boundaries)
- optional strict gates (when enabled):
  - walk-forward strict pass across folds
  - martingale residual ACF bound

If any gate fails, inference remains on fallback behavior for that pair.

## Kill-Switch and Fallback Logic

- Per-inference abstention: model output is rejected when confidence is below `PREDICTION_CONFIDENCE_MIN_FOR_MODEL`.
- Pair-level fallback: non-promoted or artifact-missing pairs automatically return fallback predictions.
- Ops-level rollback: `scripts/auto_rollback_guard.py` can revert active model version if quality/health degrades.

## Risk Disclosure

- This is a research and decision-support system, not guaranteed alpha.
- Accuracy and edge are non-stationary and sensitive to market microstructure changes.
- The strict gate is intentionally hard; fewer promotions are preferable to low-quality false positives.
