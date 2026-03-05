# Interview Narrative

## Problem Framing

Build a production-minded crypto prediction platform that combines:

- real-time market visibility (Robinhood-style UX)
- horizon-based probabilistic prediction outputs
- strict risk and promotion controls so weak models never silently go live

Target output per symbol+horizon:

- directional view (bullish/bearish)
- predicted close price at horizon end
- calibrated confidence
- risk range (min/max return %) with risk score

## Design Choices

1. Strict baseline-first promotion
- Reason: avoid model theater.
- Implementation: model must beat always-up baseline on classification (`f1`, `accuracy`) and persistence baseline on regression (`rmse`) before activation.

2. Time-ordered validation with walk-forward diagnostics
- Reason: preserve temporal causality.
- Implementation: purged chronological splits, optional strict walk-forward fold gate.

3. Execution-aware evaluation
- Reason: predictive accuracy alone is not tradable edge.
- Implementation: fee/slippage/turnover-adjusted metrics and paper-trading PnL metrics.

4. Reliability-first inference
- Reason: quality varies by pair/horizon.
- Implementation: model registry, artifact-backed inference, confidence-based abstention, automatic fallback.

5. Operations and lineage
- Reason: reproducibility and rollback are required in real systems.
- Implementation: experiment event logs, model registry history, artifact checksums, auto rollback guard.

## Key Tradeoffs

- Tradeoff: strict gates reduce promotion count.
  Outcome: higher precision on what gets activated, slower coverage growth.

- Tradeoff: purged split leakage protection reduces sample size.
  Outcome: lower false optimism, stronger out-of-sample credibility.

- Tradeoff: richer model family increases compute cost.
  Outcome: better chance to beat baseline across heterogeneous horizons.

## How This Differs From Pure Research Code

- Health endpoints for data-readiness and source telemetry.
- Quality-gated retraining instead of unconditional retraining.
- Registry-driven activation and rollback support.
- Explicit failure-mode documentation and fallback behavior.

## What Institutional Teams Still Have More Of

- deeper proprietary datasets and exchange connectivity
- lower-latency execution stack and richer transaction cost models
- larger hyperparameter and architecture search budgets
- broader portfolio-level optimization stack

## What I Would Improve With More Resources

1. Expand data depth and quality
- add paid-grade order book, derivatives term-structure, and on-chain entity-level signals.

2. Add portfolio optimization layer
- objective-level optimization for return/risk/turnover with constraints.

3. Add online learning and drift alarms
- adaptive thresholding and regime detector retrain triggers.

4. Add formal experiment tracking service
- move from JSONL logs to a queryable tracking backend with dashboards.

5. Add shadow deployment and live A/B policy checks
- compare candidate vs active behavior before promotion.

## Honest Positioning

This system is designed to be credible and extensible under strict controls, not to claim guaranteed outperformance.
It demonstrates institutional engineering patterns: gating, risk controls, reproducibility, and rollback discipline.
