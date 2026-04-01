# Backend Service

## Overview

This service provides the prediction API backend.

Current scope:

- FastAPI service with market and prediction APIs
- US-tradable symbol universe validation (`quote=USD`)
- market-data retrieval + SQLite caching
  Candle ingestion uses Coinbase as primary, Binance (free spot API) as secondary, and CryptoCompare (free API) as tertiary fallback/supplement.
  Credibility guards are enforced (OHLCV integrity, timestamp consistency, freshness, cross-source divergence checks).
- background ingestion loop for tradable symbols
- risk-aware prediction range output
- model registry + promotion flow
- baseline and all-symbol training/evaluation CLIs

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /` root service metadata
- `GET /api/v1/health` service health status
- `GET /api/v1/health/data-readiness` latest data-quality snapshot for training gate readiness
- `GET /api/v1/health/source-health?hours=24` source reliability telemetry summary
- `GET /api/v1/health/model-scorecard` latest model scorecard snapshot
- `GET /api/v1/health/robust-validation` latest robust-validation bundle for the active model
- `GET /api/v1/health/shadow-trading` latest live shadow-trading summary
- `GET /api/v1/markets/symbols?quote=USD` tradable symbols catalog
- `GET /api/v1/market-data/candles?symbol=BTC-USD&granularity=1h&limit=200` OHLCV candles
- `GET /api/v1/market-data/ticker?symbol=BTC-USD` latest ticker
- `WS /api/v1/market-data/ws/ticker?symbol=BTC-USD` streaming ticker updates
- `POST /api/v1/predictions` prediction output with risk-aware return range
- `POST /api/v1/risk/portfolio-snapshot` portfolio-level VaR/CVaR/drawdown snapshot
- `POST /api/v1/risk/limit-check` exposure/turnover risk-limit enforcement

All prediction and market-data endpoints validate symbols against the US-tradable USD pair universe.
Symbols outside that universe return `400`.

Background ingestion loop:

- On startup, the service can continuously refresh candles for US-tradable USD symbols.
- Controlled by `INGESTION_*` env variables in `backend/.env`.
- Priority symbols from `SUPPORTED_SYMBOLS` are always ingested even when `INGESTION_SYMBOL_LIMIT` is low.
- Source controls:
  `MARKET_DATA_SOURCE_BASE_URL` (primary, Coinbase),
  `MARKET_DATA_SECONDARY_SOURCE_ENABLED`,
  `MARKET_DATA_SECONDARY_SOURCE_BASE_URL` (secondary, Binance),
  `MARKET_DATA_TERTIARY_SOURCE_ENABLED`,
  `MARKET_DATA_TERTIARY_SOURCE_BASE_URL` (tertiary, CryptoCompare),
  `MARKET_DATA_TERTIARY_SOURCE_API_KEY` (optional).
  Phase-1 focus controls:
  `PHASE1_FOCUS_SYMBOLS` (default `BTC-USD,ETH-USD,SOL-USD`),
  `PHASE1_FOCUS_HORIZONS` (default `6h,12h,1d`).

Source telemetry:

- every market-data fetch writes a `source_health_events` record in SQLite
- includes source credibility outcome, selected source, divergence, and stale-cache fallback usage
- powers `/api/v1/health/source-health` for ops visibility

## Data Foundation Workflow (Before Retraining)

Deep backfill priority symbols to target history depth:

```bash
python scripts/backfill_market_data.py \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --granularities 1m,5m,15m,1h,6h,1d \
  --target-rows-map 1m:30000,5m:18000,15m:12000,1h:6000,6h:2500,1d:1500 \
  --deep-history \
  --deep-history-max-limit-map 1m:60000,5m:30000,15m:20000,1h:15000,6h:12000,1d:5000 \
  --max-passes 3 \
  --output reports/backfill_market_data.json
```

Generate archive completeness report after deep backfill:

```bash
python scripts/history_completeness_report.py \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --granularities 1m,5m,15m,1h,6h,1d \
  --output reports/history_completeness_report.json
```

Generate data quality report:

```bash
python scripts/data_quality_report.py \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --granularities 1m,5m,15m,1h,6h,1d \
  --min-coverage-ratio-map 1m:0.80,5m:0.82,15m:0.85,1h:0.88,6h:0.90,1d:0.92 \
  --max-cross-symbol-lag-steps-map 1m:480,5m:192,15m:96,1h:48,6h:20,1d:10 \
  --output reports/data_quality_report.json
```

Key gates checked per symbol/granularity:

- minimum row depth
- freshness (staleness steps)
- timestamp interval consistency
- gap ratio threshold

Cross-symbol synchronization gate (per granularity):

- latest timestamp spread across requested symbols must stay within configured step limits
- blocks retraining when one symbol is materially stale relative to others (e.g., ETH/SOL lagging BTC)

Use retrain quality gate to block training when data is weak:

```bash
python scripts/daily_retrain.py \
  --phase phase3 \
  --enforce-data-quality \
  --symbols BTC-USD,ETH-USD,SOL-USD
```

Quality gate remediation behavior:

- automatic backfill remediation runs for failing symbols/granularities (enabled by default)
- unresolved failing pairs are excluded for the current run instead of aborting all training
- exclusions are passed into training as `--exclude-symbol-horizons` and stored in retrain reports

Run explicit SLA gate (quality + source uptime):

```bash
python scripts/sla_gate.py \
  --require-quality-pass \
  --min-live-source-ratio 0.70 \
  --max-stale-cache-ratio 0.30 \
  --output reports/sla_gate_report.json
```

## Baseline Accuracy Evaluation

Run baseline training/evaluation for one symbol:

```bash
python scripts/train_baseline.py --symbol BTC-USD --output reports/baseline_report_btc_usd.json
```

The report includes per-horizon:

- classification metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- baseline comparison: `baseline_up_accuracy`
- regression metrics: `mae`, `rmse`, `mape`
- data notes: selected granularity and step horizon

If a horizon shows `insufficient_data`, ingest more history first and rerun.

## Full-Universe Training (All US-Tradable USD Symbols)

Train candidate models for all tradable symbols and all supported horizons:

```bash
python scripts/train_all_symbols.py --model-version daily-20260213-020000 --output reports/summary_report.json
```

Lock to core phase-1 wedge only:

```bash
python scripts/train_all_symbols.py \
  --model-version core-$(date +%Y%m%d-%H%M%S) \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --horizons 6h,12h,1d \
  --output reports/summary_report_core.json
```

Optional training controls:

- `--exclude-symbol-horizons BTC-USD:6h,ETH-USD:1d` to skip specific pairs
- `--asof-cutoff-map 1m:1772559060,1h:1772557200` for explicit synchronized cutoffs
- when `TRAIN_ASOF_SYNC_ENABLED=true`, cutoffs are auto-derived per granularity from dataset coverage
- when `TRAIN_SYNC_ROW_DEPTH_ENABLED=true`, each granularity is truncated to synchronized trailing row-depth parity before feature generation
- when `TRAIN_HORIZON_WINDOWING_ENABLED=true`, each horizon is trained on a recent capped tail so stale launch-era regimes do not dominate short/medium horizon models

Training targets horizons:

- `5m`, `1h`, `3h`, `6h`, `12h`, `1d`, `1w`, `1mo`, `3mo`

Candidate models per symbol+horizon:

- Classification (theory mode): Logistic Regression + HistGradientBoosting (+ optional ensemble challenger).
  Expanded model zoo remains available via `MODEL_CANDIDATE_MODE=expanded`.
  Validation-tuned probability blending of top classifiers remains enabled for near-pass optimization.
- Meta-labeling layer: a secondary classifier learns when to act (or abstain) on base signals using edge/probability/regime features.
- Regression (theory mode): BayesianRidge ARX, Ridge, Huber, HistGradientBoosting (+ optional ensemble challenger).
  Expanded model zoo remains available via `MODEL_CANDIDATE_MODE=expanded`.
  Regression target is modeled as `log_return` and transformed back to price with horizon-aware clipping to reduce extreme RMSE outliers.
  Training now also evaluates a residual target (`target_close - current_close`) and promotes it only when it gives a minimum relative RMSE improvement over log-return on validation.
  Final regression output uses a validation-optimized blend with persistence (`regression_blend_alpha`) to reduce RMSE drift.
  Per-horizon feature subsets are used (short-horizon microstructure/volatility burst vs long-horizon trend/regime persistence).

Stochastic feature layer (used by all model candidates):

- GBM-style drift/volatility features (`mu_20`, `sigma_20`, expected return/variance)
- shock normalization (`shock_z_20`)
- Markov transition probabilities over return regimes (`down/flat/up`)
- volatility-scaled triple-barrier classification labels (`CLASSIFICATION_LABEL_MODE=triple_barrier`)

Promotion gate (must pass all):

- returns-first execution edge:
  `net_mean_return > selected_financial_baseline.net_mean_return`
- optional return constraints:
  `net_mean_return > 0`, `sharpe_net > baseline.sharpe_net`, `total_return > baseline.total_return`
- risk `max_drawdown >= max(PROMOTION_MAX_DRAWDOWN_LIMIT, baseline.max_drawdown)`
- optional predictive edge checks:
  classification `f1 > baseline.f1` and `accuracy > baseline.accuracy`
  regression `rmse < baseline.rmse`
- leakage diagnostics must pass
- walk-forward strict mode must pass when enabled
- martingale diagnostic `abs(residual_acf1) <= 0.10` when `MARTINGALE_GATE_MODE=strict`

Financial baseline policy (strict + fair):

- default comparator is fixed (`FINANCIAL_BASELINE_SELECTION_MODE=fixed`, `FINANCIAL_BASELINE_STRATEGY=always_long`)
- optional modes:
  - `best_validation`: choose baseline strategy on validation window, evaluate on test window
  - `best_expost`: choose strongest baseline on test window (diagnostic only, strictest comparator)
- this avoids look-ahead/oracle bias in default promotion comparisons

Returns tuning (pre-gate optimization):

- when `RETURNS_TUNING_ENABLED=true`, training performs validation-time joint tuning of:
  - classifier decision threshold
  - edge filter threshold scale
  - trade direction policy (`RETURNS_TUNING_POLICIES`)
  - position-size multiplier (`RETURNS_TUNING_POSITION_SCALE_CANDIDATES`)
- tuned runtime choices are persisted in calibration/metrics artifacts:
  - `decision_threshold`
  - `edge_threshold_scale`
  - `trade_direction_policy`
  - `position_scale`

Training reports additionally include:

- high-confidence slice quality at `HIGH_CONFIDENCE_THRESHOLD`
- regime breakdown metrics (`down/flat/up`) on the test window
- bootstrap confidence intervals for `accuracy`, `f1`, and `rmse`
- purged split leakage diagnostics (must pass to qualify)
- paper-trading metrics (`equity`, `drawdown`, `VaR/CVaR`, `turnover`) with configured fees/slippage
- near-pass deltas vs baseline and top feature importance lists for targeted retraining
- alpha-hypothesis diagnostics (information coefficient, decile spread, sign alignment) for
  `reversal_pressure_5`, `mean_reversion_z_20`, `volatility_cluster_20_60`, `kalman_level_ratio`, and `kalman_trend_5`
- alpha kill-switch diagnostics showing which weak signals were disabled before training
- ablation metrics comparing with-vs-without meta abstention for execution and paper-trading outcomes

Horizon data readiness uses adaptive minimum sample targets (short horizons require more history than long horizons) so early-stage training can activate qualified pairs sooner while still keeping gate checks strict.

Baselines:

- direction baseline: always-up
- price baseline: persistence (`next_close = current_close`)

## Promotion / Activation

Promote a trained candidate version:

```bash
python scripts/promote_model.py --candidate daily-20260213-020000 --active daily-20260212-020000 --phase phase1
```

Require CI gate before promotion:

```bash
python scripts/promote_model.py \
  --candidate <version> \
  --phase phase1 \
  --require-ci-gate \
  --ci-gate-report reports/ci_gate_latest.json
```

To merge newly passing pairs into the existing promoted registry (recommended for targeted retrains):

```bash
python scripts/promote_model.py --candidate <version> --phase phase3 --merge-existing
```

Phases:

- `phase1`: activates only `BTC-USD`, `ETH-USD`, `SOL-USD` entries that pass gates
- `phase2`: activates previous active set + next batch (`--phase2-batch-size`, default 20)
- `phase3`: activates all passed symbol+horizon entries
  In bootstrap mode, phase1 additionally restricts promotions to short horizons from `BOOTSTRAP_PHASE1_HORIZONS` (default `5m,1h,3h,6h,12h`).

Promotion safety:

- if a candidate produces `0` promoted pairs, active model version is preserved (no empty rollout)

Any symbol+horizon without promoted artifacts automatically stays on fallback inference.
Model inference also supports reliability abstention:
if confidence is below `PREDICTION_CONFIDENCE_MIN_FOR_MODEL` and `PREDICTION_ABSTAIN_TO_FALLBACK=true`,
the endpoint returns fallback output instead of low-edge model output.
If meta-label artifacts are available, inference also abstains when meta edge probability is below its calibrated threshold.
Prediction responses now include conformal USD intervals (`conformal_low_usd`, `conformal_high_usd`, `conformal_confidence`).
Inference also uses calibrated per-horizon decision thresholds (not fixed 0.5) and optional regime-routed models when available.
Inference also applies persisted `regression_blend_alpha` from metrics artifacts.

## Artifact Layout

```text
backend/data/models/
  registry.json
  <model_version>/
    manifest.json
    artifact_manifest.json
    dataset_manifest.json
    <symbol>/
      cls_<horizon>.joblib
      reg_<horizon>.joblib
      calibration_<horizon>.json
      metrics_<horizon>.json
```

Reports:

- per-symbol reports: `backend/reports/symbols/<symbol>.json`
- aggregate report: `backend/reports/summary_report.json`
- promotion report: `backend/reports/promotion_report.json`
- robust validation bundle: `backend/reports/robust_validation_<model_version>.json`
- shadow trading report: `backend/reports/shadow_report_latest.json`
- robust validation now requires model-matched shadow evidence and emits JSON-safe nulls instead of `NaN` tokens

## Retraining Cadence

- Target schedule: daily (`RETRAIN_SCHEDULE_CRON`, default `0 2 * * *`)
- Recommended ingestion depth for training readiness: `INGESTION_LIMIT_PER_SYMBOL=5000`
- Staged stochastic gate control: `MARTINGALE_GATE_MODE=bootstrap|strict` (default: `bootstrap`)
- Bootstrap short-horizon promotion set: `BOOTSTRAP_PHASE1_HORIZONS=5m,1h,3h,6h,12h`
- Labeling and confidence controls:
  - `CLASSIFICATION_LABEL_MODE=triple_barrier|terminal_direction`
  - `TRIPLE_BARRIER_SIGMA_MULT=1.0`
  - `REGIME_MODELS_ENABLED=true|false`
  - classification threshold search uses an expanded grid + probability quantiles and optimizes gate-oriented margins
  - classifier training uses dynamic down-class sample-weight boost candidates on imbalanced windows
  - `HIGH_CONFIDENCE_THRESHOLD=0.62`
  - `PREDICTION_CONFIDENCE_MIN_FOR_MODEL=0.56`
  - `PREDICTION_ABSTAIN_TO_FALLBACK=true`
  - `WALK_FORWARD_THRESHOLD_ENABLED=true`
  - `WALK_FORWARD_THRESHOLD_FOLDS=4`
  - `WALK_FORWARD_GATE_MODE=diagnostic|strict`
  - `WALK_FORWARD_GATE_FOLDS=4`
  - `META_LABELING_ENABLED=true`
  - `META_LABEL_MIN_MOVE_BPS=8.0`
  - `META_LABEL_MIN_TAKE_RATE=0.05`
  - `CONFORMAL_ALPHA=0.10`
  - `SLA_MIN_LIVE_SOURCE_RATIO=0.70`
  - `SLA_MAX_STALE_CACHE_RATIO=0.30`
  - quality preflight optionally supports `--max-cross-symbol-lag-steps-map` for freshness parity enforcement
  - `CI_GATE_MAX_AGE_HOURS=24`
  - `EXECUTION_FEE_BPS=4.0`
  - `EXECUTION_SLIPPAGE_BPS=3.0`
  - `EXECUTION_MAX_TURNOVER_PER_STEP=1.0`
  - `PAPER_TRADE_INITIAL_CAPITAL=10000`
  - `PROMOTION_REQUIRE_PNL_ABOVE_BASELINE=true`
  - `PROMOTION_MAX_DRAWDOWN_LIMIT=-0.45`
  - `PROMOTION_GATE_MODE=returns_first`
  - `PROMOTION_REQUIRE_POSITIVE_NET_RETURN=true`
  - `PROMOTION_REQUIRE_SHARPE_ABOVE_BASELINE=true`
  - `PROMOTION_REQUIRE_TOTAL_RETURN_ABOVE_BASELINE=true`
  - `PROMOTION_REQUIRE_CLASSIFICATION_EDGE=false`
  - `PROMOTION_REQUIRE_REGRESSION_EDGE=false`
  - `PROMOTION_MIN_SHARPE_NET=0.0`
  - `FINANCIAL_BASELINE_SELECTION_MODE=fixed|best_validation|best_expost`
  - `FINANCIAL_BASELINE_STRATEGY=always_long|vol_targeted_momentum|vol_targeted_mean_reversion|vol_targeted_kalman_trend`
  - `POSITION_TARGET_ANNUAL_VOL=0.35`
  - `POSITION_MAX_LEVERAGE=1.0`
  - `TRADE_EDGE_UNCERTAINTY_BUFFER_MULT=1.0`
  - `EDGE_ACTION_MIN_TAKE_RATE=0.05`
  - `EDGE_ACTION_COST_RELAX_MULT=0.50`
  - `EDGE_ACTION_THRESHOLD_SCALE=1.0`
  - `TRADE_DIRECTION_POLICY=auto|long_flat|long_only|long_short|short_only`
  - `RETURNS_TUNING_ENABLED=true`
  - `RETURNS_TUNING_POLICIES=long_flat,long_short,long_only`
  - `RETURNS_TUNING_POSITION_SCALE_CANDIDATES=0.5,0.75,1.0,1.25,1.5`
  - `MODEL_CANDIDATE_MODE=theory|expanded`
  - `MODEL_ENABLE_ENSEMBLE_CHALLENGER=true`
  - `ALPHA_KILL_SWITCH_ENABLED=true`
  - `ALPHA_KILL_LOOKBACK_ROWS=360`
  - `ALPHA_KILL_MIN_IC=0.005`
  - `ALPHA_KILL_MIN_DECILE_SPREAD_BPS=0.5`
  - `ALPHA_KILL_MIN_SIGN_ALIGNMENT=0.51`
  - `ALPHA_KILL_MIN_SURVIVAL_RATIO=0.50`
  - `TRAIN_ASOF_SYNC_ENABLED=true`
  - `TRAIN_SYNC_ROW_DEPTH_ENABLED=true`
  - `TRAIN_SYNC_ROW_DEPTH_MIN_COVERAGE_RATIO=1.0`
  - `TRAIN_SYNC_ROW_DEPTH_REQUIRE_ALL_GRANULARITIES=true`
  - `TRAIN_HORIZON_WINDOWING_ENABLED=true`
  - `TRAIN_HORIZON_WINDOW_ROWS_MAP=5m:12000,1h:12000,3h:11000,6h:9000,12h:8000,1d:7000,1w:5000,1mo:3500,3mo:2500`
  - `SHADOW_CHAMPION_PATH=data/models/shadow_champion.json`
  - `SHADOW_PREDICTIONS_PATH=reports/shadow_predictions.jsonl`
  - `SHADOW_REPORT_PATH=reports/shadow_report_latest.json`
  - `SHADOW_SETTLEMENT_GRACE_SECONDS=300`
  - `METRIC_CI_BOOTSTRAP_SAMPLES=400`
  - `METRIC_CI_LEVEL=0.95`
- Flow:
  1. ingest latest candles
  2. train candidate version
  3. run promotion command
  4. keep previous active version for rollback

### One-command daily pipeline

```bash
python scripts/daily_retrain.py --phase phase3
```

Every successful retrain now also emits a robust-validation bundle for the post-promotion active model.

Phase-1 strict startup run (core symbols + core horizons + gates):

```bash
python scripts/daily_retrain.py \
  --phase phase1 \
  --enforce-data-quality \
  --enforce-sla \
  --enforce-ci-gate \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --horizons 6h,12h,1d
```

This command runs:

1. `run_ingestion_cycle()`
2. `scripts/train_all_symbols.py`
3. `scripts/promote_model.py`

Outputs:

- `backend/reports/daily_retrain_<model_version>.json`
- `backend/reports/summary_report_<model_version>.json`
- `backend/reports/promotion_report_<model_version>.json`
- `backend/reports/hypothesis_report_<model_version>.json`
- `backend/reports/oos_validation_<model_version>.json`
- `backend/reports/scorecard_<model_version>.json`
- `backend/reports/robust_validation_<active_model_version>.json`
- `docs/model-cards/<model_version>.md`

Additional operations scripts:

- `python scripts/ci_gate.py --output reports/ci_gate_latest.json`
- `python scripts/daily_scorecard.py --output reports/scorecard_latest.json`
- `python scripts/generate_model_card.py --model-version <version>`

### One-command reproducibility bundle

```bash
python scripts/repro_pipeline.py --phase phase3 --model-version repro-$(date +%Y%m%d-%H%M%S)
```

Bundle output includes command traces, settings snapshot, report checksums, and git commit linkage.

### Continuous Near-Promotion Loop (hourly-friendly)

```bash
python scripts/near_promotion_retrain.py --phase phase3 --symbols BTC-USD,ETH-USD,SOL-USD --max-pairs 6
```

Single-pair sequential run:

```bash
python scripts/near_promotion_retrain.py --phase phase3 --target-pairs BTC-USD:1h
```

Optional near-pass soft-promotion controls (pair-by-pair operations):

```bash
python scripts/near_promotion_retrain.py \
  --phase phase3 \
  --target-pairs ETH-USD:6h \
  --soft-promote-f1-delta-min -0.015 \
  --soft-promote-accuracy-delta-min -0.01 \
  --soft-promote-rmse-delta-min -0.1
```

Behavior:

- selects near-pass candidates from the broadest available `summary_report_*.json` (coverage-aware selection)
- retrains only selected near-promotion symbol+horizon pairs
- carries forward previously promoted symbol+horizon artifacts into the candidate version (copy/retrain) so active promoted pairs never lose artifact availability
- runs `promote_model.py --merge-existing` so previously promoted pairs remain active
- reuses the full symbol/horizon summary tree format so downstream validation tooling can evaluate incremental candidate versions
- writes:
  - `backend/reports/summary_report_<model_version>.json`
  - `backend/reports/promotion_report_<model_version>.json`
  - `backend/reports/near_promotion_<model_version>.json`
  - `backend/reports/hypothesis_report_<model_version>.json`
  - `backend/reports/oos_validation_<model_version>.json`
  - `backend/reports/scorecard_<model_version>.json`
  - `backend/reports/robust_validation_<active_model_version>.json`

## Institutional Evaluation Additions

Per symbol/horizon metrics artifacts now include:

- walk-forward threshold tuning diagnostics (nested threshold selection on time-ordered folds)
- walk-forward strict-gate diagnostics (`strict_pass_all_folds`)
- purged split leakage diagnostics (`required_gap_rows >= steps_ahead`)
- bootstrap confidence intervals for key gate metrics
- execution-aware metrics (fee/slippage/turnover-adjusted return and risk metrics)
- paper-trading metrics (cost-adjusted PnL and tail risk)
- portfolio risk metrics (`VaR`, `CVaR`, `max_drawdown`) from execution-aware return stream

`WALK_FORWARD_GATE_MODE=strict` now acts as the default promotion posture, and `promote_model.py` rejects new candidate artifacts that only have diagnostic walk-forward metadata.

## Experiment Tracking and Rollback Guard

Experiment lineage:

- event log file: `backend/reports/experiment_events.jsonl`
- events are recorded for training, promotion, daily retrain, and rollback guard checks

Registry lineage:

- `backend/data/models/registry.json` now stores promotion/rollback history entries

Safe rollback guard (dry-run by default):

```bash
python scripts/auto_rollback_guard.py --hours 24 --max-stale-cache-ratio 0.35
```

Apply rollback (only if target exists in registry history):

```bash
python scripts/auto_rollback_guard.py --hours 24 --max-stale-cache-ratio 0.35 --apply
```

Automation helper:

- `backend/scripts/run_daily_retrain.sh` runs the near-promotion loop with thread caps and an ingestion-first flow so each scheduled cycle refreshes candles before retraining.

### Frozen champion shadow trading

Freeze the current active model for live shadow validation:

```bash
python scripts/freeze_shadow_champion.py
```

Capture live shadow predictions from the frozen champion:

```bash
python scripts/shadow_trading.py capture
```

Settle matured predictions once the target horizon has elapsed:

```bash
python scripts/shadow_trading.py settle
```

Aggregate the settled rows into a shadow report:

```bash
python scripts/shadow_trading.py report
```

### macOS cron example (daily 2:00 AM local time)

```bash
crontab -e
```

Add:

```cron
0 2 * * * /bin/zsh -lc 'cd /Users/devampatel/Documents/advanced-modular-crypto-prediction-model/backend && source .venv/bin/activate && python scripts/daily_retrain.py --phase phase3 >> /Users/devampatel/Documents/advanced-modular-crypto-prediction-model/backend/reports/cron_daily_retrain.log 2>&1'
```

Example request:

```json
{
  "symbol": "BTC-USD",
  "horizon": "1h",
  "include_debug": true
}
```

Example response fields (abridged):

```json
{
  "direction": "up",
  "market_bias": "bullish",
  "confidence": 0.71,
  "current_price": 62123.11,
  "predicted_close": 123.45,
  "predicted_low_usd": 61220.55,
  "predicted_high_usd": 62880.77,
  "return_range_min_pct": -4.2,
  "return_range_max_pct": 6.1,
  "risk_score": 48.3,
  "risk_level": "medium",
  "horizon_end_at": "2026-02-27T20:15:00+00:00"
}
```
