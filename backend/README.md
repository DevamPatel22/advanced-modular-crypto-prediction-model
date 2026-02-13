# Backend Service

## Overview

This service provides the prediction API backend.

Current scope:

- FastAPI service with market and prediction APIs
- US-tradable symbol universe validation (`quote=USD`)
- market-data retrieval + SQLite caching
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
- `GET /api/v1/markets/symbols?quote=USD` tradable symbols catalog
- `GET /api/v1/market-data/candles?symbol=BTC-USD&granularity=1h&limit=200` OHLCV candles
- `GET /api/v1/market-data/ticker?symbol=BTC-USD` latest ticker
- `WS /api/v1/market-data/ws/ticker?symbol=BTC-USD` streaming ticker updates
- `POST /api/v1/predictions` prediction output with risk-aware return range

All prediction and market-data endpoints validate symbols against the US-tradable USD pair universe.
Symbols outside that universe return `400`.

Background ingestion loop:

- On startup, the service can continuously refresh candles for US-tradable USD symbols.
- Controlled by `INGESTION_*` env variables in `backend/.env`.

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

Training targets horizons:

- `5m`, `1h`, `6h`, `12h`, `1d`, `1w`, `1mo`, `3mo`

Candidate models per symbol+horizon:

- Classification: Logistic Regression (baseline) + Gradient Boosting
- Regression: Random Forest (baseline) + Gradient Boosting

Stochastic feature layer (used by all model candidates):

- GBM-style drift/volatility features (`mu_20`, `sigma_20`, expected return/variance)
- shock normalization (`shock_z_20`)
- Markov transition probabilities over return regimes (`down/flat/up`)

Promotion gate (must pass all):

- classification `f1 > baseline.f1`
- classification `accuracy > baseline.accuracy`
- regression `rmse < baseline.rmse`
- martingale diagnostic `abs(residual_acf1) <= 0.10`

Baselines:

- direction baseline: always-up
- price baseline: persistence (`next_close = current_close`)

## Promotion / Activation

Promote a trained candidate version:

```bash
python scripts/promote_model.py --candidate daily-20260213-020000 --active daily-20260212-020000 --phase phase1
```

Phases:

- `phase1`: activates only `BTC-USD`, `ETH-USD`, `SOL-USD` entries that pass gates
- `phase2`: activates previous active set + next batch (`--phase2-batch-size`, default 20)
- `phase3`: activates all passed symbol+horizon entries

Any symbol+horizon without promoted artifacts automatically stays on fallback inference.

## Artifact Layout

```text
backend/data/models/
  registry.json
  <model_version>/
    manifest.json
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

## Retraining Cadence

- Target schedule: daily (`RETRAIN_SCHEDULE_CRON`, default `0 2 * * *`)
- Flow:
  1. ingest latest candles
  2. train candidate version
  3. run promotion command
  4. keep previous active version for rollback

### One-command daily pipeline

```bash
python scripts/daily_retrain.py --phase phase3
```

This command runs:

1. `run_ingestion_cycle()`
2. `scripts/train_all_symbols.py`
3. `scripts/promote_model.py`

Outputs:

- `backend/reports/daily_retrain_<model_version>.json`
- `backend/reports/summary_report_<model_version>.json`
- `backend/reports/promotion_report_<model_version>.json`

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
  "confidence": 0.71,
  "predicted_close": 123.45,
  "return_range_min_pct": -4.2,
  "return_range_max_pct": 6.1,
  "risk_score": 48.3,
  "risk_level": "medium"
}
```
