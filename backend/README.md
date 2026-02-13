# Backend Service

## Overview

This service provides the prediction API backend.

Current scope:

- FastAPI service with market and prediction APIs
- US-tradable symbol universe validation (`quote=USD`)
- market-data retrieval + SQLite caching
- background ingestion loop for tradable symbols
- risk-aware prediction range output
- baseline model training/evaluation CLI

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
