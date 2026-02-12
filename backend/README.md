# Backend Service

## Overview

This service provides the prediction API backend.

Current scope in this milestone:

- FastAPI application bootstrap
- environment-based settings loading
- health endpoint
- prediction endpoint contract (service stub)

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
- `POST /api/v1/predictions` prediction response (stub until model integration)

Example request:

```json
{
  "symbol": "BTC-USD",
  "horizon": "1h",
  "include_debug": true
}
```
