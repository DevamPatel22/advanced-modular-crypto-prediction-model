# Backend Service

## Overview

This service provides the prediction API backend.

Current scope in this milestone:

- FastAPI application bootstrap
- environment-based settings loading
- health endpoint

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

