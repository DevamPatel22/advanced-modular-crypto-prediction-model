# Advanced Modular Crypto Prediction Platform

Production-oriented web application for short-horizon cryptocurrency forecasting, with a modular backend, reproducible ML pipeline, and React frontend.

## Vision

Build a reliable, scalable prediction platform that can evolve from MVP analytics to startup-grade production services.

## Tech Stack

- Frontend: React + TypeScript
- Backend API: FastAPI (Python)
- Data/ML: Python pipelines
- Storage: PostgreSQL (planned), SQLite for local MVP milestones

## Current Status

- US-tradable USD crypto universe endpoint and validation in place
- Market data APIs available (candles, ticker, websocket ticker stream)
- Candle ingestion now supports multi-source merge (Coinbase primary + Binance secondary) for better data depth
- Background ingestion loop for tradable symbols enabled via env config
- Model registry + artifact-backed prediction inference with automatic fallback
- All-symbol training and promotion scripts with per-horizon quality gates
- Stochastic feature layer (GBM + Markov regime probabilities) and martingale residual diagnostic in promotion gate
- Triple-barrier classification labels, regime-aware evaluation breakdown, and confidence-slice diagnostics in training reports
- Horizon-specific feature sets, calibrated per-horizon decision thresholds, and regime-routed model artifacts
- Log-return regression target with clipped inverse transform and staged martingale gate (`bootstrap` to `strict`)
- Validation-optimized regression blend (`regression_blend_alpha`) persisted per symbol/horizon for inference stability
- Inference reliability guard: low-confidence model outputs can abstain to safe fallback based on config threshold
- Source credibility safeguards in market data ingestion (integrity/freshness/divergence checks)
- Bootstrap promotions now focus on short horizons first (`5m,1h,3h,6h,12h`) for faster first qualified pairs
- Interactive frontend with:
  - markets homepage (search by symbol or coin name)
  - asset detail page (line/candlestick, ranges, MA/EMA/VA toggles)
  - dedicated prediction model tab
- Prediction response includes risk-aware return ranges and risk score/level
- Prediction response includes explicit bullish/bearish bias, current price anchor, projected USD bounds, and horizon-end timestamp
- Baseline evaluation + full-universe training reports

## Repository Structure

```text
.
├── backend/                # FastAPI service and backend docs
├── docs/                   # Product and implementation documentation
├── frontend/               # React frontend (to be initialized)
├── data/                   # Local datasets and generated artifacts (non-sensitive)
└── scripts/                # Utility scripts and automation helpers
```

## Quick Start (Backend)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- API root: `http://localhost:8000/`
- Health: `http://localhost:8000/api/v1/health`
- Swagger docs: `http://localhost:8000/docs`

## ML Ops Workflow

Daily candidate training:

```bash
cd backend
python scripts/train_all_symbols.py --model-version daily-$(date +%Y%m%d) --output reports/summary_report.json
```

Or run full daily pipeline (ingest -> train -> promote):

```bash
cd backend
python scripts/daily_retrain.py --phase phase3
```

Promotion (phased):

```bash
cd backend
python scripts/promote_model.py --candidate daily-20260213 --phase phase1
python scripts/promote_model.py --candidate <version> --phase phase3 --merge-existing
```

Model artifacts and registry:

```text
backend/data/models/registry.json
backend/data/models/<model_version>/<symbol>/cls_<horizon>.joblib
backend/data/models/<model_version>/<symbol>/reg_<horizon>.joblib
backend/data/models/<model_version>/<symbol>/calibration_<horizon>.json
backend/data/models/<model_version>/<symbol>/metrics_<horizon>.json
```

## Quick Start (Frontend)

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Then open `http://localhost:5173`.

## Product Roadmap

1. Expand daily retraining automation and promotion observability
2. Improve feature engineering and add more model families
3. Add horizon-specific calibration + confidence reliability tracking
4. Introduce alerting and user-level preferences/watchlists
5. Add auth + role model + audit logging
6. Expand testing, CI/CD gates, and deployment hardening

## Engineering Principles

- Modular architecture with clean boundaries
- Reproducible training and evaluation
- Versioned APIs and strict schema validation
- Small, meaningful commits with runnable project state
- Security and operational readiness from early milestones

## Notes

- This repository is intentionally developed in phased milestones.
- Early commits prioritize stable foundations before feature breadth.
- Public prediction API shape remains stable while backend model quality evolves.

## License

This project is proprietary and licensed under a strict All Rights Reserved model.
No copying, redistribution, modification, or reuse is permitted without explicit prior written permission.
