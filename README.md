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
- Near-promotion tuning path: expanded threshold optimization, classifier probability blending, and dynamic class-balance weighting
- Horizon-specific feature sets, calibrated per-horizon decision thresholds, and regime-routed model artifacts
- Dual regression target selection (`log_return` and residual-from-persistence) with clipped transforms and staged martingale gate (`bootstrap` to `strict`)
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

Fast near-promotion cycle (for continuous incremental promotion work on key symbols):

```bash
cd backend
python scripts/near_promotion_retrain.py --phase phase3 --skip-ingest --symbols BTC-USD,ETH-USD,SOL-USD --max-pairs 6
```

This loop auto-selects candidates from the broadest available summary report (not just the latest tiny near-loop report), then retrains near-pass pairs and merges newly qualified promotions into the active registry.
It also carries forward prior promoted artifacts so promoted pairs stay deployable across incremental near-loop model versions.
For pair-by-pair promotion workflows, use `--target-pairs SYMBOL:HORIZON` (with optional soft-promotion thresholds).

Data-first workflow before strict promotion loops:

```bash
cd backend
python scripts/backfill_market_data.py \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --granularities 1m,5m,15m,1h,6h,1d \
  --target-rows-map 1m:30000,5m:18000,15m:12000,1h:6000,6h:2500,1d:1500

python scripts/data_quality_report.py \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --granularities 1m,5m,15m,1h,6h,1d

python scripts/daily_retrain.py --phase phase3 --enforce-data-quality --symbols BTC-USD,ETH-USD,SOL-USD
```

This sequence hardens training inputs and avoids promoting models trained on stale/thin data.

Operational readiness endpoints:

- `GET /api/v1/health/data-readiness`
- `GET /api/v1/health/source-health?hours=24`

These expose data-quality gate status and live source reliability telemetry.

Institutional-grade additions now included:

- walk-forward threshold tuning + optional strict walk-forward gate enforcement
- purged split leakage diagnostics + bootstrap confidence intervals
- execution-aware evaluation metrics (fee/slippage/turnover-adjusted)
- paper-trading PnL metrics (equity, drawdown, VaR/CVaR, turnover)
- portfolio risk API (`/api/v1/risk/portfolio-snapshot`, `/api/v1/risk/limit-check`)
- expanded free proxy features for derivatives/microstructure/on-chain signals
- experiment event logging and safe auto-rollback guard tooling
- immutable artifact checksum manifests for each model version

One-command reproducibility run (ingest -> quality-gated retrain -> promote -> rollback guard dry-run):

```bash
cd backend
python scripts/repro_pipeline.py --phase phase3 --model-version repro-$(date +%Y%m%d-%H%M%S)
```

This generates a bundle report with command traces, settings snapshot, report checksums, and git commit linkage.

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

## Presentation Docs

- model risks and failure modes: `docs/model-risk-and-failure-modes.md`
- interview-ready architecture narrative: `docs/interview-narrative.md`

## Notes

- This repository is intentionally developed in phased milestones.
- Early commits prioritize stable foundations before feature breadth.
- Public prediction API shape remains stable while backend model quality evolves.

## License

This project is proprietary and licensed under a strict All Rights Reserved model.
No copying, redistribution, modification, or reuse is permitted without explicit prior written permission.
