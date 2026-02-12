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

- Repository scaffolding completed
- Backend service bootstrap completed
- Health endpoint available at `GET /api/v1/health`
- Product scope and milestone roadmap documented

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

## Product Roadmap

1. Backend foundation and configuration
2. Data ingestion and preprocessing pipeline
3. Baseline classification/regression models
4. Stacking ensemble (base + meta model)
5. Prediction API contracts and inference service
6. React frontend prediction workflow
7. Evaluation dashboard and model diagnostics
8. Testing, CI/CD, and deployment hardening

## Engineering Principles

- Modular architecture with clean boundaries
- Reproducible training and evaluation
- Versioned APIs and strict schema validation
- Small, meaningful commits with runnable project state
- Security and operational readiness from early milestones

## Notes

- This repository is intentionally developed in phased milestones.
- Early commits prioritize stable foundations before feature breadth.

