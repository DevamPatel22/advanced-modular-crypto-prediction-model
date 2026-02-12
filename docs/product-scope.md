# Advanced Modular Crypto Prediction Platform - Product Scope (v1)

## 1. Product Goal

Build a modular web application that forecasts short-horizon cryptocurrency movement and exposes those forecasts through:

- a backend prediction API,
- a browser-based dashboard,
- reproducible data + model pipelines.

The first release focuses on prediction and analytics. Live trading execution is explicitly out of scope for v1.

## 2. v1 Scope

### 2.1 Supported Markets

- Primary symbols: `BTC-USD`, `ETH-USD`, `SOL-USD`
- Market data format: OHLCV candles
- Initial granularities: `5m`, `1h`, `4h`

### 2.2 Prediction Tasks

- Classification: next candle direction (`up` or `down`)
- Regression: next candle close price
- Multi-horizon outputs: short (`5m`), medium (`1h`), extended (`4h`)

### 2.3 Feature Set

Core feature set for MVP training/inference:

- Trend/momentum: `EMA12`, `EMA26`, `MACD`, `Signal`, `RSI`
- Volatility/price structure: `MA20`, rolling std (`sigma10`), `ATR`
- Volume/flow: `OBV`
- Oscillators: `%K`, `%D`
- Lags: `Lag1`, `Lag2`, `Lag3`

Planned extension after baseline is stable:

- Additional alpha-style factors (momentum, volume, breakout, statistical)
- Optional denoising features

### 2.4 Modeling Approach

Baseline (must ship first):

- one classification model for direction
- one regression model for next close

Planned enhancement:

- stacking ensemble combining multiple base learners and a meta-learner

### 2.5 Validation + Quality

- walk-forward validation with expanding windows
- train/validation/test split with strict time ordering
- logged metrics by symbol and horizon

Minimum tracked metrics:

- Classification: accuracy, precision, recall, F1
- Regression: MAE, RMSE
- Strategy diagnostics (analytics only): hit rate, drawdown, Sharpe-like summary

### 2.6 Risk and Guardrails (Analytics Layer)

- model confidence thresholds configurable per horizon
- max exposure assumptions capped at 25% per position in analytics simulations
- transaction cost and slippage assumptions configurable in backtests

## 3. System Boundaries

### 3.1 In Scope for v1

- data ingestion and preprocessing pipeline
- feature computation
- model training/evaluation scripts
- prediction API
- frontend dashboard for forecasts and metrics
- local reproducibility docs

### 3.2 Out of Scope for v1

- real-money order execution
- portfolio rebalancing automation on exchange accounts
- advanced multi-exchange routing
- mobile application

## 4. Technical Direction

- Frontend: Next.js (TypeScript)
- Backend API: FastAPI (Python)
- Storage:
  - `SQLite` for local MVP
  - optional migration path to `PostgreSQL`
- Model artifacts: versioned files under `backend/models/`
- Project structure:
  - `frontend/`
  - `backend/`
  - `docs/`
  - `data/`
  - `scripts/`

## 5. MVP Success Criteria

v1 is considered complete when all items are true:

1. API can return predictions for `BTC-USD`, `ETH-USD`, and `SOL-USD` for at least one supported horizon.
2. Frontend can request and display predictions with loading/error/result states.
3. Training pipeline is reproducible from documented commands.
4. Evaluation report is generated with core metrics.
5. Repository includes setup instructions and environment examples.

## 6. Milestone Sequence (Commit Plan)

1. Scope and docs foundation
2. Repository scaffold (frontend/backend/docs/data/scripts)
3. Backend bootstrap (health + config)
4. Data pipeline (ingestion + preprocessing)
5. Baseline models (classification + regression)
6. Prediction endpoint integration
7. Frontend prediction flow
8. Evaluation dashboard
9. Tests + CI + deployment docs

