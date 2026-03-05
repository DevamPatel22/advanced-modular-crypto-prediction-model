#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/devampatel/Documents/advanced-modular-crypto-prediction-model/backend"
LOG_DIR="$PROJECT_DIR/reports"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=4

python scripts/near_promotion_retrain.py \
  --phase phase3 \
  --skip-ingest \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --max-pairs 6 \
  >> "$LOG_DIR/launchd_daily_retrain.log" 2>&1
