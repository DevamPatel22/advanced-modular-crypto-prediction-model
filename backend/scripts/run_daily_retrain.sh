#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/devampatel/Documents/advanced-modular-crypto-prediction-model/backend"
LOG_DIR="$PROJECT_DIR/reports"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"
source .venv/bin/activate
python scripts/daily_retrain.py \
  --phase phase1 \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  >> "$LOG_DIR/launchd_daily_retrain.log" 2>&1
