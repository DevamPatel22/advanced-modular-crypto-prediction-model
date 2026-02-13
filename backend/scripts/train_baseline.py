#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.baseline_training import run_baseline_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate baseline models and emit metrics report")
    parser.add_argument("--symbol", default="BTC-USD", help="Symbol to evaluate (default: BTC-USD)")
    parser.add_argument(
        "--output",
        default="reports/baseline_report.json",
        help="Output JSON report path (default: reports/baseline_report.json)",
    )
    args = parser.parse_args()

    report = run_baseline_report(symbol=args.symbol, output_path=Path(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
