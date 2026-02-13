from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.ml.training import all_horizon_specs, evaluate_symbol_horizon


def run_baseline_report(symbol: str, output_path: Path | None = None) -> dict[str, object]:
    normalized_symbol = symbol.upper().strip()
    model_version = "baseline-eval"

    horizons: list[dict[str, object]] = []
    for spec in all_horizon_specs():
        horizons.append(
            evaluate_symbol_horizon(
                symbol=normalized_symbol,
                spec=spec,
                model_version=model_version,
                models_root=Path("data") / "models",
                write_artifacts=False,
            )
        )

    report: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "symbol": normalized_symbol,
        "model_version": model_version,
        "horizons": horizons,
    }

    if output_path is None:
        output_path = Path("reports") / f"baseline_report_{normalized_symbol.lower().replace('-', '_')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
