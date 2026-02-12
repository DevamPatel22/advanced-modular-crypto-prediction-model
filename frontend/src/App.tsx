import { FormEvent, useMemo, useState } from "react";
import { fetchPrediction } from "./lib/api";
import type { Horizon, PredictionResponse, Symbol } from "./types/prediction";

const symbols: Symbol[] = ["BTC-USD", "ETH-USD", "SOL-USD"];
const horizons: Horizon[] = ["5m", "1h", "4h"];

function App() {
  const [symbol, setSymbol] = useState<Symbol>("BTC-USD");
  const [horizon, setHorizon] = useState<Horizon>("1h");
  const [includeDebug, setIncludeDebug] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const confidenceLabel = useMemo(() => {
    if (!result) return "-";
    return `${(result.confidence * 100).toFixed(1)}%`;
  }, [result]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const prediction = await fetchPrediction({
        symbol,
        horizon,
        include_debug: includeDebug,
      });
      setResult(prediction);
    } catch (submissionError) {
      setResult(null);
      setError(submissionError instanceof Error ? submissionError.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">Prediction Platform</p>
        <h1>Short-Horizon Crypto Forecasts</h1>
        <p className="subtext">
          Generate directional and price forecasts from the backend API and inspect confidence in a clean,
          production-ready interface.
        </p>
      </section>

      <section className="card">
        <h2>Request Prediction</h2>
        <form className="grid" onSubmit={handleSubmit}>
          <label>
            Symbol
            <select value={symbol} onChange={(e) => setSymbol(e.target.value as Symbol)}>
              {symbols.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label>
            Horizon
            <select value={horizon} onChange={(e) => setHorizon(e.target.value as Horizon)}>
              {horizons.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label className="checkbox-row">
            <input type="checkbox" checked={includeDebug} onChange={(e) => setIncludeDebug(e.target.checked)} />
            Include debug metadata
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Calculating..." : "Generate Forecast"}
          </button>
        </form>
      </section>

      <section className="card results">
        <h2>Prediction Result</h2>
        {error ? <p className="error">{error}</p> : null}
        {!error && !result ? <p className="muted">No prediction yet. Submit a request to see results.</p> : null}

        {result ? (
          <dl>
            <div>
              <dt>Direction</dt>
              <dd className={result.direction === "up" ? "up" : "down"}>{result.direction.toUpperCase()}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>{confidenceLabel}</dd>
            </div>
            <div>
              <dt>Predicted Close</dt>
              <dd>{result.predicted_close.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Model Version</dt>
              <dd>{result.model_version}</dd>
            </div>
            <div>
              <dt>Generated At</dt>
              <dd>{new Date(result.generated_at).toLocaleString()}</dd>
            </div>
          </dl>
        ) : null}
      </section>
    </main>
  );
}

export default App;
