import { FormEvent, useEffect, useMemo, useState } from "react";
import { fetchPrediction, fetchTradableSymbols } from "./lib/api";
import type { Horizon, PredictionResponse, Symbol } from "./types/prediction";

const horizons: Horizon[] = ["5m", "1h", "4h"];
const fallbackSymbols: Symbol[] = ["BTC-USD", "ETH-USD", "SOL-USD"];

function App() {
  const [symbols, setSymbols] = useState<Symbol[]>(fallbackSymbols);
  const [symbol, setSymbol] = useState<Symbol>("BTC-USD");
  const [horizon, setHorizon] = useState<Horizon>("1h");
  const [includeDebug, setIncludeDebug] = useState(false);
  const [loading, setLoading] = useState(false);
  const [symbolsLoading, setSymbolsLoading] = useState(true);
  const [symbolsError, setSymbolsError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const confidenceLabel = useMemo(() => {
    if (!result) return "-";
    return `${(result.confidence * 100).toFixed(1)}%`;
  }, [result]);

  const projectedLabel = useMemo(() => {
    if (!result) return "-";
    return result.predicted_close.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }, [result]);

  useEffect(() => {
    let active = true;
    async function loadSymbols() {
      setSymbolsLoading(true);
      setSymbolsError(null);
      try {
        const liveSymbols = await fetchTradableSymbols("USD");
        if (!active) return;
        if (liveSymbols.length > 0) {
          setSymbols(liveSymbols);
          if (!liveSymbols.includes(symbol)) {
            setSymbol(liveSymbols[0]);
          }
        } else {
          setSymbols(fallbackSymbols);
        }
      } catch (loadError) {
        if (!active) return;
        setSymbolsError(loadError instanceof Error ? loadError.message : "Failed to load symbols");
        setSymbols(fallbackSymbols);
      } finally {
        if (active) setSymbolsLoading(false);
      }
    }

    void loadSymbols();
    return () => {
      active = false;
    };
  }, []);

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
        <div className="hero-chips">
          <span>Live API Contract</span>
          <span>Multi-Horizon</span>
          <span>Risk-Aware Roadmap</span>
        </div>
      </section>

      <section className="layout-grid">
        <article className="card request-card">
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
            {symbolsLoading ? <p className="muted">Loading tradable USD symbols...</p> : null}
            {symbolsError ? <p className="error">{symbolsError}</p> : null}

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
        </article>

        <article className={`card results ${result ? "panel-enter" : ""}`}>
          <h2>Prediction Result</h2>
          {error ? <p className="error">{error}</p> : null}
          {!error && !result ? <p className="muted">No prediction yet. Submit a request to see results.</p> : null}

          {result ? (
            <>
              <div className={`direction-banner ${result.direction}`}>
                {result.direction === "up" ? "Bullish Signal" : "Bearish Signal"}
              </div>
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
                  <dd>{projectedLabel}</dd>
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
            </>
          ) : null}
        </article>
      </section>
    </main>
  );
}

export default App;
