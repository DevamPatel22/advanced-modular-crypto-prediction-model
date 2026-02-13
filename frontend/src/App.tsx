import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { fetchCandles, fetchPrediction, fetchTradableSymbols } from "./lib/api";
import type {
  CandleGranularity,
  CandlePoint,
  Horizon,
  PredictionResponse,
  Symbol,
} from "./types/prediction";

const horizons: Horizon[] = ["5m", "1h", "4h"];
const fallbackSymbols: Symbol[] = ["BTC-USD", "ETH-USD", "SOL-USD"];
const rangeKeys = ["1D", "1W", "1M", "1Y", "MAX"] as const;
type RangeKey = (typeof rangeKeys)[number];
type ChartType = "candlestick" | "line";

const rangeConfig: Record<RangeKey, { granularity: CandleGranularity; limit: number }> = {
  "1D": { granularity: "5m", limit: 288 },
  "1W": { granularity: "1h", limit: 168 },
  "1M": { granularity: "1h", limit: 720 },
  "1Y": { granularity: "1d", limit: 365 },
  MAX: { granularity: "1d", limit: 1500 },
};

function movingAverage(values: number[], windowSize: number): Array<number | null> {
  const output: Array<number | null> = Array(values.length).fill(null);
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) {
    sum += values[index];
    if (index >= windowSize) sum -= values[index - windowSize];
    if (index >= windowSize - 1) output[index] = sum / windowSize;
  }
  return output;
}

function exponentialMovingAverage(values: number[], period: number): Array<number | null> {
  const output: Array<number | null> = Array(values.length).fill(null);
  if (values.length === 0) return output;
  const alpha = 2 / (period + 1);
  let ema = values[0];
  for (let index = 0; index < values.length; index += 1) {
    ema = alpha * values[index] + (1 - alpha) * ema;
    if (index >= period - 1) output[index] = ema;
  }
  return output;
}

function linePath(values: Array<number | null>, xAt: (i: number) => number, yAt: (v: number) => number): string {
  let path = "";
  let started = false;
  values.forEach((value, index) => {
    if (value === null || Number.isNaN(value)) {
      started = false;
      return;
    }
    const x = xAt(index);
    const y = yAt(value);
    if (!started) {
      path += `M ${x} ${y}`;
      started = true;
      return;
    }
    path += ` L ${x} ${y}`;
  });
  return path;
}

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
  const [chartType, setChartType] = useState<ChartType>("candlestick");
  const [range, setRange] = useState<RangeKey>("1W");
  const [showMA, setShowMA] = useState(true);
  const [showEMA, setShowEMA] = useState(false);
  const [showVolumeArea, setShowVolumeArea] = useState(true);
  const [candles, setCandles] = useState<CandlePoint[]>([]);
  const [candlesLoading, setCandlesLoading] = useState(false);
  const [candlesError, setCandlesError] = useState<string | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const chartRef = useRef<SVGSVGElement | null>(null);

  const confidenceLabel = useMemo(() => {
    if (!result) return "-";
    return `${(result.confidence * 100).toFixed(1)}%`;
  }, [result]);

  const projectedLabel = useMemo(() => {
    if (!result) return "-";
    return result.predicted_close.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }, [result]);
  const closes = useMemo(() => candles.map((candle) => candle.close), [candles]);
  const ma20 = useMemo(() => movingAverage(closes, 20), [closes]);
  const ema50 = useMemo(() => exponentialMovingAverage(closes, 50), [closes]);
  const hovered = hoverIndex !== null ? candles[hoverIndex] : null;

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

  useEffect(() => {
    let active = true;
    async function loadCandles() {
      setCandlesLoading(true);
      setCandlesError(null);
      setHoverIndex(null);
      try {
        const config = rangeConfig[range];
        const response = await fetchCandles({
          symbol,
          granularity: config.granularity,
          limit: config.limit,
          refresh: true,
        });
        if (!active) return;
        setCandles(response.candles);
      } catch (loadError) {
        if (!active) return;
        setCandles([]);
        setCandlesError(loadError instanceof Error ? loadError.message : "Failed to load chart data");
      } finally {
        if (active) setCandlesLoading(false);
      }
    }

    void loadCandles();
    return () => {
      active = false;
    };
  }, [symbol, range]);

  const chartMetrics = useMemo(() => {
    const width = 980;
    const height = 460;
    const left = 58;
    const right = 18;
    const top = 20;
    const bottom = showVolumeArea ? 112 : 36;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;

    const lows = candles.map((candle) => candle.low);
    const highs = candles.map((candle) => candle.high);
    const indicatorValues = [
      ...(showMA ? ma20.filter((value): value is number => value !== null) : []),
      ...(showEMA ? ema50.filter((value): value is number => value !== null) : []),
    ];
    const minBase = Math.min(...(lows.length > 0 ? lows : [0]), ...(indicatorValues.length > 0 ? indicatorValues : [0]));
    const maxBase = Math.max(...(highs.length > 0 ? highs : [1]), ...(indicatorValues.length > 0 ? indicatorValues : [1]));
    const spread = Math.max(maxBase - minBase, 1e-6);
    const pad = spread * 0.06;
    const priceMin = minBase - pad;
    const priceMax = maxBase + pad;
    const priceRange = Math.max(priceMax - priceMin, 1e-6);
    const maxVolume = Math.max(...candles.map((candle) => candle.volume), 1);
    const volumeTop = height - 92;
    const volumeHeight = 64;

    const xFor = (index: number) => {
      if (candles.length <= 1) return left + plotWidth / 2;
      return left + (index / (candles.length - 1)) * plotWidth;
    };
    const yForPrice = (price: number) => top + ((priceMax - price) / priceRange) * plotHeight;
    const yForVolume = (volume: number) => volumeTop + volumeHeight - (volume / maxVolume) * volumeHeight;

    return {
      width,
      height,
      left,
      right,
      top,
      bottom,
      plotWidth,
      plotHeight,
      priceMin,
      priceMax,
      volumeTop,
      volumeHeight,
      xFor,
      yForPrice,
      yForVolume,
      maxVolume,
    };
  }, [candles, ma20, ema50, showEMA, showMA, showVolumeArea]);

  const closePath = useMemo(
    () => linePath(closes, chartMetrics.xFor, chartMetrics.yForPrice),
    [closes, chartMetrics],
  );
  const maPath = useMemo(() => linePath(ma20, chartMetrics.xFor, chartMetrics.yForPrice), [ma20, chartMetrics]);
  const emaPath = useMemo(() => linePath(ema50, chartMetrics.xFor, chartMetrics.yForPrice), [ema50, chartMetrics]);

  const volumeAreaPath = useMemo(() => {
    if (!showVolumeArea || candles.length === 0) return "";
    const baseY = chartMetrics.volumeTop + chartMetrics.volumeHeight;
    let path = `M ${chartMetrics.xFor(0)} ${baseY}`;
    candles.forEach((candle, index) => {
      path += ` L ${chartMetrics.xFor(index)} ${chartMetrics.yForVolume(candle.volume)}`;
    });
    path += ` L ${chartMetrics.xFor(candles.length - 1)} ${baseY} Z`;
    return path;
  }, [candles, chartMetrics, showVolumeArea]);

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

  function handleChartMove(clientX: number) {
    if (!chartRef.current || candles.length === 0) return;
    const rect = chartRef.current.getBoundingClientRect();
    const relativeX = clientX - rect.left;
    const clamped = Math.max(chartMetrics.left, Math.min(chartMetrics.left + chartMetrics.plotWidth, relativeX));
    const ratio = (clamped - chartMetrics.left) / chartMetrics.plotWidth;
    const index = Math.round(ratio * (candles.length - 1));
    setHoverIndex(Math.max(0, Math.min(candles.length - 1, index)));
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

      <section className="card chart-card">
        <div className="chart-header">
          <h2>{symbol} Market Chart</h2>
          <div className="chart-range-group">
            {rangeKeys.map((key) => (
              <button
                key={key}
                type="button"
                className={key === range ? "chip active" : "chip"}
                onClick={() => setRange(key)}
              >
                {key}
              </button>
            ))}
          </div>
        </div>

        <div className="chart-controls">
          <div className="toggle-group">
            <button
              type="button"
              className={chartType === "candlestick" ? "chip active" : "chip"}
              onClick={() => setChartType("candlestick")}
            >
              Candlestick
            </button>
            <button
              type="button"
              className={chartType === "line" ? "chip active" : "chip"}
              onClick={() => setChartType("line")}
            >
              Line
            </button>
          </div>
          <div className="toggle-group">
            <label className="switch">
              <input type="checkbox" checked={showMA} onChange={(e) => setShowMA(e.target.checked)} />
              <span>MA (20)</span>
            </label>
            <label className="switch">
              <input type="checkbox" checked={showEMA} onChange={(e) => setShowEMA(e.target.checked)} />
              <span>EMA (50)</span>
            </label>
            <label className="switch">
              <input
                type="checkbox"
                checked={showVolumeArea}
                onChange={(e) => setShowVolumeArea(e.target.checked)}
              />
              <span>VA (Volume Area)</span>
            </label>
          </div>
        </div>

        {candlesLoading ? <p className="muted">Loading chart data...</p> : null}
        {candlesError ? <p className="error">{candlesError}</p> : null}

        {!candlesLoading && !candlesError && candles.length > 0 ? (
          <>
            <div className="chart-meta">
              <span>Points: {candles.length}</span>
              <span>
                Range: {chartMetrics.priceMin.toFixed(2)} - {chartMetrics.priceMax.toFixed(2)}
              </span>
              <span>Granularity: {rangeConfig[range].granularity}</span>
            </div>
            <div className="chart-canvas-wrap">
              <svg
                ref={chartRef}
                viewBox={`0 0 ${chartMetrics.width} ${chartMetrics.height}`}
                className="chart-canvas"
                role="img"
                aria-label={`${symbol} price chart`}
                onMouseMove={(event) => handleChartMove(event.clientX)}
                onMouseLeave={() => setHoverIndex(null)}
                onTouchMove={(event) => {
                  const touch = event.touches[0];
                  if (touch) handleChartMove(touch.clientX);
                }}
              >
                <rect
                  x={chartMetrics.left}
                  y={chartMetrics.top}
                  width={chartMetrics.plotWidth}
                  height={chartMetrics.plotHeight}
                  className="chart-plot-bg"
                />
                {[0, 1, 2, 3, 4].map((tick) => {
                  const y = chartMetrics.top + (tick / 4) * chartMetrics.plotHeight;
                  return <line key={tick} x1={chartMetrics.left} y1={y} x2={chartMetrics.left + chartMetrics.plotWidth} y2={y} className="chart-grid" />;
                })}
                {showVolumeArea && volumeAreaPath ? <path d={volumeAreaPath} className="volume-area" /> : null}
                {chartType === "line" ? <path d={closePath} className="price-line" /> : null}
                {chartType === "candlestick"
                  ? candles.map((candle, index) => {
                      const x = chartMetrics.xFor(index);
                      const openY = chartMetrics.yForPrice(candle.open);
                      const closeY = chartMetrics.yForPrice(candle.close);
                      const highY = chartMetrics.yForPrice(candle.high);
                      const lowY = chartMetrics.yForPrice(candle.low);
                      const candleWidth = Math.max(2, chartMetrics.plotWidth / Math.max(candles.length, 80));
                      const topY = Math.min(openY, closeY);
                      const bodyHeight = Math.max(1, Math.abs(closeY - openY));
                      const isUp = candle.close >= candle.open;
                      return (
                        <g key={candle.start_time}>
                          <line x1={x} y1={highY} x2={x} y2={lowY} className={isUp ? "wick up" : "wick down"} />
                          <rect
                            x={x - candleWidth / 2}
                            y={topY}
                            width={candleWidth}
                            height={bodyHeight}
                            className={isUp ? "candle up" : "candle down"}
                          />
                        </g>
                      );
                    })
                  : null}
                {showMA ? <path d={maPath} className="indicator-ma" /> : null}
                {showEMA ? <path d={emaPath} className="indicator-ema" /> : null}
                {hovered && hoverIndex !== null ? (
                  <>
                    <line
                      x1={chartMetrics.xFor(hoverIndex)}
                      y1={chartMetrics.top}
                      x2={chartMetrics.xFor(hoverIndex)}
                      y2={chartMetrics.top + chartMetrics.plotHeight}
                      className="hover-line"
                    />
                    <circle
                      cx={chartMetrics.xFor(hoverIndex)}
                      cy={chartMetrics.yForPrice(hovered.close)}
                      r="4"
                      className="hover-point"
                    />
                  </>
                ) : null}
              </svg>
            </div>
            <div className="chart-inspector">
              {hovered ? (
                <>
                  <span>{new Date(hovered.start_time * 1000).toLocaleString()}</span>
                  <span>O: {hovered.open.toFixed(2)}</span>
                  <span>H: {hovered.high.toFixed(2)}</span>
                  <span>L: {hovered.low.toFixed(2)}</span>
                  <span>C: {hovered.close.toFixed(2)}</span>
                  <span>V: {hovered.volume.toFixed(2)}</span>
                </>
              ) : (
                <span>Hover chart to inspect candle values.</span>
              )}
            </div>
          </>
        ) : null}
      </section>
    </main>
  );
}

export default App;
