import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { fetchCandles, fetchPrediction, fetchTradableSymbols, getApiBaseUrl } from "./lib/api";
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

function formatCurrency(value: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: value >= 100 ? 2 : 4,
  }).format(value);
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

  const [chartType, setChartType] = useState<ChartType>("line");
  const [range, setRange] = useState<RangeKey>("1W");
  const [showMA, setShowMA] = useState(false);
  const [showEMA, setShowEMA] = useState(true);
  const [showVolumeArea, setShowVolumeArea] = useState(false);
  const [candles, setCandles] = useState<CandlePoint[]>([]);
  const [candlesLoading, setCandlesLoading] = useState(false);
  const [candlesError, setCandlesError] = useState<string | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [liveSource, setLiveSource] = useState<string>("historical");
  const chartRef = useRef<SVGSVGElement | null>(null);

  const closes = useMemo(() => candles.map((candle) => candle.close), [candles]);
  const ma20 = useMemo(() => movingAverage(closes, 20), [closes]);
  const ema50 = useMemo(() => exponentialMovingAverage(closes, 50), [closes]);
  const hovered = hoverIndex !== null ? candles[hoverIndex] : null;

  const latestPrice = useMemo(() => (candles.length > 0 ? candles[candles.length - 1].close : null), [candles]);
  const displayedPrice = livePrice ?? latestPrice;
  const firstPrice = useMemo(() => (candles.length > 0 ? candles[0].open : null), [candles]);
  const priceDelta = displayedPrice !== null && firstPrice !== null ? displayedPrice - firstPrice : null;
  const priceDeltaPct =
    displayedPrice !== null && firstPrice !== null && firstPrice !== 0 ? (priceDelta! / firstPrice) * 100 : null;

  const confidenceLabel = useMemo(() => {
    if (!result) return "-";
    return `${(result.confidence * 100).toFixed(1)}%`;
  }, [result]);

  const projectedLabel = useMemo(() => {
    if (!result) return "-";
    return formatCurrency(result.predicted_close);
  }, [result]);

  const watchlist = useMemo(
    () =>
      symbols.slice(0, 12).map((asset) => {
        const score = asset.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
        const pseudoMove = ((score % 220) - 110) / 10;
        return {
          symbol: asset,
          changePct: pseudoMove,
        };
      }),
    [symbols],
  );

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

  useEffect(() => {
    setLivePrice(null);
    const apiBase = getApiBaseUrl();
    const wsBase = apiBase.replace(/^http/i, "ws");
    const wsUrl = `${wsBase}/api/v1/market-data/ws/ticker?symbol=${encodeURIComponent(symbol)}`;
    let socket: WebSocket | null = null;
    try {
      socket = new WebSocket(wsUrl);
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as { price?: number; source?: string };
          if (typeof payload.price === "number") {
            setLivePrice(payload.price);
            setLiveSource(payload.source ?? "stream");
          }
        } catch {
          return;
        }
      };
      socket.onerror = () => {
        setLiveSource("stream_error");
      };
    } catch {
      setLiveSource("stream_error");
    }

    return () => {
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        socket.close();
      }
    };
  }, [symbol]);

  const chartMetrics = useMemo(() => {
    const width = 980;
    const height = 420;
    const left = 16;
    const right = 16;
    const top = 14;
    const bottom = showVolumeArea ? 90 : 20;
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
    const volumeTop = height - 72;
    const volumeHeight = 58;

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
      top,
      plotWidth,
      plotHeight,
      priceMin,
      priceMax,
      volumeTop,
      volumeHeight,
      xFor,
      yForPrice,
      yForVolume,
    };
  }, [candles, ma20, ema50, showEMA, showMA, showVolumeArea]);

  const closePath = useMemo(() => linePath(closes, chartMetrics.xFor, chartMetrics.yForPrice), [closes, chartMetrics]);
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
    <div className="rh-app">
      <header className="rh-topbar">
        <div className="brand">Lapse Markets</div>
        <nav className="rh-nav">
          <a>Investing</a>
          <a>Charting</a>
          <a>Predictions</a>
        </nav>
      </header>

      <main className="rh-layout">
        <section className="rh-main">
          <div className="rh-price-header">
            <div>
              <h1>{symbol}</h1>
              <div className="price-row">
                <span className="price">{displayedPrice !== null ? formatCurrency(displayedPrice) : "--"}</span>
                <span className={priceDelta !== null && priceDelta >= 0 ? "delta up" : "delta down"}>
                  {priceDelta !== null ? `${priceDelta >= 0 ? "+" : ""}${priceDelta.toFixed(2)}` : "--"}
                  {priceDeltaPct !== null ? ` (${priceDeltaPct >= 0 ? "+" : ""}${priceDeltaPct.toFixed(2)}%)` : ""}
                </span>
              </div>
            </div>
            <div className="symbol-control">
              <label>
                Symbol
                <select value={symbol} onChange={(event) => setSymbol(event.target.value as Symbol)}>
                  {symbols.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
          <p className="muted">US-tradable pairs loaded: {symbols.length} | Price feed: {liveSource}</p>

          <div className="chart-actions">
            <div className="range-buttons">
              {rangeKeys.map((key) => (
                <button
                  key={key}
                  type="button"
                  className={key === range ? "pill active" : "pill"}
                  onClick={() => setRange(key)}
                >
                  {key}
                </button>
              ))}
            </div>
            <div className="tool-buttons">
              <button
                type="button"
                className={chartType === "line" ? "pill active" : "pill"}
                onClick={() => setChartType("line")}
              >
                Line
              </button>
              <button
                type="button"
                className={chartType === "candlestick" ? "pill active" : "pill"}
                onClick={() => setChartType("candlestick")}
              >
                Candles
              </button>
            </div>
          </div>

          <div className="indicator-row">
            <label className="indicator-switch">
              <input type="checkbox" checked={showMA} onChange={(event) => setShowMA(event.target.checked)} />
              MA
            </label>
            <label className="indicator-switch">
              <input type="checkbox" checked={showEMA} onChange={(event) => setShowEMA(event.target.checked)} />
              EMA
            </label>
            <label className="indicator-switch">
              <input type="checkbox" checked={showVolumeArea} onChange={(event) => setShowVolumeArea(event.target.checked)} />
              VA
            </label>
          </div>

          {symbolsLoading ? <p className="muted">Loading tradable USD symbols...</p> : null}
          {symbolsError ? <p className="error">{symbolsError}</p> : null}
          {candlesLoading ? <p className="muted">Loading chart data...</p> : null}
          {candlesError ? <p className="error">{candlesError}</p> : null}

          {!candlesLoading && !candlesError && candles.length > 0 ? (
            <>
              <div className="rh-chart-shell">
                <svg
                  ref={chartRef}
                  viewBox={`0 0 ${chartMetrics.width} ${chartMetrics.height}`}
                  className="rh-chart"
                  role="img"
                  aria-label={`${symbol} chart`}
                  onMouseMove={(event) => handleChartMove(event.clientX)}
                  onMouseLeave={() => setHoverIndex(null)}
                  onTouchMove={(event) => {
                    const touch = event.touches[0];
                    if (touch) handleChartMove(touch.clientX);
                  }}
                >
                  {showVolumeArea && volumeAreaPath ? <path d={volumeAreaPath} className="volume-area" /> : null}

                  {chartType === "line" ? <path d={closePath} className="close-line" /> : null}

                  {chartType === "candlestick"
                    ? candles.map((candle, index) => {
                        const x = chartMetrics.xFor(index);
                        const openY = chartMetrics.yForPrice(candle.open);
                        const closeY = chartMetrics.yForPrice(candle.close);
                        const highY = chartMetrics.yForPrice(candle.high);
                        const lowY = chartMetrics.yForPrice(candle.low);
                        const candleWidth = Math.max(2, chartMetrics.plotWidth / Math.max(candles.length, 90));
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

                  {showMA ? <path d={maPath} className="line-ma" /> : null}
                  {showEMA ? <path d={emaPath} className="line-ema" /> : null}

                  {hovered && hoverIndex !== null ? (
                    <>
                      <line
                        x1={chartMetrics.xFor(hoverIndex)}
                        y1={chartMetrics.top}
                        x2={chartMetrics.xFor(hoverIndex)}
                        y2={chartMetrics.top + chartMetrics.plotHeight}
                        className="hover-guide"
                      />
                      <circle
                        cx={chartMetrics.xFor(hoverIndex)}
                        cy={chartMetrics.yForPrice(hovered.close)}
                        r="4"
                        className="hover-dot"
                      />
                    </>
                  ) : null}
                </svg>
              </div>

              <div className="hover-bar">
                {hovered ? (
                  <>
                    <span>{new Date(hovered.start_time * 1000).toLocaleString()}</span>
                    <span>O {hovered.open.toFixed(2)}</span>
                    <span>H {hovered.high.toFixed(2)}</span>
                    <span>L {hovered.low.toFixed(2)}</span>
                    <span>C {hovered.close.toFixed(2)}</span>
                    <span>V {hovered.volume.toFixed(2)}</span>
                  </>
                ) : (
                  <span>Hover chart to inspect OHLCV values</span>
                )}
              </div>
            </>
          ) : null}
        </section>

        <aside className="rh-side">
          <section className="side-card">
            <h3>Prediction</h3>
            <form className="prediction-form" onSubmit={handleSubmit}>
              <label>
                Horizon
                <select value={horizon} onChange={(event) => setHorizon(event.target.value as Horizon)}>
                  {horizons.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={includeDebug}
                  onChange={(event) => setIncludeDebug(event.target.checked)}
                />
                Include debug metadata
              </label>

              <button type="submit" className="action" disabled={loading}>
                {loading ? "Calculating..." : "Generate Forecast"}
              </button>
            </form>

            {error ? <p className="error">{error}</p> : null}
            {result ? (
              <div className="result-box">
                <p className="signal">{result.direction === "up" ? "Bullish" : "Bearish"}</p>
                <p>Confidence: {confidenceLabel}</p>
                <p>Predicted Close: {projectedLabel}</p>
                <p>Model: {result.model_version}</p>
              </div>
            ) : (
              <p className="muted">Run a prediction to see model output.</p>
            )}
          </section>

          <section className="side-card watchlist">
            <h3>Watchlist</h3>
            <ul>
              {watchlist.map((asset) => (
                <li key={asset.symbol}>
                  <button type="button" onClick={() => setSymbol(asset.symbol)} className="watch-item">
                    <span>{asset.symbol}</span>
                    <span className={asset.changePct >= 0 ? "up" : "down"}>
                      {asset.changePct >= 0 ? "+" : ""}
                      {asset.changePct.toFixed(2)}%
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;
