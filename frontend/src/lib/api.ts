import type {
  CandleGranularity,
  CandleSeriesResponse,
  MarketSymbolsResponse,
  PredictionRequest,
  PredictionResponse,
  TickerResponse,
} from "../types/prediction";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export async function fetchPrediction(payload: PredictionRequest): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/predictions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Prediction request failed (${response.status}): ${text}`);
  }

  return response.json() as Promise<PredictionResponse>;
}

export async function fetchTradableSymbols(quote = "USD"): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/markets/symbols?quote=${encodeURIComponent(quote)}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to load symbols (${response.status}): ${text}`);
  }

  const data = (await response.json()) as MarketSymbolsResponse;
  return data.symbols.map((item) => item.symbol);
}

export async function fetchCandles(params: {
  symbol: string;
  granularity: CandleGranularity;
  limit: number;
  refresh?: boolean;
}): Promise<CandleSeriesResponse> {
  const query = new URLSearchParams({
    symbol: params.symbol,
    granularity: params.granularity,
    limit: String(params.limit),
    refresh: String(Boolean(params.refresh)),
  });
  const response = await fetch(`${API_BASE_URL}/api/v1/market-data/candles?${query.toString()}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to load candles (${response.status}): ${text}`);
  }
  return response.json() as Promise<CandleSeriesResponse>;
}

export async function fetchTicker(symbol: string): Promise<TickerResponse> {
  const query = new URLSearchParams({ symbol });
  const response = await fetch(`${API_BASE_URL}/api/v1/market-data/ticker?${query.toString()}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Failed to load ticker (${response.status}): ${text}`);
  }
  return response.json() as Promise<TickerResponse>;
}
