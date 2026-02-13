export type Symbol = string;
export type Horizon = "5m" | "1h" | "6h" | "12h" | "1d" | "1w" | "1mo" | "3mo";

export interface PredictionRequest {
  symbol: Symbol;
  horizon: Horizon;
  include_debug: boolean;
}

export interface PredictionResponse {
  symbol: Symbol;
  horizon: Horizon;
  direction: "up" | "down";
  confidence: number;
  predicted_close: number;
  return_range_min_pct: number;
  return_range_max_pct: number;
  risk_score: number;
  risk_level: "low" | "medium" | "high";
  model_version: string;
  generated_at: string;
  debug?: Record<string, string> | null;
}

export interface MarketSymbol {
  symbol: string;
  base_currency: string;
  quote_currency: string;
  status: string;
}

export interface MarketSymbolsResponse {
  quote: string;
  count: number;
  symbols: MarketSymbol[];
  source: string;
}

export type CandleGranularity = "1m" | "5m" | "15m" | "1h" | "6h" | "1d";

export interface CandlePoint {
  start_time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CandleSeriesResponse {
  symbol: string;
  granularity: CandleGranularity;
  count: number;
  source: string;
  candles: CandlePoint[];
}
