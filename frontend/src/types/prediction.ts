export type Symbol = string;
export type Horizon = "5m" | "1h" | "4h";

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
