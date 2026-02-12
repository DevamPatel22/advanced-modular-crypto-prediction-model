export type Symbol = "BTC-USD" | "ETH-USD" | "SOL-USD";
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
