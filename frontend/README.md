# Frontend (React)

## Setup

```bash
npm install
cp .env.example .env
npm run dev
```

Default app URL: `http://localhost:5173`

## Environment

- `VITE_API_BASE_URL` backend base URL, default `http://localhost:8000`

## Current Features

- Markets homepage listing US-tradable USD crypto pairs
- Search by symbol or full coin name (case-insensitive, alias-friendly)
- Asset detail page with:
  - live price header
  - range switching (`1D`, `1W`, `1M`, `1Y`, `MAX`)
  - chart type switching (candlestick and line)
  - indicator toggles (`MA`, `EMA`, and volume area)
  - hover inspection for OHLCV values
- Dedicated prediction model tab/page
- Prediction result UI includes bullish/bearish bias, current price, predicted close, USD range, horizon-end time, and risk score/level
- Prediction values are backend model-driven when promoted artifacts exist, with automatic fallback continuity
- Prediction horizons supported in UI: `5m`, `1h`, `3h`, `6h`, `12h`, `1d`, `1w`, `1mo`, `3mo`
