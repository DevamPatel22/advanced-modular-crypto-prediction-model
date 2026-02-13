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
- Prediction result UI includes risk-aware return range and risk score/level
