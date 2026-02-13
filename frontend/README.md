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

- Prediction request form for symbol and horizon
- Dynamic symbol loader (USD quote pairs from backend markets endpoint)
- API integration with `POST /api/v1/predictions`
- Loading, error, and result states
- Interactive market chart with:
  - range switching (`1D`, `1W`, `1M`, `1Y`, `MAX`)
  - chart type switching (candlestick and line)
  - indicator toggles (`MA`, `EMA`, and volume area)
  - hover inspection for OHLCV values
