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
- API integration with `POST /api/v1/predictions`
- Loading, error, and result states
