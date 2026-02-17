from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Advanced Modular Crypto Prediction API"
    app_env: str = "dev"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    default_horizon: str = "1h"
    default_model_version: str = "daily-bootstrap"
    supported_symbols: str = "BTC-USD,ETH-USD,SOL-USD"
    supported_horizons: str = "5m,1h,6h,12h,1d,1w,1mo,3mo"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    markets_source_url: str = "https://api.exchange.coinbase.com/products"
    symbol_cache_ttl_seconds: int = 300
    market_data_source_base_url: str = "https://api.exchange.coinbase.com"
    market_data_sqlite_path: str = "data/market_data.db"
    model_artifacts_root: str = "data/models"
    model_registry_path: str = "data/models/registry.json"
    market_data_default_granularity: str = "1h"
    market_data_default_limit: int = 200
    ticker_stream_interval_seconds: int = 2
    ingestion_enabled: bool = True
    ingestion_quote_currency: str = "USD"
    ingestion_symbol_limit: int = 120
    ingestion_granularities: str = "1m,1h,1d"
    ingestion_limit_per_symbol: int = 200
    ingestion_cycle_seconds: int = 600
    retrain_schedule_cron: str = "0 2 * * *"
    martingale_gate_mode: str = "bootstrap"
    bootstrap_phase1_horizons: str = "5m,1h,6h,12h"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
