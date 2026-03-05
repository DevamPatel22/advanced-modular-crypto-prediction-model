"""Central application configuration loaded from environment variables."""

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
    supported_horizons: str = "5m,1h,3h,6h,12h,1d,1w,1mo,3mo"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    markets_source_url: str = "https://api.exchange.coinbase.com/products"
    symbol_cache_ttl_seconds: int = 300
    market_data_source_base_url: str = "https://api.exchange.coinbase.com"
    market_data_secondary_source_enabled: bool = True
    market_data_secondary_source_base_url: str = "https://api.binance.com"
    market_data_tertiary_source_enabled: bool = True
    market_data_tertiary_source_base_url: str = "https://min-api.cryptocompare.com"
    market_data_tertiary_source_api_key: str = ""
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
    ingestion_limit_per_symbol: int = 5000
    ingestion_cycle_seconds: int = 600
    retrain_schedule_cron: str = "0 2 * * *"
    martingale_gate_mode: str = "bootstrap"
    bootstrap_phase1_horizons: str = "5m,1h,3h,6h,12h"
    classification_label_mode: str = "triple_barrier"
    triple_barrier_sigma_mult: float = 1.0
    regime_models_enabled: bool = True
    high_confidence_threshold: float = 0.62
    prediction_confidence_min_for_model: float = 0.56
    prediction_abstain_to_fallback: bool = True
    walk_forward_threshold_enabled: bool = True
    walk_forward_threshold_folds: int = 4
    walk_forward_gate_mode: str = "diagnostic"
    walk_forward_gate_folds: int = 4
    meta_labeling_enabled: bool = True
    meta_label_min_move_bps: float = 8.0
    meta_label_min_take_rate: float = 0.05
    conformal_alpha: float = 0.10
    execution_fee_bps: float = 4.0
    execution_slippage_bps: float = 3.0
    execution_max_turnover_per_step: float = 1.0
    paper_trade_initial_capital: float = 10000.0
    promotion_require_pnl_above_baseline: bool = True
    promotion_max_drawdown_limit: float = -0.45
    metric_ci_bootstrap_samples: int = 400
    metric_ci_level: float = 0.95

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Get settings."""
    return Settings()
