from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Advanced Modular Crypto Prediction API"
    app_env: str = "dev"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    default_horizon: str = "1h"
    default_model_version: str = "baseline-v1"
    supported_symbols: str = "BTC-USD,ETH-USD,SOL-USD"
    supported_horizons: str = "5m,1h,4h"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
