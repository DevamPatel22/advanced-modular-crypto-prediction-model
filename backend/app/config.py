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
    shadow_champion_path: str = "data/models/shadow_champion.json"
    shadow_predictions_path: str = "reports/shadow_predictions.jsonl"
    shadow_report_path: str = "reports/shadow_report_latest.json"
    shadow_settlement_grace_seconds: int = 300
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
    bootstrap_phase1_horizons: str = "6h,12h,1d"
    phase1_focus_symbols: str = "BTC-USD,ETH-USD,SOL-USD"
    phase1_focus_horizons: str = "6h,12h,1d"
    classification_label_mode: str = "triple_barrier"
    triple_barrier_sigma_mult: float = 1.0
    regime_models_enabled: bool = True
    high_confidence_threshold: float = 0.62
    prediction_confidence_min_for_model: float = 0.56
    prediction_abstain_to_fallback: bool = True
    walk_forward_threshold_enabled: bool = True
    walk_forward_threshold_folds: int = 4
    walk_forward_gate_mode: str = "strict"
    walk_forward_gate_folds: int = 4
    meta_labeling_enabled: bool = True
    meta_label_min_move_bps: float = 8.0
    meta_label_min_take_rate: float = 0.05
    conformal_alpha: float = 0.10
    execution_fee_bps: float = 4.0
    execution_slippage_bps: float = 3.0
    execution_max_turnover_per_step: float = 1.0
    paper_trade_initial_capital: float = 10000.0
    sla_min_live_source_ratio: float = 0.70
    sla_max_stale_cache_ratio: float = 0.30
    ci_gate_max_age_hours: int = 24
    promotion_require_pnl_above_baseline: bool = True
    promotion_max_drawdown_limit: float = -0.45
    metric_ci_bootstrap_samples: int = 400
    metric_ci_level: float = 0.95
    # Returns-first promotion controls.
    promotion_gate_mode: str = "returns_first"
    promotion_require_positive_net_return: bool = True
    promotion_require_sharpe_above_baseline: bool = True
    promotion_require_total_return_above_baseline: bool = True
    promotion_require_classification_edge: bool = False
    promotion_require_regression_edge: bool = False
    promotion_min_sharpe_net: float = 0.0
    financial_baseline_selection_mode: str = "fixed"
    financial_baseline_strategy: str = "always_long"
    # Position sizing and decision policy controls.
    position_target_annual_vol: float = 0.35
    position_max_leverage: float = 1.0
    trade_edge_uncertainty_buffer_mult: float = 1.0
    edge_action_min_take_rate: float = 0.05
    edge_action_cost_relax_mult: float = 0.50
    edge_action_threshold_scale: float = 1.0
    trade_direction_policy: str = "auto"
    baseline_risk_matched: bool = True
    returns_tuning_enabled: bool = True
    returns_tuning_policies: str = "long_flat,long_short,long_only"
    returns_tuning_position_scale_candidates: str = "0.5,0.75,1.0,1.25,1.5"
    # Candidate model family controls.
    model_candidate_mode: str = "theory"
    model_enable_ensemble_challenger: bool = True
    # Hypothesis governance controls.
    alpha_kill_switch_enabled: bool = True
    alpha_kill_lookback_rows: int = 360
    alpha_kill_min_ic: float = 0.005
    alpha_kill_min_decile_spread_bps: float = 0.5
    alpha_kill_min_sign_alignment: float = 0.51
    alpha_kill_min_survival_ratio: float = 0.50
    alpha_kill_require_financial_pass: bool = True
    alpha_kill_min_signal_net_mean_return: float = 0.0
    alpha_kill_min_signal_sharpe_net: float = 0.0
    alpha_kill_max_signal_drawdown_limit: float = -0.45
    # As-of synchronization for training data.
    train_asof_sync_enabled: bool = True
    # Row-depth synchronization for cross-symbol parity windows.
    train_sync_row_depth_enabled: bool = True
    train_sync_row_depth_min_coverage_ratio: float = 1.0
    train_sync_row_depth_require_all_granularities: bool = True
    # Horizon-specific recent-window caps prevent stale regimes from diluting short/medium horizon training.
    train_horizon_windowing_enabled: bool = True
    train_horizon_window_rows_map: str = "5m:12000,1h:12000,3h:11000,6h:9000,12h:8000,1d:7000,1w:5000,1mo:3500,3mo:2500"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Get settings."""
    return Settings()
