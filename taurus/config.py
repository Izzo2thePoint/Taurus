"""Configuration for the Taurus trading agent.

Everything that controls how aggressive the system behaves lives here, so the
risk posture of a run is a single reviewable object rather than constants
scattered through the code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml

# Tickers the agent watches by default. Liquid, optionable, high-volume names —
# an aggressive strategy needs to be able to get out as fast as it gets in.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "AVGO", "NFLX", "CRM", "QQQ", "SPY", "SMH", "XLE", "XLF",
]


@dataclass
class DataConfig:
    universe: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    interval: str = "1d"
    history_start: str = "2015-01-01"
    cache_dir: str = "data_cache"
    # Benchmark used for relative-strength features and for reporting alpha.
    benchmark: str = "SPY"
    # "yfinance" for free Yahoo bars, or "csv" to read <csv_dir>/<SYMBOL>.csv.
    # Use csv behind a restricted network, or to feed a paid vendor's export.
    provider: str = "yfinance"
    csv_dir: str = "data_csv"


@dataclass
class FeatureConfig:
    momentum_windows: tuple[int, ...] = (5, 10, 21, 63, 126)
    vol_windows: tuple[int, ...] = (10, 21, 63)
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


@dataclass
class LabelConfig:
    """Triple-barrier labeling parameters.

    A bar is labeled +1 if the profit barrier is hit before the stop barrier
    within `max_holding_days`, -1 if the stop is hit first, 0 if neither.
    Barriers are scaled by realized volatility so the label means the same
    thing across a quiet index and a volatile single name.
    """
    profit_atr_mult: float = 2.0
    stop_atr_mult: float = 1.0
    max_holding_days: int = 10


@dataclass
class ModelConfig:
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    min_samples_leaf: int = 40
    random_state: int = 7
    # Walk-forward evaluation: train on `train_days`, trade the next
    # `test_days`, then roll forward. `embargo_days` are dropped between
    # train and test so forward-looking labels cannot leak into training.
    train_days: int = 756          # ~3 years
    test_days: int = 63            # ~1 quarter
    embargo_days: int = 10
    # A model that cannot beat this out-of-sample accuracy is not traded.
    min_oos_accuracy: float = 0.52
    model_dir: str = "models"


@dataclass
class RiskConfig:
    """Risk limits. These are the guardrails that stay on in every mode.

    `aggression` scales position sizing. 1.0 is a conventional vol-targeted
    book; values above 1 concentrate harder and lever up, bounded by the
    hard caps below. It does not disable any limit.
    """
    aggression: float = 2.0
    target_annual_vol: float = 0.35       # 35% annualized portfolio vol target
    max_gross_leverage: float = 2.0       # sum(|weights|) ceiling
    max_position_weight: float = 0.25     # per-name cap as fraction of equity
    max_positions: int = 8                # concentration: few, larger bets
    kelly_fraction: float = 0.5           # fractional Kelly on model edge

    # Hard stops. Breaching these flattens the book and halts new entries.
    max_drawdown_halt: float = 0.20       # 20% peak-to-trough equity
    daily_loss_halt: float = 0.06         # 6% single-day loss
    per_trade_stop_atr: float = 2.0       # stop distance in ATR units
    trailing_stop_atr: float = 3.0
    # Minimum model confidence required to OPEN a position.
    min_signal_confidence: float = 0.55
    # Hysteresis: an open position is kept until confidence falls this far
    # below the entry threshold. Without it, signals flickering around the
    # threshold churn the book and bleed the account through slippage.
    exit_confidence_buffer: float = 0.05
    # Minimum bars to hold before a signal-driven exit. Stops always
    # override this — a risk exit is never delayed.
    min_holding_days: int = 2


@dataclass
class ExecutionConfig:
    broker: str = "paper"                 # paper | alpaca
    starting_cash: float = 100_000.0
    commission_per_share: float = 0.0
    slippage_bps: float = 5.0             # 5 bps each way, modeled on fills
    # Live trading is off unless BOTH this flag and TAURUS_ALLOW_LIVE=1 are set.
    allow_live: bool = False


@dataclass
class AgentConfig:
    journal_path: str = "runs/journal.jsonl"
    # How often the agent re-runs research and retrains, in trading days.
    retrain_every_days: int = 21
    rebalance_every_days: int = 1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """Load config from YAML, falling back to defaults for absent keys."""
        cfg = cls()
        if not path or not os.path.exists(path):
            return cfg
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        sections = {
            "data": DataConfig, "features": FeatureConfig, "labels": LabelConfig,
            "model": ModelConfig, "risk": RiskConfig,
            "execution": ExecutionConfig, "agent": AgentConfig,
        }
        for name, klass in sections.items():
            if name in raw and isinstance(raw[name], dict):
                current = asdict(getattr(cfg, name))
                current.update(raw[name])
                setattr(cfg, name, klass(**current))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def live_enabled(self) -> bool:
        """Live trading requires the config flag *and* an explicit env var.

        Two independent switches means a stray config edit, or a copied
        example file, cannot by itself put real money at risk.
        """
        return bool(self.execution.allow_live) and os.environ.get("TAURUS_ALLOW_LIVE") == "1"
