"""Strategy interface.

A strategy turns a snapshot of the market into target weights. It does not
know about brokers, cash, or fills — the execution layer handles those — so
the same strategy object runs unchanged in a backtest and against a live
broker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class Signal:
    """One strategy's view on one symbol at one point in time."""
    symbol: str
    direction: int              # +1 long, -1 short, 0 flat
    confidence: float           # 0..1, the model's P(win) where available
    atr: float = 0.0            # for stop placement and sizing
    volatility: float = 0.0     # annualized, for vol targeting
    price: float = 0.0
    rationale: str = ""         # human-readable reason, written to the journal
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction != 0 and self.confidence > 0.0


@dataclass
class MarketSnapshot:
    """Everything a strategy may look at for a given bar.

    `features` and `bars` are already truncated to `as_of`, so a strategy
    physically cannot read the future even if it tries.
    """
    as_of: date
    features: pd.DataFrame          # (symbol) -> latest feature row
    bars: dict[str, pd.DataFrame]   # symbol -> history up to and including as_of
    prices: dict[str, float]

    def symbols(self) -> list[str]:
        return list(self.prices.keys())


class Strategy(ABC):
    """Base class for signal generators."""

    name: str = "strategy"

    @abstractmethod
    def generate(self, snapshot: MarketSnapshot) -> list[Signal]:
        """Return signals for the given bar. May return an empty list."""

    def on_fill(self, symbol: str, quantity: float, price: float) -> None:
        """Hook for strategies that track their own state. Default: no-op."""
