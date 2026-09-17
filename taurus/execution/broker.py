"""Broker interface and the order/position value types.

The agent talks only to this interface, so the same code path drives the
backtest, the paper broker, and a live account. That matters: a strategy that
is only ever exercised through a bespoke backtest loop tends to break the
first time it meets a real order router.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    symbol: str
    quantity: int
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    reason: str = ""

    def signed_quantity(self) -> int:
        return self.quantity if self.side == OrderSide.BUY else -self.quantity


@dataclass
class Fill:
    symbol: str
    quantity: int          # signed: positive bought, negative sold
    price: float           # execution price, inclusive of slippage
    commission: float
    timestamp: datetime
    order_reason: str = ""

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    # Tracked for trailing stops: the best price seen since entry.
    peak_price: float = 0.0
    stop_price: float = 0.0
    opened_at: datetime | None = None

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.quantity

    def unrealized_pct(self, price: float) -> float:
        if self.avg_price <= 0:
            return 0.0
        direction = 1 if self.quantity > 0 else -1
        return direction * (price / self.avg_price - 1.0)


@dataclass
class AccountState:
    cash: float
    equity: float
    positions: dict[str, Position] = field(default_factory=dict)

    def gross_exposure(self, prices: dict[str, float]) -> float:
        if self.equity <= 0:
            return 0.0
        gross = sum(abs(p.market_value(prices.get(s, p.avg_price)))
                    for s, p in self.positions.items())
        return gross / self.equity


class Broker(ABC):
    """Minimal surface the agent needs from any execution venue."""

    @abstractmethod
    def submit(self, order: Order) -> Fill | None:
        """Send an order. Returns the fill, or None if it did not execute."""

    @abstractmethod
    def account(self) -> AccountState:
        """Current cash, equity, and open positions."""

    @abstractmethod
    def positions(self) -> dict[str, Position]:
        ...

    @abstractmethod
    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Revalue the book at the given prices and return total equity."""

    def close_all(self, prices: dict[str, float], reason: str = "flatten") -> list[Fill]:
        """Flatten every open position. Used by the risk kill switch."""
        fills: list[Fill] = []
        for symbol, pos in list(self.positions().items()):
            if pos.quantity == 0:
                continue
            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            fill = self.submit(Order(symbol=symbol, quantity=abs(pos.quantity),
                                     side=side, reason=reason))
            if fill:
                fills.append(fill)
        return fills
