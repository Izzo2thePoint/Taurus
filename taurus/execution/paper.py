"""Simulated broker with explicit cost modeling.

Backtests that fill at the untouched close and charge nothing are the single
most common way an aggressive strategy looks profitable on paper and loses
money live. This broker charges slippage on every fill and commission per
share, and refuses orders it cannot fund.
"""
from __future__ import annotations

import logging
from datetime import datetime

from .broker import (AccountState, Broker, Fill, Order, OrderSide, OrderType,
                     Position)

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    """Fills orders against the current mark, with slippage and commission."""

    def __init__(self, starting_cash: float = 100_000.0,
                 commission_per_share: float = 0.0,
                 slippage_bps: float = 5.0,
                 allow_margin: bool = True,
                 max_leverage: float = 2.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps
        self.allow_margin = allow_margin
        self.max_leverage = max_leverage

        self._positions: dict[str, Position] = {}
        self._prices: dict[str, float] = {}
        self._equity = starting_cash
        self.fills: list[Fill] = []
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.rejected: list[tuple[str, str]] = []
        self.now: datetime = datetime.utcnow()

    # --- pricing -----------------------------------------------------------

    def set_prices(self, prices: dict[str, float], when: datetime | None = None) -> None:
        self._prices.update({k: float(v) for k, v in prices.items() if v and v > 0})
        if when is not None:
            self.now = when

    def _fill_price(self, symbol: str, side: OrderSide) -> float | None:
        mark = self._prices.get(symbol)
        if not mark or mark <= 0:
            return None
        # Slippage always works against us — buys fill higher, sells lower.
        slip = mark * (self.slippage_bps / 10_000.0)
        return mark + slip if side == OrderSide.BUY else mark - slip

    # --- order handling ----------------------------------------------------

    def submit(self, order: Order) -> Fill | None:
        if order.quantity <= 0:
            return None

        price = self._fill_price(order.symbol, order.side)
        if price is None:
            self.rejected.append((order.symbol, "no mark price"))
            return None

        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            # Conservative: a limit only fills if the mark is already through
            # it. Assuming otherwise invents fills that would not have happened.
            if order.side == OrderSide.BUY and price > order.limit_price:
                return None
            if order.side == OrderSide.SELL and price < order.limit_price:
                return None

        signed_qty = order.signed_quantity()
        commission = abs(signed_qty) * self.commission_per_share
        cost = signed_qty * price + commission

        if not self._can_afford(order, price, cost):
            self.rejected.append((order.symbol, "insufficient buying power"))
            log.warning("rejected %s %s x%d: insufficient buying power",
                        order.side.value, order.symbol, order.quantity)
            return None

        mark = self._prices[order.symbol]
        self.total_slippage += abs(price - mark) * abs(signed_qty)
        self.total_commission += commission
        self.cash -= cost
        self._apply_fill(order.symbol, signed_qty, price)

        fill = Fill(symbol=order.symbol, quantity=signed_qty, price=price,
                    commission=commission, timestamp=self.now,
                    order_reason=order.reason)
        self.fills.append(fill)
        self.mark_to_market(self._prices)
        return fill

    def _can_afford(self, order: Order, price: float, cost: float) -> bool:
        """Buying-power check, including the leverage ceiling."""
        existing = self._positions.get(order.symbol)
        # Closing or reducing a position always frees capital.
        if existing is not None:
            reducing = (existing.quantity > 0 and order.side == OrderSide.SELL) or \
                       (existing.quantity < 0 and order.side == OrderSide.BUY)
            if reducing and order.quantity <= abs(existing.quantity):
                return True

        if not self.allow_margin:
            return cost <= self.cash

        equity = self.mark_to_market(self._prices)
        if equity <= 0:
            return False
        projected_gross = sum(
            abs(p.quantity * self._prices.get(s, p.avg_price))
            for s, p in self._positions.items()
        ) + abs(order.quantity * price)
        return projected_gross <= equity * self.max_leverage

    def _apply_fill(self, symbol: str, signed_qty: int, price: float) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            self._positions[symbol] = Position(
                symbol=symbol, quantity=signed_qty, avg_price=price,
                peak_price=price, opened_at=self.now,
            )
            return

        old_qty = pos.quantity
        new_qty = old_qty + signed_qty

        if old_qty == 0:
            pos.quantity, pos.avg_price, pos.peak_price = signed_qty, price, price
            pos.opened_at = self.now
        elif (old_qty > 0) == (signed_qty > 0):
            # Adding to the position: weighted-average the cost basis.
            pos.avg_price = (old_qty * pos.avg_price + signed_qty * price) / new_qty
            pos.quantity = new_qty
        else:
            # Reducing or flipping: realize P&L on the closed portion.
            closed = min(abs(signed_qty), abs(old_qty))
            direction = 1 if old_qty > 0 else -1
            self.realized_pnl += direction * closed * (price - pos.avg_price)
            pos.quantity = new_qty
            if new_qty == 0:
                del self._positions[symbol]
                return
            if (new_qty > 0) != (old_qty > 0):
                pos.avg_price, pos.peak_price = price, price
                pos.opened_at = self.now

        if pos.quantity > 0:
            pos.peak_price = max(pos.peak_price or price, price)
        elif pos.quantity < 0:
            pos.peak_price = min(pos.peak_price or price, price)

    # --- state -------------------------------------------------------------

    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def mark_to_market(self, prices: dict[str, float]) -> float:
        if prices:
            self._prices.update({k: float(v) for k, v in prices.items() if v and v > 0})
        holdings = sum(p.quantity * self._prices.get(s, p.avg_price)
                       for s, p in self._positions.items())
        self._equity = self.cash + holdings
        for symbol, pos in self._positions.items():
            price = self._prices.get(symbol)
            if not price:
                continue
            if pos.quantity > 0:
                pos.peak_price = max(pos.peak_price or price, price)
            else:
                pos.peak_price = min(pos.peak_price or price, price)
        return self._equity

    def account(self) -> AccountState:
        return AccountState(cash=self.cash, equity=self._equity,
                            positions=self.positions())

    @property
    def equity(self) -> float:
        return self._equity

    def total_return(self) -> float:
        return self._equity / self.starting_cash - 1.0
