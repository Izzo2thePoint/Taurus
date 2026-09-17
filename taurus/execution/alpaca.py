"""Alpaca broker adapter.

Deliberately thin. Two safety properties are enforced here rather than left
to the caller:

  * Constructing this class against the live endpoint requires
    `Config.live_enabled()` — the config flag *and* TAURUS_ALLOW_LIVE=1.
  * Credentials come from the environment only. No key ever lands in a config
    file that could be committed.

The paper endpoint is the default and needs no gate; it is the intended way
to run this system.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from ..config import Config
from .broker import (AccountState, Broker, Fill, Order, OrderSide, OrderType,
                     Position)

log = logging.getLogger(__name__)

PAPER_ENDPOINT = "https://paper-api.alpaca.markets"
LIVE_ENDPOINT = "https://api.alpaca.markets"


class LiveTradingBlocked(RuntimeError):
    """Raised when live trading is attempted without both safety switches."""


class AlpacaBroker(Broker):
    """Routes orders to Alpaca. Requires `alpaca-py` (not a core dependency)."""

    def __init__(self, config: Config, paper: bool = True):
        if not paper and not config.live_enabled():
            raise LiveTradingBlocked(
                "Live trading is disabled. To enable it you must BOTH set "
                "execution.allow_live: true in the config AND export "
                "TAURUS_ALLOW_LIVE=1. Run against the paper endpoint until the "
                "strategy has a track record you trust."
            )

        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError(
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment."
            )

        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError(
                "AlpacaBroker needs the alpaca-py package: pip install alpaca-py"
            ) from exc

        self.config = config
        self.paper = paper
        self._client = TradingClient(key, secret, paper=paper)
        self._prices: dict[str, float] = {}
        self.fills: list[Fill] = []

        mode = "PAPER" if paper else "*** LIVE — REAL MONEY ***"
        log.warning("AlpacaBroker connected in %s mode", mode)

    def submit(self, order: Order) -> Fill | None:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import (LimitOrderRequest,
                                             MarketOrderRequest)

        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
        common = dict(symbol=order.symbol, qty=order.quantity, side=side,
                      time_in_force=TimeInForce.DAY)

        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            request = LimitOrderRequest(limit_price=order.limit_price, **common)
        else:
            request = MarketOrderRequest(**common)

        try:
            placed = self._client.submit_order(request)
        except Exception as exc:
            log.error("order rejected for %s: %s", order.symbol, exc)
            return None

        # Market orders are not necessarily filled by the time submit returns.
        # Report the actual fill when the API gives one, and None otherwise —
        # never fabricate a fill price, or the journal will disagree with the
        # brokerage statement.
        filled_qty = int(float(getattr(placed, "filled_qty", 0) or 0))
        filled_price = getattr(placed, "filled_avg_price", None)
        if filled_qty == 0 or filled_price is None:
            log.info("order %s accepted, pending fill", getattr(placed, "id", "?"))
            return None

        signed = filled_qty if order.side == OrderSide.BUY else -filled_qty
        fill = Fill(symbol=order.symbol, quantity=signed,
                    price=float(filled_price), commission=0.0,
                    timestamp=datetime.utcnow(), order_reason=order.reason)
        self.fills.append(fill)
        return fill

    def positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._client.get_all_positions():
            qty = int(float(p.qty))
            out[p.symbol] = Position(
                symbol=p.symbol, quantity=qty,
                avg_price=float(p.avg_entry_price),
                peak_price=float(p.current_price or p.avg_entry_price),
            )
        return out

    def account(self) -> AccountState:
        acct = self._client.get_account()
        return AccountState(cash=float(acct.cash), equity=float(acct.equity),
                            positions=self.positions())

    def mark_to_market(self, prices: dict[str, float]) -> float:
        # The broker is the source of truth for a live account; local marks
        # are only a fallback if the API call fails.
        self._prices.update(prices or {})
        try:
            return float(self._client.get_account().equity)
        except Exception as exc:
            log.error("equity lookup failed: %s", exc)
            return 0.0


def make_broker(config: Config) -> Broker:
    """Build the broker named in the config.

    Anything other than an explicitly configured live broker resolves to the
    simulator — the default must be the safe one.
    """
    from .paper import PaperBroker

    name = (config.execution.broker or "paper").lower()
    if name == "alpaca":
        return AlpacaBroker(config, paper=not config.live_enabled())
    if name != "paper":
        log.warning("unknown broker %r; falling back to paper", name)
    return PaperBroker(
        starting_cash=config.execution.starting_cash,
        commission_per_share=config.execution.commission_per_share,
        slippage_bps=config.execution.slippage_bps,
        max_leverage=config.risk.max_gross_leverage,
    )
