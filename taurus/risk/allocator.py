"""Turns signals into target weights, and target weights into orders.

This module exists so the backtester and the live agent share one
implementation. If they each had their own, the backtest would eventually
stop describing what the live agent does — and a backtest that does not
describe live behavior is worse than none, because it is trusted.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ..config import Config
from ..execution.broker import Order, OrderSide, Position
from ..strategy.base import Signal
from .sizing import apply_portfolio_caps, position_weight, shares_for_weight


def target_weights(signals: Iterable[Signal], config: Config,
                   incumbents: set[str], risk_scale: float = 1.0
                   ) -> dict[str, float]:
    """Desired portfolio weights for this bar.

    Applies, in order: the entry threshold (with hysteresis for positions
    already held), volatility- and Kelly-scaled sizing, the concentration and
    leverage caps, and finally the drawdown de-risking scalar.
    """
    risk = config.risk
    # Payoff ratio implied by the label barriers — what Kelly needs to know.
    win_loss_ratio = (config.labels.profit_atr_mult / config.labels.stop_atr_mult
                      if config.labels.stop_atr_mult > 0 else 2.0)

    desired: dict[str, float] = {}
    for sig in signals:
        if sig.direction == 0 or sig.confidence <= 0:
            continue

        # Hysteresis: hold an existing position down to a lower bar than the
        # one required to open it, so signals hovering around the threshold
        # do not churn the book and bleed it through slippage.
        threshold = risk.min_signal_confidence
        if sig.symbol in incumbents:
            threshold -= risk.exit_confidence_buffer
        if sig.confidence < threshold:
            continue

        # Size an incumbent as though it just cleared the entry bar, so the
        # same buffer that keeps a position does not shrink it to nothing.
        sizing_conf = (max(sig.confidence, risk.min_signal_confidence)
                       if sig.symbol in incumbents else sig.confidence)

        weight = sig.direction * position_weight(
            sizing_conf, sig.volatility, risk, win_loss_ratio)
        if abs(weight) > 1e-6:
            desired[sig.symbol] = weight

    capped = apply_portfolio_caps(desired, risk, incumbents=incumbents)
    return {k: v * risk_scale for k, v in capped.items()}


def orders_to_reach(targets: Mapping[str, float],
                    current: Mapping[str, Position],
                    prices: Mapping[str, float],
                    equity: float,
                    min_trade_fraction: float = 0.005,
                    skip_exit: Iterable[str] = (),
                    ) -> list[Order]:
    """Orders that move the book from `current` to `targets`.

    `min_trade_fraction` suppresses adjustments smaller than that share of
    equity: rounding-level churn costs real slippage and buys nothing.
    `skip_exit` names positions that must not be closed yet (e.g. inside
    their minimum holding period).
    """
    orders: list[Order] = []
    protected = set(skip_exit)

    # Exit anything no longer wanted.
    for symbol, pos in current.items():
        if symbol in targets or pos.quantity == 0 or symbol in protected:
            continue
        orders.append(Order(
            symbol=symbol, quantity=abs(pos.quantity),
            side=OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY,
            reason="signal_exit"))

    # Enter or resize the rest.
    for symbol, weight in targets.items():
        price = prices.get(symbol)
        if not price or price <= 0:
            continue
        target_shares = shares_for_weight(weight, equity, price)
        held = current[symbol].quantity if symbol in current else 0
        delta = target_shares - held
        if abs(delta) * price < equity * min_trade_fraction:
            continue
        orders.append(Order(
            symbol=symbol, quantity=abs(delta),
            side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            reason="rebalance"))
    return orders
