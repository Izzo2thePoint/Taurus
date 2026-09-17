"""Event-driven backtester.

The timing rule this engine enforces, which is the whole point of it:

    Signals are computed from the CLOSE of bar t.
    Orders are filled at the OPEN of bar t+1.

Filling at the same close that produced the signal is the classic lookahead
bug, and it can turn a losing strategy into a beautiful equity curve. Stops
are also checked against bar t+1's low/high rather than its close, so a gap
through the stop is modeled as a gap, not as a clean exit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..execution.broker import Order, OrderSide
from ..execution.paper import PaperBroker
from ..risk.limits import RiskManager
from ..risk.allocator import orders_to_reach, target_weights
from ..risk.sizing import trailing_stop_price
from ..strategy.base import MarketSnapshot, Strategy
from .metrics import PerformanceReport, build_report

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """One round trip: flat -> position -> flat.

    Scaling in and partial exits are folded into the same record, so a name
    traded up and down over two weeks is one trade, not six. `return_pct` is
    realized P&L over the capital actually committed at the largest point of
    the trade, which is the number that matters for position sizing.
    """
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    quantity: int                 # current open quantity, signed
    return_pct: float | None
    exit_reason: str = ""
    realized_pnl: float = 0.0
    peak_quantity: int = 0        # largest absolute size held during the trade


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    report: PerformanceReport
    trades: list[TradeRecord]
    daily_positions: pd.DataFrame
    halted_at: pd.Timestamp | None = None
    halt_reason: str = ""
    rejected_orders: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [str(self.report)]
        if self.halted_at is not None:
            lines.append(f"\n  RISK HALT on {self.halted_at.date()}: {self.halt_reason}")
        if self.rejected_orders:
            lines.append(f"  Rejected orders: {len(self.rejected_orders)}")
        return "\n".join(lines)


class BacktestEngine:
    """Replays a strategy bar by bar against historical data."""

    def __init__(self, config: Config, strategy: Strategy):
        self.config = config
        self.strategy = strategy
        self.risk = RiskManager(config.risk)

    def run(self, panel: pd.DataFrame, bars_by_symbol: dict[str, pd.DataFrame],
            start: str | None = None, end: str | None = None) -> BacktestResult:
        cfg = self.config
        broker = PaperBroker(
            starting_cash=cfg.execution.starting_cash,
            commission_per_share=cfg.execution.commission_per_share,
            slippage_bps=cfg.execution.slippage_bps,
            max_leverage=cfg.risk.max_gross_leverage,
        )
        self.risk.reset(cfg.execution.starting_cash)

        dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
        if start:
            dates = dates[dates >= pd.Timestamp(start)]
        if end:
            dates = dates[dates <= pd.Timestamp(end)]
        if len(dates) < 2:
            raise ValueError("need at least two bars to backtest")

        equity_history: list[tuple[pd.Timestamp, float]] = []
        position_history: list[dict] = []
        trades: list[TradeRecord] = []
        open_trades: dict[str, TradeRecord] = {}
        pending_orders: list[Order] = []
        exposure_days = 0
        halted_at: pd.Timestamp | None = None
        halt_reason = ""

        for i, today in enumerate(dates):
            opens = self._prices_at(bars_by_symbol, today, "open")
            closes = self._prices_at(bars_by_symbol, today, "close")
            highs = self._prices_at(bars_by_symbol, today, "high")
            lows = self._prices_at(bars_by_symbol, today, "low")
            if not closes:
                continue

            # --- 1. Fill orders decided at yesterday's close, at today's open.
            broker.set_prices(opens or closes, when=today.to_pydatetime())
            for order in pending_orders:
                fill = broker.submit(order)
                if fill:
                    self._record_trade(trades, open_trades, fill, today)
            pending_orders = []

            # --- 2. Intraday stop checks against today's actual range.
            broker.set_prices({**closes}, when=today.to_pydatetime())
            stop_fills = self._check_stops(broker, highs, lows, closes, today)
            for fill in stop_fills:
                self._record_trade(trades, open_trades, fill, today)

            # --- 3. Mark the book at today's close.
            equity = broker.mark_to_market(closes)
            equity_history.append((today, equity))
            if broker.positions():
                exposure_days += 1
            position_history.append({
                "date": today, "equity": equity, "cash": broker.cash,
                "n_positions": len(broker.positions()),
                "gross_exposure": broker.account().gross_exposure(closes),
            })

            # --- 4. Risk gate.
            decision = self.risk.check(equity, today.date())
            if decision.halt:
                if halted_at is None:
                    halted_at, halt_reason = today, decision.reason
                    log.error("halted on %s: %s", today.date(), decision.reason)
                    broker.close_all(closes, reason="risk_halt")
                    for symbol in list(open_trades):
                        self._close_trade(trades, open_trades, symbol, today,
                                          closes.get(symbol, 0.0), "risk_halt")
                continue

            # --- 5. Decide tomorrow's trades from today's close. No peeking:
            #        the snapshot is truncated at `today`.
            if i >= len(dates) - 1:
                break
            if i % max(1, cfg.agent.rebalance_every_days) != 0:
                continue

            snapshot = self._snapshot(panel, bars_by_symbol, today, closes)
            if snapshot is None:
                continue
            signals = self.strategy.generate(snapshot)
            pending_orders = self._build_orders(
                broker, signals, closes, equity, decision.scale, today)

        equity_series = pd.Series(dict(equity_history)).sort_index()
        # Value any still-open trade at the final close so the trade stats
        # are not silently biased by dropping unfinished positions.
        final_date = equity_series.index[-1]
        final_closes = self._prices_at(bars_by_symbol, final_date, "close")
        for symbol in list(open_trades):
            self._close_trade(trades, open_trades, symbol, final_date,
                              final_closes.get(symbol, 0.0), "end_of_backtest")

        returns = [t.return_pct for t in trades if t.return_pct is not None]
        report = build_report(
            equity_series, returns,
            exposure=exposure_days / max(1, len(equity_history)),
        )
        return BacktestResult(
            equity_curve=equity_series, report=report, trades=trades,
            daily_positions=pd.DataFrame(position_history).set_index("date"),
            halted_at=halted_at, halt_reason=halt_reason,
            rejected_orders=broker.rejected,
        )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _prices_at(bars_by_symbol: dict[str, pd.DataFrame], when: pd.Timestamp,
                   column: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for symbol, bars in bars_by_symbol.items():
            if when in bars.index:
                value = bars.at[when, column]
                if np.isfinite(value) and value > 0:
                    out[symbol] = float(value)
        return out

    def _snapshot(self, panel: pd.DataFrame, bars_by_symbol: dict[str, pd.DataFrame],
                  today: pd.Timestamp, closes: dict[str, float]) -> MarketSnapshot | None:
        try:
            todays_rows = panel.xs(today, level="date")
        except KeyError:
            return None
        if todays_rows.empty:
            return None
        return MarketSnapshot(
            as_of=today.date(),
            features=todays_rows,
            bars={s: b.loc[:today] for s, b in bars_by_symbol.items()},
            prices=closes,
        )

    def _build_orders(self, broker: PaperBroker, signals, closes: dict[str, float],
                      equity: float, risk_scale: float,
                      today: pd.Timestamp | None = None) -> list[Order]:
        """Translate signals into orders via the shared allocator."""
        current = broker.positions()
        incumbents = {s for s, p in current.items() if p.quantity != 0}
        targets = target_weights(signals, self.config, incumbents, risk_scale)
        # Positions inside their minimum hold are not closed on a signal;
        # stops and the risk halt bypass this and are handled elsewhere.
        protected = [s for s, p in current.items()
                     if self._within_min_hold(p, today)]
        return orders_to_reach(targets, current, closes, equity,
                               skip_exit=protected)

    def _check_stops(self, broker: PaperBroker, highs: dict[str, float],
                     lows: dict[str, float], closes: dict[str, float],
                     today: pd.Timestamp):
        """Exit positions whose stop was touched during the bar."""
        fills = []
        for symbol, pos in list(broker.positions().items()):
            if pos.quantity == 0:
                continue
            close = closes.get(symbol)
            if close is None:
                continue
            atr_est = pos.avg_price * 0.02  # fallback when no live ATR is at hand
            trail = trailing_stop_price(pos.peak_price or pos.avg_price, atr_est,
                                        self.config.risk, 1 if pos.quantity > 0 else -1)
            low, high = lows.get(symbol, close), highs.get(symbol, close)

            hit = (pos.quantity > 0 and low <= trail) or (pos.quantity < 0 and high >= trail)
            if not hit:
                continue

            # Fill at the stop, or at the open-equivalent worst case if the bar
            # gapped straight through it. Assuming the stop price on a gap is
            # how backtests understate tail losses.
            exit_price = trail if (low <= trail <= high) else close
            broker.set_prices({symbol: exit_price}, when=today.to_pydatetime())
            fill = broker.submit(Order(
                symbol=symbol, quantity=abs(pos.quantity),
                side=OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY,
                reason="trailing_stop"))
            if fill:
                fills.append(fill)
            broker.set_prices({symbol: close}, when=today.to_pydatetime())
        return fills

    def _within_min_hold(self, pos, today: pd.Timestamp | None) -> bool:
        """True if the position is younger than `min_holding_days`.

        Applies to signal-driven exits only. Stops and the risk halt bypass
        this entirely — a minimum hold must never trap a losing position.
        """
        min_days = self.config.risk.min_holding_days
        if min_days <= 0 or today is None or pos.opened_at is None:
            return False
        held = (today.to_pydatetime() - pos.opened_at).days
        return held < min_days

    @staticmethod
    def _record_trade(trades: list[TradeRecord], open_trades: dict[str, TradeRecord],
                      fill, today: pd.Timestamp) -> None:
        """Fold a fill into the symbol's open round trip.

        Partial exits realize P&L on the closed portion and leave the trade
        open; only a fill that takes the position to flat closes the record.
        Getting this wrong makes win rate and profit factor meaningless, since
        every resize would otherwise be counted as a completed trade.
        """
        symbol = fill.symbol
        if fill.quantity == 0:
            return

        record = open_trades.get(symbol)
        if record is None:
            open_trades[symbol] = TradeRecord(
                symbol=symbol, entry_date=today, exit_date=None,
                entry_price=fill.price, exit_price=None,
                quantity=fill.quantity, return_pct=None,
                peak_quantity=abs(fill.quantity),
            )
            return

        if (record.quantity > 0) == (fill.quantity > 0):
            # Scaling in: blend the cost basis.
            total = record.quantity + fill.quantity
            record.entry_price = (
                (record.entry_price * record.quantity + fill.price * fill.quantity) / total)
            record.quantity = total
            record.peak_quantity = max(record.peak_quantity, abs(total))
            return

        # Reducing or closing.
        direction = 1 if record.quantity > 0 else -1
        closed = min(abs(fill.quantity), abs(record.quantity))
        record.realized_pnl += direction * closed * (fill.price - record.entry_price)
        remaining = record.quantity + fill.quantity

        if remaining == 0 or (remaining > 0) != (record.quantity > 0):
            BacktestEngine._finalize(trades, open_trades, symbol, today,
                                     fill.price, fill.order_reason)
            # A fill that flipped the side opens a fresh trade on the remainder.
            if remaining != 0:
                open_trades[symbol] = TradeRecord(
                    symbol=symbol, entry_date=today, exit_date=None,
                    entry_price=fill.price, exit_price=None,
                    quantity=remaining, return_pct=None,
                    peak_quantity=abs(remaining),
                )
        else:
            record.quantity = remaining

    @staticmethod
    def _finalize(trades: list[TradeRecord], open_trades: dict[str, TradeRecord],
                  symbol: str, today: pd.Timestamp, price: float,
                  reason: str) -> None:
        """Close out the open record and compute its realized return."""
        record = open_trades.pop(symbol, None)
        if record is None:
            return
        record.exit_date = today
        record.exit_price = price
        record.exit_reason = reason
        basis = abs(record.peak_quantity) * record.entry_price
        record.return_pct = (record.realized_pnl / basis) if basis > 0 else 0.0
        trades.append(record)

    @staticmethod
    def _close_trade(trades: list[TradeRecord], open_trades: dict[str, TradeRecord],
                     symbol: str, today: pd.Timestamp, price: float,
                     reason: str) -> None:
        """Mark a still-open position out at `price` and close its record.

        Used by the risk halt and at the end of the backtest, where positions
        are liquidated outside the normal fill path.
        """
        record = open_trades.get(symbol)
        if record is None or price <= 0:
            open_trades.pop(symbol, None)
            return
        direction = 1 if record.quantity > 0 else -1
        record.realized_pnl += direction * abs(record.quantity) * (price - record.entry_price)
        BacktestEngine._finalize(trades, open_trades, symbol, today, price, reason)
