"""The trading agent.

One cycle is: observe the market, decide, act, record. Research (retraining
the model on everything seen so far) runs on its own slower cadence, which is
the sense in which this thing "learns" — the strategy's behavior is a
function of a model that is refit as new data arrives.

The agent is broker-agnostic. Against `PaperBroker` it simulates; against
`AlpacaBroker` it places real orders. Nothing else changes, which is the
point: the code that gets tested is the code that trades.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..config import Config
from ..data.providers import MarketDataProvider, load_universe, make_provider
from ..execution.broker import Broker, Order
from ..research.dataset import build_panel
from ..research.model import AlphaModel
from ..research.walkforward import train_production_model
from ..risk.limits import RiskManager
from ..risk.allocator import orders_to_reach, target_weights
from ..strategy.base import MarketSnapshot, Strategy
from ..strategy.ensemble import EnsembleStrategy
from ..strategy.ml_alpha import MLAlphaStrategy
from ..strategy.momentum import MomentumBreakoutStrategy
from .journal import Journal

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """What one pass of the agent loop did."""
    as_of: date
    equity: float
    signals: int
    orders_submitted: int
    fills: int
    halted: bool
    halt_reason: str = ""

    def __str__(self) -> str:
        state = f"HALTED ({self.halt_reason})" if self.halted else "active"
        return (f"[{self.as_of}] equity={self.equity:,.0f} signals={self.signals} "
                f"orders={self.orders_submitted} fills={self.fills} {state}")


class TradingAgent:
    """Observe -> research -> decide -> act -> record."""

    def __init__(self, config: Config, broker: Broker,
                 provider: MarketDataProvider | None = None,
                 model: AlphaModel | None = None,
                 strategy: Strategy | None = None):
        self.config = config
        self.broker = broker
        self.provider = provider or make_provider(config.data)
        self.model = model
        self.journal = Journal(config.agent.journal_path)
        self.risk = RiskManager(config.risk)
        self.risk.reset(broker.account().equity or config.execution.starting_cash)
        self.strategy = strategy or self._default_strategy()
        self._bars: dict[str, pd.DataFrame] = {}
        self._panel: pd.DataFrame | None = None
        self._cycles_since_retrain = 0

    def _default_strategy(self) -> Strategy:
        """Model plus rules when a model exists; rules alone when it does not.

        Running both and requiring agreement is more conservative than either,
        which is the right default for a leveraged book.
        """
        rules = MomentumBreakoutStrategy(self.config.risk)
        if self.model is None:
            log.info("no model loaded; trading rules only")
            return rules
        return EnsembleStrategy(
            [MLAlphaStrategy(self.model, self.config.risk), rules],
            weights=[0.65, 0.35],
        )

    # --- observe -----------------------------------------------------------

    def observe(self, end: str | None = None) -> None:
        """Refresh market data and rebuild the feature panel."""
        cfg = self.config
        self._bars = load_universe(
            self.provider, cfg.data.universe, cfg.data.history_start,
            end, cfg.data.interval,
        )
        self._panel = build_panel(self._bars, cfg, with_labels=True)
        log.info("observed %d symbols, panel %s", len(self._bars), self._panel.shape)

    # --- research ----------------------------------------------------------

    def research(self, when: date | None = None) -> AlphaModel | None:
        """Retrain the model on everything observed so far.

        A model that fails its out-of-sample gate is rejected rather than
        deployed — the agent keeps trading the previous model, or falls back
        to rules. Trading a model you have measured to be no better than a
        coin flip is worse than trading no model at all.
        """
        if self._panel is None:
            raise RuntimeError("call observe() before research()")

        when = when or date.today()
        try:
            candidate, metrics = train_production_model(self._panel, self.config.model)
        except Exception as exc:
            log.error("training failed: %s", exc)
            self.journal.log_research(when, {"error": str(exc)})
            return self.model

        gate = self.config.model.min_oos_accuracy
        accepted = metrics.is_tradeable(gate)
        log.info("model trained: acc=%.3f auc=%.3f -> %s",
                 metrics.accuracy, metrics.auc,
                 "ACCEPTED" if accepted else "REJECTED")

        top_features: dict[str, float] = {}
        if accepted:
            self.model = candidate
            self.strategy = self._default_strategy()
            try:
                candidate.save(self.config.model.model_dir)
            except Exception as exc:
                log.warning("model save failed: %s", exc)

        self.journal.log_research(
            when,
            {"accuracy": metrics.accuracy, "auc": metrics.auc,
             "brier": metrics.brier, "n_train": metrics.n_train,
             "n_test": metrics.n_test, "accepted": accepted,
             "gate": gate},
            top_features,
        )
        self._cycles_since_retrain = 0
        return self.model

    # --- decide + act ------------------------------------------------------

    def run_cycle(self, as_of: date | None = None) -> CycleResult:
        """One full decision cycle against the live (or paper) broker."""
        if self._panel is None:
            raise RuntimeError("call observe() before run_cycle()")

        as_of = as_of or date.today()
        latest = self._latest_snapshot(as_of)
        equity = self.broker.mark_to_market(latest.prices) if latest else \
            self.broker.account().equity

        account = self.broker.account()
        self.journal.log_equity(
            as_of, equity, account.cash, len(account.positions),
            account.gross_exposure(latest.prices if latest else {}),
            self.risk.drawdown(equity),
        )

        decision = self.risk.check(equity, as_of)
        if decision.halt:
            fills = self.broker.close_all(latest.prices if latest else {},
                                          reason="risk_halt")
            self.journal.log_risk(as_of, "halt", decision.reason)
            return CycleResult(as_of, equity, 0, len(fills), len(fills),
                               halted=True, halt_reason=decision.reason)

        if latest is None:
            log.warning("no market snapshot for %s; skipping cycle", as_of)
            return CycleResult(as_of, equity, 0, 0, 0, halted=False)

        signals = self.strategy.generate(latest)
        orders = self._orders_from_signals(signals, latest.prices, equity,
                                           decision.scale)

        n_fills = 0
        for order in orders:
            fill = self.broker.submit(order)
            if fill:
                n_fills += 1
                self.journal.log_decision(
                    as_of, order.symbol, order.side.value, order.quantity,
                    fill.price, self._confidence_for(signals, order.symbol),
                    order.reason,
                )

        self._cycles_since_retrain += 1
        return CycleResult(as_of, equity, len(signals), len(orders), n_fills,
                           halted=False)

    def run(self, cycles: int = 1, as_of: date | None = None) -> list[CycleResult]:
        """Observe, research if due, then run `cycles` decision cycles."""
        self.observe()
        if self.model is None or self._cycles_since_retrain >= self.config.agent.retrain_every_days:
            self.research(as_of)

        results = []
        for _ in range(cycles):
            results.append(self.run_cycle(as_of))
        return results

    # --- internals ---------------------------------------------------------

    def _latest_snapshot(self, as_of: date) -> MarketSnapshot | None:
        if self._panel is None or self._panel.empty:
            return None
        dates = self._panel.index.get_level_values("date")
        usable = dates[dates <= pd.Timestamp(as_of)]
        if len(usable) == 0:
            return None
        latest = usable.max()

        rows = self._panel.xs(latest, level="date")
        prices = {s: float(rows.at[s, "close"]) for s in rows.index
                  if "close" in rows.columns and pd.notna(rows.at[s, "close"])}
        return MarketSnapshot(
            as_of=latest.date(), features=rows,
            bars={s: b.loc[:latest] for s, b in self._bars.items()},
            prices=prices,
        )

    def _orders_from_signals(self, signals, prices: dict[str, float],
                             equity: float, risk_scale: float) -> list[Order]:
        """Same allocator the backtester uses — deliberately not a second copy."""
        current = self.broker.positions()
        incumbents = {s for s, p in current.items() if p.quantity != 0}
        targets = target_weights(signals, self.config, incumbents, risk_scale)
        return orders_to_reach(targets, current, prices, equity)

    @staticmethod
    def _confidence_for(signals, symbol: str) -> float:
        for sig in signals:
            if sig.symbol == symbol:
                return sig.confidence
        return 0.0
