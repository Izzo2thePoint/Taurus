"""Strategy driven by precomputed out-of-sample model predictions.

This exists to close a lookahead hole that bar-level timing does not catch.

`MLAlphaStrategy` asks a *single* fitted model for a probability. If that
model was trained on the whole history and then backtested over the same
history, every probability it emits is contaminated: the model saw the answer
during training. The bar-level rule (decide on t's close, fill at t+1's open)
is still satisfied, so the backtest looks clean while being fiction.

This strategy instead reads probabilities from the walk-forward run, where
each value was produced by a model trained only on data preceding it. It is
the only honest way to backtest a learned signal.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import RiskConfig
from .base import MarketSnapshot, Signal, Strategy

log = logging.getLogger(__name__)


class WalkForwardAlphaStrategy(Strategy):
    """Emits signals from a (date, symbol) -> probability table."""

    name = "walkforward_alpha"

    def __init__(self, predictions: pd.DataFrame, risk_config: RiskConfig | None = None,
                 proba_column: str = "proba"):
        if not isinstance(predictions.index, pd.MultiIndex):
            raise ValueError("predictions must be indexed by (date, symbol)")
        if proba_column not in predictions.columns:
            raise ValueError(f"predictions has no {proba_column!r} column")

        self.risk = risk_config or RiskConfig()
        # A dict lookup per bar beats slicing a MultiIndex frame thousands of
        # times; the backtest calls this once for every trading day.
        self._proba: dict[pd.Timestamp, dict[str, float]] = {}
        for (when, symbol), value in predictions[proba_column].items():
            if np.isfinite(value):
                self._proba.setdefault(pd.Timestamp(when), {})[symbol] = float(value)

        self.coverage = len(self._proba)
        log.info("walk-forward signals available for %d dates", self.coverage)

    def generate(self, snapshot: MarketSnapshot) -> list[Signal]:
        todays = self._proba.get(pd.Timestamp(snapshot.as_of))
        if not todays:
            # Dates before the first fold's test window have no honest
            # prediction. Emitting nothing is correct — the alternative is to
            # invent a signal the model could not have produced.
            return []

        feats = snapshot.features
        signals: list[Signal] = []

        for symbol, proba in todays.items():
            if proba < self.risk.min_signal_confidence:
                continue
            if symbol not in feats.index:
                continue

            row = feats.loc[symbol]
            price = float(snapshot.prices.get(symbol, 0.0))
            if price <= 0:
                continue

            atr_pct = float(row.get("atr_pct", np.nan))
            vol = float(row.get("vol_21", np.nan))
            if not np.isfinite(vol) or vol <= 0:
                vol = atr_pct * np.sqrt(252) if np.isfinite(atr_pct) else 0.30

            signals.append(Signal(
                symbol=symbol, direction=1, confidence=float(proba),
                atr=float(atr_pct * price) if np.isfinite(atr_pct) else 0.0,
                volatility=vol, price=price,
                rationale=f"walk-forward P(win)={proba:.3f}",
                meta={"proba": float(proba), "oos": True},
            ))

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
