"""The model-driven strategy.

Asks the trained classifier for P(profit barrier before stop) on every symbol
and turns the high-confidence names into long signals. This is the strategy
that carries the "learns from research" part of the system: its behavior
changes every time the model is retrained, without a line of code changing.
"""
from __future__ import annotations

import logging

import numpy as np

from ..config import RiskConfig
from ..research.model import AlphaModel
from .base import MarketSnapshot, Signal, Strategy

log = logging.getLogger(__name__)


class MLAlphaStrategy(Strategy):
    name = "ml_alpha"

    def __init__(self, model: AlphaModel, risk_config: RiskConfig | None = None,
                 allow_shorts: bool = False):
        self.model = model
        self.risk = risk_config or RiskConfig()
        # Shorts are off by default: borrow is not modeled and the loss is
        # unbounded. Turn on only against a broker that reports locate/fees.
        self.allow_shorts = allow_shorts

    def generate(self, snapshot: MarketSnapshot) -> list[Signal]:
        feats = snapshot.features
        if feats.empty:
            return []

        missing = [c for c in self.model.features if c not in feats.columns]
        if missing:
            log.warning("missing features %s; no signals this bar", missing[:5])
            return []

        usable = feats[self.model.features].replace(
            [np.inf, -np.inf], np.nan).dropna()
        if usable.empty:
            return []

        proba = self.model.predict_proba(usable)
        signals: list[Signal] = []

        for symbol, p in zip(usable.index, proba):
            row = feats.loc[symbol]
            atr_pct = float(row.get("atr_pct", np.nan))
            price = float(snapshot.prices.get(symbol, 0.0))
            vol = float(row.get("vol_21", np.nan))
            if not np.isfinite(vol) or vol <= 0:
                vol = atr_pct * np.sqrt(252) if np.isfinite(atr_pct) else 0.30

            if p >= self.risk.min_signal_confidence:
                direction, conf = 1, float(p)
            elif self.allow_shorts and (1.0 - p) >= self.risk.min_signal_confidence:
                direction, conf = -1, float(1.0 - p)
            else:
                continue

            signals.append(Signal(
                symbol=symbol, direction=direction, confidence=conf,
                atr=float(atr_pct * price) if np.isfinite(atr_pct) else 0.0,
                volatility=vol, price=price,
                rationale=f"model P(win)={p:.3f} vol={vol:.2f}",
                meta={"proba": float(p)},
            ))

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
