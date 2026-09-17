"""Rules-based aggressive momentum/breakout strategy.

Useful on its own and as a baseline: if the machine-learned model cannot beat
this, the extra complexity is not earning its keep. It also gives the agent
something to trade before a model has been trained.

The setup is deliberately aggressive — buy strength, not weakness: a fresh
N-day breakout, in an uptrend, with volume confirming, and volatility not yet
blown out.
"""
from __future__ import annotations

import numpy as np

from ..config import RiskConfig
from .base import MarketSnapshot, Signal, Strategy


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout"

    def __init__(self, risk_config: RiskConfig | None = None,
                 min_momentum: float = 0.02, max_vol: float = 0.90,
                 min_volume_ratio: float = 1.0):
        self.risk = risk_config or RiskConfig()
        self.min_momentum = min_momentum
        self.max_vol = max_vol
        self.min_volume_ratio = min_volume_ratio

    def generate(self, snapshot: MarketSnapshot) -> list[Signal]:
        signals: list[Signal] = []

        for symbol, row in snapshot.features.iterrows():
            mom21 = float(row.get("mom_21", np.nan))
            trend = float(row.get("sma_ratio_20_50", np.nan))
            breakout = float(row.get("breakout_20", 0.0) or 0.0)
            vol = float(row.get("vol_21", np.nan))
            vol_ratio = float(row.get("volume_ratio", 1.0) or 1.0)
            rsi = float(row.get("rsi", np.nan))
            atr_pct = float(row.get("atr_pct", np.nan))
            price = float(snapshot.prices.get(symbol, 0.0))

            if not all(np.isfinite(x) for x in (mom21, trend, vol, price)) or price <= 0:
                continue
            # Skip names whose volatility has already exploded — that is where
            # stops get gapped through and aggressive sizing turns fatal.
            if vol > self.max_vol or vol <= 0:
                continue

            long_setup = (
                mom21 > self.min_momentum
                and trend > 0
                and breakout >= 0
                and vol_ratio >= self.min_volume_ratio
                # An RSI above ~0.80 is a blow-off; chasing it is how momentum
                # books give back a quarter's gains in three sessions.
                and (not np.isfinite(rsi) or rsi < 0.80)
            )
            if not long_setup:
                continue

            # Map setup quality onto the same 0..1 scale the model emits, so
            # both strategies feed sizing identically.
            score = 0.50
            score += min(0.15, mom21 * 0.75)
            score += min(0.10, max(0.0, trend) * 2.0)
            score += 0.08 if breakout > 0 else 0.0
            score += min(0.05, max(0.0, vol_ratio - 1.0) * 0.05)
            confidence = float(np.clip(score, 0.0, 0.95))

            if confidence < self.risk.min_signal_confidence:
                continue

            signals.append(Signal(
                symbol=symbol, direction=1, confidence=confidence,
                atr=float(atr_pct * price) if np.isfinite(atr_pct) else 0.0,
                volatility=vol, price=price,
                rationale=(f"mom21={mom21:.1%} trend={trend:.1%} "
                           f"breakout={breakout:+.0f} volx={vol_ratio:.1f}"),
                meta={"mom_21": mom21, "breakout": breakout},
            ))

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
