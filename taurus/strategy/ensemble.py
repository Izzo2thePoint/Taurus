"""Combines several strategies into one set of signals.

Agreement between a learned model and an independent rules-based strategy is
worth more than either alone, so overlapping signals are boosted and
contradictory ones are dropped rather than averaged into a weak position.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import MarketSnapshot, Signal, Strategy


class EnsembleStrategy(Strategy):
    name = "ensemble"

    def __init__(self, strategies: list[Strategy], weights: list[float] | None = None,
                 agreement_bonus: float = 0.05):
        if not strategies:
            raise ValueError("ensemble needs at least one strategy")
        self.strategies = strategies
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        if len(weights) != len(strategies):
            raise ValueError("weights and strategies must be the same length")
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.agreement_bonus = agreement_bonus

    def generate(self, snapshot: MarketSnapshot) -> list[Signal]:
        by_symbol: dict[str, list[tuple[float, Signal]]] = defaultdict(list)
        for weight, strat in zip(self.weights, self.strategies):
            for sig in strat.generate(snapshot):
                by_symbol[sig.symbol].append((weight, sig))

        merged: list[Signal] = []
        for symbol, entries in by_symbol.items():
            directions = {s.direction for _, s in entries}
            if len(directions) > 1:
                # Strategies disagree on the side. Sitting out is free; being
                # wrong with leverage is not.
                continue

            total_w = sum(w for w, _ in entries)
            confidence = sum(w * s.confidence for w, s in entries) / total_w
            if len(entries) > 1:
                confidence = min(0.99, confidence + self.agreement_bonus * (len(entries) - 1))

            first = entries[0][1]
            merged.append(Signal(
                symbol=symbol,
                direction=first.direction,
                confidence=float(confidence),
                atr=float(np.mean([s.atr for _, s in entries])),
                volatility=float(np.mean([s.volatility for _, s in entries])),
                price=first.price,
                rationale=" | ".join(f"{s.rationale}" for _, s in entries),
                meta={"n_strategies": len(entries),
                      "sources": [type(s).__name__ for _, s in entries]},
            ))

        merged.sort(key=lambda s: s.confidence, reverse=True)
        return merged
