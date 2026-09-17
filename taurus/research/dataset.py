"""Assembles the cross-symbol training panel.

Features and labels are built per symbol, then stacked into one long frame
indexed by (date, symbol). Training across the whole universe rather than one
model per ticker gives the learner far more examples of the same setups and
stops it memorizing a single name's history.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from ..features.builder import FeatureBuilder
from .labeling import binary_target, triple_barrier_labels

log = logging.getLogger(__name__)


def build_panel(bars_by_symbol: dict[str, pd.DataFrame], config: Config,
                with_labels: bool = True) -> pd.DataFrame:
    """Return a (date, symbol)-indexed frame of features, labels, and ATR."""
    builder = FeatureBuilder(config.features)
    benchmark = bars_by_symbol.get(config.data.benchmark)
    frames: list[pd.DataFrame] = []

    for symbol, bars in bars_by_symbol.items():
        # Don't hand a symbol itself as its own benchmark — the relative
        # strength features would be identically zero and the beta undefined.
        bench = benchmark if symbol != config.data.benchmark else None
        feats = builder.build(bars, benchmark=bench)
        atr_abs = feats.attrs["atr_abs"]

        part = feats.copy()
        part["atr_abs"] = atr_abs
        part["close"] = bars["close"]

        if with_labels:
            labels = triple_barrier_labels(bars, atr_abs, config.labels)
            part = part.join(labels)
            part["target"] = binary_target(part["label"])

        part["symbol"] = symbol
        part.index.name = "date"
        frames.append(part.reset_index().set_index(["date", "symbol"]))

    panel = pd.concat(frames).sort_index()
    log.info("panel: %d rows across %d symbols", len(panel), len(bars_by_symbol))
    return panel


def feature_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Split the panel into a clean X matrix and the list of feature names."""
    reserved = {"label", "label_return", "holding_days", "target",
                "atr_abs", "close", "symbol"}
    cols = [c for c in panel.columns if c not in reserved]
    return panel[cols], cols


def training_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the model cannot learn from.

    Early bars have NaN features from indicator warm-up, and the final
    `max_holding_days` bars of each symbol have no resolved label yet.
    Keeping either would quietly teach the model from garbage.
    """
    _, cols = feature_matrix(panel)
    needed = cols + ["target"]
    return panel.dropna(subset=needed)
