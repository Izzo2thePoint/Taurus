"""Triple-barrier labels must reflect what actually happened next."""
import numpy as np
import pandas as pd

from taurus.config import LabelConfig
from taurus.features import indicators as ind
from taurus.research.labeling import binary_target, triple_barrier_labels
from tests.synth import make_bars


def _bars_and_atr(n=300, seed=2):
    bars = make_bars(n=n, seed=seed)
    return bars, ind.atr(bars["high"], bars["low"], bars["close"])


def test_labels_are_in_range():
    bars, atr = _bars_and_atr()
    labels = triple_barrier_labels(bars, atr)["label"].dropna()
    assert set(labels.unique()).issubset({-1.0, 0.0, 1.0})


def test_holding_never_exceeds_horizon():
    bars, atr = _bars_and_atr()
    cfg = LabelConfig(max_holding_days=7)
    out = triple_barrier_labels(bars, atr, cfg)
    assert out["holding_days"].dropna().max() <= 7


def test_profit_barrier_labels_a_rally():
    """A monotone rally must be labeled +1, not 0 or -1."""
    n = 60
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(np.linspace(100, 160, n), index=idx)
    bars = pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": 1e6}, index=idx)
    atr = pd.Series(1.0, index=idx)
    out = triple_barrier_labels(bars, atr, LabelConfig(2.0, 1.0, 10))
    assert out["label"].iloc[0] == 1.0


def test_stop_barrier_labels_a_crash():
    n = 60
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(np.linspace(160, 100, n), index=idx)
    bars = pd.DataFrame({"open": close, "high": close * 1.005,
                         "low": close * 0.995, "close": close,
                         "volume": 1e6}, index=idx)
    atr = pd.Series(1.0, index=idx)
    out = triple_barrier_labels(bars, atr, LabelConfig(2.0, 1.0, 10))
    assert out["label"].iloc[0] == -1.0


def test_ambiguous_bar_resolves_pessimistically():
    """When both barriers fall inside one bar, assume the stop."""
    idx = pd.bdate_range("2023-01-02", periods=4)
    bars = pd.DataFrame(
        {"open": [100, 100, 100, 100], "high": [100, 110, 100, 100],
         "low": [100, 90, 100, 100], "close": [100, 100, 100, 100],
         "volume": [1e6] * 4}, index=idx)
    atr = pd.Series(2.0, index=idx)
    out = triple_barrier_labels(bars, atr, LabelConfig(2.0, 1.0, 3))
    assert out["label"].iloc[0] == -1.0


def test_binary_target_counts_timeouts_as_losses():
    labels = pd.Series([1.0, 0.0, -1.0, np.nan])
    target = binary_target(labels)
    assert list(target.dropna()) == [1, 0, 0]


def test_final_bars_have_no_label():
    """Bars with no future cannot be labeled and must not leak in as zeros."""
    bars, atr = _bars_and_atr(n=100)
    out = triple_barrier_labels(bars, atr, LabelConfig(max_holding_days=5))
    assert np.isnan(out["label"].iloc[-1])
