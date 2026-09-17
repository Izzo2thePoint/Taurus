"""Indicators must be causal and numerically sane."""
import numpy as np
import pandas as pd
import pytest

from taurus.features import indicators as ind
from tests.synth import make_bars


@pytest.fixture
def bars():
    return make_bars(n=400, seed=5)


def test_rsi_bounded(bars):
    rsi = ind.rsi(bars["close"]).dropna()
    assert not rsi.empty
    assert rsi.between(0, 100).all()


def test_atr_positive(bars):
    atr = ind.atr(bars["high"], bars["low"], bars["close"]).dropna()
    assert (atr > 0).all()


def test_indicators_are_causal(bars):
    """Appending future bars must not change any historical value.

    This is the property the entire backtest rests on. If an indicator
    leaks, every performance number downstream is fiction.
    """
    cutoff = 300
    past = bars.iloc[:cutoff]
    checks = {
        "rsi": lambda d: ind.rsi(d["close"]),
        "atr": lambda d: ind.atr(d["high"], d["low"], d["close"]),
        "macd": lambda d: ind.macd(d["close"])[0],
        "bb_pos": lambda d: ind.bollinger_position(d["close"]),
        "mom_21": lambda d: ind.momentum(d["close"], 21),
        "vol_21": lambda d: ind.realized_vol(d["close"], 21),
        "breakout": lambda d: ind.donchian_breakout(d["high"], d["low"], d["close"]),
    }
    for name, fn in checks.items():
        on_full = fn(bars).iloc[:cutoff]
        on_past = fn(past)
        pd.testing.assert_series_equal(
            on_full, on_past, check_names=False,
            obj=f"{name} changed when future bars were appended",
        )


def test_drawdown_non_positive(bars):
    assert (ind.drawdown(bars["close"]) <= 1e-12).all()


def test_donchian_excludes_current_bar():
    """A breakout compared against a window containing itself can never fire."""
    close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20], dtype=float)
    high = close * 1.0
    low = close * 0.99
    out = ind.donchian_breakout(high, low, close, window=5)
    assert out.iloc[-1] == 1.0


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(close).dropna().iloc[-1] == pytest.approx(100.0)
