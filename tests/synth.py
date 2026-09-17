"""Synthetic OHLCV generator for tests and for smoke-testing the pipeline.

Produces bars with a controllable trend and volatility so tests are
deterministic and never touch the network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(n: int = 900, seed: int = 0, drift: float = 0.0004,
              vol: float = 0.016, start: str = "2019-01-02",
              start_price: float = 100.0,
              regime_switch: bool = True) -> pd.DataFrame:
    """Generate one symbol's daily bars as a geometric random walk.

    With `regime_switch`, volatility alternates between calm and stressed
    stretches, which is what separates a strategy that survives from one
    tuned to a single quiet market.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)

    vols = np.full(n, vol)
    if regime_switch:
        pos = 0
        while pos < n:
            span = int(rng.integers(40, 160))
            vols[pos:pos + span] = vol * rng.choice([0.6, 1.0, 1.9])
            pos += span

    rets = rng.normal(drift, vols)
    close = start_price * np.exp(np.cumsum(rets))

    intraday = np.abs(rng.normal(0, vols)) * close
    high = close + intraday * rng.uniform(0.3, 1.0, n)
    low = close - intraday * rng.uniform(0.3, 1.0, n)
    open_ = np.clip(close * (1 + rng.normal(0, vols * 0.5)), low, high)
    volume = rng.integers(1_000_000, 9_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": np.maximum(high, close),
         "low": np.minimum(low, close), "close": close, "volume": volume},
        index=idx,
    )


def make_universe(symbols: list[str], n: int = 900, seed: int = 0,
                  **kwargs) -> dict[str, pd.DataFrame]:
    return {
        sym: make_bars(n=n, seed=seed + i, start_price=50.0 + 25 * i, **kwargs)
        for i, sym in enumerate(symbols)
    }
