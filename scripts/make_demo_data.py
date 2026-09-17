#!/usr/bin/env python3
"""Generate a synthetic market in data_csv/ so the pipeline can be run offline.

    python scripts/make_demo_data.py --symbols 8 --years 10

Use this to exercise `research`, `backtest`, and `signals` without a data
vendor — on a restricted network, in CI, or before you have decided which
feed to pay for.

These are random walks with volatility regimes. Results on them say whether
the machinery works, never whether a strategy is any good.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DEFAULT_NAMES = ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG",
                 "HHH", "III", "JJJ", "KKK"]


def generate(symbol: str, n: int, seed: int, start: str,
             start_price: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)

    # Alternating calm and stressed volatility regimes.
    vol = np.full(n, 0.014)
    pos = 0
    while pos < n:
        span = int(rng.integers(60, 200))
        vol[pos:pos + span] = 0.014 * rng.choice([0.6, 1.0, 1.8])
        pos += span

    rets = rng.normal(0.0003, vol)
    close = start_price * np.exp(np.cumsum(rets))

    intraday = np.abs(rng.normal(0, vol)) * close
    high = close + intraday * rng.uniform(0.3, 1.0, n)
    low = close - intraday * rng.uniform(0.3, 1.0, n)
    open_ = np.clip(close * (1 + rng.normal(0, vol * 0.5)), low, high)

    return pd.DataFrame(
        {"open": open_, "high": np.maximum(high, close),
         "low": np.minimum(low, close), "close": close,
         "volume": rng.integers(1_000_000, 9_000_000, n).astype(float)},
        index=idx,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data_csv")
    parser.add_argument("--symbols", type=int, default=8)
    parser.add_argument("--years", type=float, default=10.0)
    parser.add_argument("--start", default="2015-01-02")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = DEFAULT_NAMES[:args.symbols]
    n = int(args.years * 252)

    for i, symbol in enumerate(names):
        frame = generate(symbol, n, args.seed + i, args.start, 50.0 + 25.0 * i)
        frame.to_csv(os.path.join(args.out, f"{symbol}.csv"))

    print(f"Wrote {len(names)} symbols x {n} bars to {args.out}/")
    print(f"Universe: {names}")
    print("\nNext:")
    print("  python -m taurus.cli research --config config.offline.yaml")
    print("  python -m taurus.cli backtest --config config.offline.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
