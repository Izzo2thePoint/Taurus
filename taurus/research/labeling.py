"""Triple-barrier labeling.

Rather than asking "does price rise tomorrow", each bar is labeled by what
happens first over the following days: a profit barrier, a stop barrier, or
neither before time runs out. That matches how the strategy actually trades —
entry, target, stop, time limit — so the model learns the decision it will
be asked to make.

Both barriers are set in ATR units, so the label means the same thing on a
quiet index ETF and on a name that moves 6% a day.

Reference: Lopez de Prado, *Advances in Financial Machine Learning*, ch. 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LabelConfig


def triple_barrier_labels(bars: pd.DataFrame, atr_abs: pd.Series,
                          config: LabelConfig | None = None) -> pd.DataFrame:
    """Label every bar by which barrier it touches first.

    Returns a frame with:
      label        +1 profit barrier first, -1 stop first, 0 neither
      label_return the return realized at the touch (or at the horizon)
      holding_days bars held until the touch
    """
    cfg = config or LabelConfig()
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    atr = atr_abs.reindex(bars.index).to_numpy(dtype=float)
    n = len(close)

    labels = np.full(n, np.nan)
    label_ret = np.full(n, np.nan)
    holding = np.full(n, np.nan)
    horizon = cfg.max_holding_days

    for i in range(n):
        # Need a valid ATR at entry and at least one bar of future to look at.
        if not np.isfinite(atr[i]) or atr[i] <= 0 or i + 1 >= n:
            continue
        entry = close[i]
        upper = entry + cfg.profit_atr_mult * atr[i]
        lower = entry - cfg.stop_atr_mult * atr[i]
        end = min(i + horizon, n - 1)

        touched = False
        for j in range(i + 1, end + 1):
            hit_up = high[j] >= upper
            hit_dn = low[j] <= lower
            if hit_up and hit_dn:
                # Both barriers inside one bar. Without intrabar data we
                # cannot know the order, so assume the stop — the pessimistic
                # read, which keeps the backtest from flattering itself.
                labels[i], label_ret[i] = -1.0, (lower / entry) - 1.0
                holding[i] = j - i
                touched = True
                break
            if hit_up:
                labels[i], label_ret[i] = 1.0, (upper / entry) - 1.0
                holding[i] = j - i
                touched = True
                break
            if hit_dn:
                labels[i], label_ret[i] = -1.0, (lower / entry) - 1.0
                holding[i] = j - i
                touched = True
                break

        if not touched:
            # Neither barrier hit: flat label, but keep the realized drift.
            labels[i], label_ret[i] = 0.0, (close[end] / entry) - 1.0
            holding[i] = end - i

    return pd.DataFrame(
        {"label": labels, "label_return": label_ret, "holding_days": holding},
        index=bars.index,
    )


def binary_target(labels: pd.Series) -> pd.Series:
    """Collapse the three-way label to 'did the profit barrier hit first'.

    Timeouts (0) count as failures: an aggressive book that ties up capital
    for ten days and exits flat has lost, in opportunity terms.
    """
    return (labels > 0).astype(int).where(labels.notna())
