"""Turns raw OHLCV bars into the model's feature matrix.

The contract: row `t` of the returned frame contains only information
knowable at the close of bar `t`. Labels are attached separately, in
`taurus.research.labeling`, and are the only forward-looking column.
"""
from __future__ import annotations

import pandas as pd

from ..config import FeatureConfig
from . import indicators as ind


class FeatureBuilder:
    """Builds a per-symbol feature frame from OHLCV bars."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def build(self, bars: pd.DataFrame, benchmark: pd.DataFrame | None = None
              ) -> pd.DataFrame:
        cfg = self.config
        o, h, l, c, v = (bars["open"], bars["high"], bars["low"],
                         bars["close"], bars["volume"])
        f = pd.DataFrame(index=bars.index)

        # --- Trend / momentum: does this thing keep going? -------------------
        for w in cfg.momentum_windows:
            f[f"mom_{w}"] = ind.momentum(c, w)
        # Medium-term momentum with the most recent week removed. Computed
        # from prices rather than from other feature columns, which silently
        # became a constant 0 whenever `momentum_windows` did not contain
        # both 21 and 5.
        f["mom_21_ex_5"] = ind.momentum(c, 21) - ind.momentum(c, 5)
        f["sma_ratio_20_50"] = ind.sma(c, 20) / ind.sma(c, 50) - 1.0
        f["sma_ratio_50_200"] = ind.sma(c, 50) / ind.sma(c, 200) - 1.0
        f["price_vs_sma20"] = c / ind.sma(c, 20) - 1.0

        macd_line, macd_sig, macd_hist = ind.macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        # Scale MACD by price so the feature is comparable across a $20 and a
        # $900 stock.
        f["macd"] = macd_line / c
        f["macd_signal"] = macd_sig / c
        f["macd_hist"] = macd_hist / c

        # --- Mean reversion / stretch ---------------------------------------
        f["rsi"] = ind.rsi(c, cfg.rsi_period) / 100.0
        f["bb_pos"] = ind.bollinger_position(c, cfg.bb_period)
        f["ret_zscore_21"] = ind.zscore(ind.returns(c), 21)
        f["drawdown"] = ind.drawdown(c)

        # --- Volatility: sets position size and barrier width ---------------
        atr_abs = ind.atr(h, l, c, cfg.atr_period)
        f["atr_pct"] = atr_abs / c
        for w in cfg.vol_windows:
            f[f"vol_{w}"] = ind.realized_vol(c, w)
        # Vol-of-vol regime: is volatility itself expanding? Uses the
        # shortest and longest configured windows rather than hardcoded ones,
        # which raised KeyError the moment `vol_windows` was changed.
        if len(cfg.vol_windows) >= 2:
            short, long = min(cfg.vol_windows), max(cfg.vol_windows)
            f[f"vol_ratio_{short}_{long}"] = (
                f[f"vol_{short}"] / f[f"vol_{long}"].replace(0.0, pd.NA))

        # --- Participation ---------------------------------------------------
        f["volume_ratio"] = ind.volume_ratio(v, 21)
        f["gap"] = ind.gap(o, c)
        f["breakout_20"] = ind.donchian_breakout(h, l, c, 20)
        f["breakout_55"] = ind.donchian_breakout(h, l, c, 55)
        f["intraday_range"] = (h - l) / c

        # --- Relative strength vs the benchmark ------------------------------
        # An aggressive book wants names beating the index, not just names
        # that happen to rise with it.
        if benchmark is not None and not benchmark.empty:
            bench = benchmark["close"].reindex(bars.index).ffill()
            for w in (21, 63):
                f[f"rel_strength_{w}"] = ind.momentum(c, w) - ind.momentum(bench, w)
            f["beta_63"] = (
                ind.returns(c).rolling(63, min_periods=63).cov(ind.returns(bench))
                / ind.returns(bench).rolling(63, min_periods=63).var()
            )

        # ATR in price terms is kept for sizing and stops, not as a feature —
        # it is unscaled and would let the model infer the price level.
        f.attrs["atr_abs"] = atr_abs
        return f.replace([float("inf"), float("-inf")], pd.NA)

    @staticmethod
    def feature_columns(frame: pd.DataFrame) -> list[str]:
        """Model input columns: everything that is not a label or bookkeeping."""
        reserved = {"label", "label_return", "holding_days", "target",
                    "symbol", "atr_abs", "close"}
        return [c for c in frame.columns if c not in reserved]
