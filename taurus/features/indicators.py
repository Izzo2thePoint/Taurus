"""Technical indicators.

Every function here is causal: the value at bar `t` uses only bars `<= t`.
That property is what makes the backtest honest, so new indicators must
preserve it — no `shift(-n)`, no `center=True` rolling windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return close.pct_change(periods)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder smoothing. Range 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # All-gain windows give a zero average loss; RSI is 100 there, not NaN.
    return out.where(avg_loss.notna() & (avg_loss != 0), 100.0).where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — the volatility unit used for stops and sizing."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (lower, middle, upper) bands."""
    mid = sma(close, period)
    sd = close.rolling(period, min_periods=period).std()
    return mid - num_std * sd, mid, mid + num_std * sd


def bollinger_position(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Where price sits in its band: 0 at the lower band, 1 at the upper."""
    lower, _, upper = bollinger(close, period, num_std)
    width = (upper - lower).replace(0.0, np.nan)
    return ((close - lower) / width).clip(-1.0, 2.0)


def realized_vol(close: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """Annualized realized volatility of daily log returns."""
    vol = log_returns(close).rolling(window, min_periods=window).std()
    return vol * np.sqrt(252) if annualize else vol


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Total return over `window` bars."""
    return close / close.shift(window) - 1.0


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return (series - mean) / sd


def drawdown(close: pd.Series) -> pd.Series:
    """Current drawdown from the running peak, as a negative fraction."""
    return close / close.cummax() - 1.0


def volume_ratio(volume: pd.Series, window: int = 21) -> pd.Series:
    """Today's volume against its recent average — a crude attention proxy."""
    avg = volume.rolling(window, min_periods=window).mean().replace(0.0, np.nan)
    return volume / avg


def donchian_breakout(high: pd.Series, low: pd.Series, close: pd.Series,
                      window: int = 20) -> pd.Series:
    """+1 when close breaks the prior N-bar high, -1 on the low, else 0.

    The window excludes the current bar, otherwise the breakout is compared
    against a high that includes itself and can never trigger.
    """
    prior_high = high.shift(1).rolling(window, min_periods=window).max()
    prior_low = low.shift(1).rolling(window, min_periods=window).min()
    out = pd.Series(0.0, index=close.index)
    out[close > prior_high] = 1.0
    out[close < prior_low] = -1.0
    return out.where(prior_high.notna())


def gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight gap: today's open against yesterday's close."""
    return open_ / close.shift(1) - 1.0
