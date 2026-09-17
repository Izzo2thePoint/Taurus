"""Market data acquisition.

A provider returns a tidy OHLCV frame indexed by timestamp. Everything
downstream depends only on this shape, so swapping yfinance for a paid feed
means writing one class, not touching the rest of the system.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class MarketDataProvider(ABC):
    """Source of historical bars for a single symbol."""

    @abstractmethod
    def history(self, symbol: str, start: str, end: str | None = None,
                interval: str = "1d") -> pd.DataFrame:
        """Return OHLCV bars indexed by tz-naive timestamp, oldest first."""


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce a raw vendor frame into the canonical OHLCV shape."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    # yfinance returns a MultiIndex when asked for multiple symbols, and
    # sometimes even for one. Flatten to the per-symbol level.
    if isinstance(df.columns, pd.MultiIndex):
        levels = df.columns.get_level_values(-1)
        if symbol in set(levels):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
    # Prefer split/dividend-adjusted closes when the vendor supplies them
    # separately; unadjusted prices produce phantom gaps that look like alpha.
    if "adj_close" in df.columns and "close" in df.columns:
        ratio = (df["adj_close"] / df["close"]).replace([float("inf")], 1.0).fillna(1.0)
        for col in ("open", "high", "low"):
            if col in df.columns:
                df[col] = df[col] * ratio
        df["close"] = df["adj_close"]

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: provider returned no {missing} column(s)")

    out = df[OHLCV_COLUMNS].copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])
    return out.astype(float)


class YFinanceProvider(MarketDataProvider):
    """Free end-of-day and intraday bars via Yahoo Finance.

    Fine for research and paper trading. Swap it for a paid feed before
    trading real size — Yahoo revises and back-fills without notice.
    """

    def history(self, symbol: str, start: str, end: str | None = None,
                interval: str = "1d") -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            symbol, start=start, end=end, interval=interval,
            auto_adjust=False, progress=False, threads=False,
        )
        return _normalize(raw, symbol)


class CSVProvider(MarketDataProvider):
    """Reads bars from `<directory>/<symbol>.csv`.

    Used for reproducible backtests and for tests that must not touch the
    network.
    """

    def __init__(self, directory: str):
        self.directory = directory

    def history(self, symbol: str, start: str, end: str | None = None,
                interval: str = "1d") -> pd.DataFrame:
        path = os.path.join(self.directory, f"{symbol}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _normalize(raw, symbol)
        return df.loc[str(start):str(end)] if end else df.loc[str(start):]


class CachedProvider(MarketDataProvider):
    """Wraps a provider with an on-disk parquet cache.

    Re-downloading years of bars on every research iteration is slow and
    rude to the vendor. Cached files older than `max_age_hours` are refetched
    so a live loop still sees today's bar.
    """

    def __init__(self, inner: MarketDataProvider, cache_dir: str,
                 max_age_hours: float = 12.0):
        self.inner = inner
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> str:
        safe = symbol.replace("/", "_").replace("^", "_")
        return os.path.join(self.cache_dir, f"{safe}_{interval}.parquet")

    def _is_fresh(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        return age < timedelta(hours=self.max_age_hours)

    def history(self, symbol: str, start: str, end: str | None = None,
                interval: str = "1d") -> pd.DataFrame:
        path = self._path(symbol, interval)
        if self._is_fresh(path):
            try:
                df = pd.read_parquet(path)
                sliced = df.loc[str(start):str(end)] if end else df.loc[str(start):]
                if not sliced.empty:
                    return sliced
            except Exception as exc:  # corrupt cache should never be fatal
                log.warning("cache read failed for %s (%s); refetching", symbol, exc)

        df = self.inner.history(symbol, start, end, interval)
        if not df.empty:
            try:
                df.to_parquet(path)
            except Exception as exc:
                log.warning("cache write failed for %s: %s", symbol, exc)
        return df


def load_universe(provider: MarketDataProvider, symbols: list[str], start: str,
                  end: str | None = None, interval: str = "1d",
                  ) -> dict[str, pd.DataFrame]:
    """Fetch bars for every symbol, skipping those that fail.

    One dead ticker should not abort a research run over fifteen good ones,
    so failures are logged and dropped rather than raised.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = provider.history(sym, start, end, interval)
        except Exception as exc:
            log.warning("skipping %s: %s", sym, exc)
            continue
        if df.empty or len(df) < 30:
            log.warning("skipping %s: only %d bars", sym, len(df))
            continue
        out[sym] = df
    if not out:
        raise RuntimeError("no symbols loaded; check the universe and data provider")
    return out


def make_provider(config) -> MarketDataProvider:
    """Build the data provider named in the config, wrapped in the disk cache.

    `config` is a DataConfig. An unknown name falls back to yfinance with a
    warning rather than failing at import time.
    """
    name = (getattr(config, "provider", "yfinance") or "yfinance").lower()
    if name == "csv":
        # Local files are already on disk; a second cache layer buys nothing.
        return CSVProvider(config.csv_dir)
    if name not in ("yfinance", "yahoo"):
        log.warning("unknown data provider %r; using yfinance", name)
    return CachedProvider(YFinanceProvider(), config.cache_dir)
