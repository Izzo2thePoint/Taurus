"""Provider parsing, against each vendor's real response format.

The network calls themselves are mocked — what these guard is the parsing and
normalization, which is where vendor quirks actually bite.
"""
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from taurus.config import DataConfig
from taurus.data.providers import (CachedProvider, CSVProvider, StooqProvider,
                                   _normalize, load_universe, make_provider)
from tests.synth import make_bars

# Stooq's actual CSV shape: capitalized headers, ISO dates, no adjusted close.
STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,472.16,473.67,470.49,472.65,123184000
2024-01-03,470.43,471.19,468.17,468.79,103350000
2024-01-04,468.3,470.96,467.05,467.28,84232000
2024-01-05,467.55,470.61,466.43,467.92,86420000
"""


def _response(text: str) -> Mock:
    response = Mock()
    response.text = text
    response.raise_for_status = Mock()
    return response


def test_stooq_parses_a_real_response():
    with patch("requests.get", return_value=_response(STOOQ_CSV)):
        df = StooqProvider().history("SPY", "2024-01-01")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 4
    assert df["close"].iloc[0] == pytest.approx(472.65)
    assert df.index[0] == pd.Timestamp("2024-01-02")


def test_stooq_qualifies_us_tickers():
    provider = StooqProvider()
    assert provider._ticker("SPY") == "spy.us"
    assert provider._ticker("spy.us") == "spy.us"      # already qualified
    assert provider._ticker("^SPX") == "^spx"          # index, left alone


def test_stooq_detects_an_unknown_ticker():
    """Stooq answers a bad symbol with HTTP 200 and a one-line body, so the
    status code cannot be trusted."""
    with patch("requests.get", return_value=_response("No data\n")):
        with pytest.raises(ValueError, match="no data"):
            StooqProvider().history("NOTATICKER", "2024-01-01")


def test_stooq_refuses_intraday():
    with pytest.raises(ValueError, match="daily"):
        StooqProvider().history("SPY", "2024-01-01", interval="1h")


def test_stooq_respects_the_date_range():
    with patch("requests.get", return_value=_response(STOOQ_CSV)):
        df = StooqProvider().history("SPY", "2024-01-03", "2024-01-04")
    assert len(df) == 2


def test_normalize_sorts_and_dedupes():
    raw = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [1.0, 2.0, 3.0],
         "Low": [1.0, 2.0, 3.0], "Close": [1.0, 2.0, 3.0],
         "Volume": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-03"]))
    out = _normalize(raw, "AAA")
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert out["close"].loc["2024-01-03"] == 3.0     # last duplicate wins


def test_normalize_strips_timezones():
    idx = pd.to_datetime(["2024-01-02"]).tz_localize("America/New_York")
    raw = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0],
                        "Close": [1.0], "Volume": [1.0]}, index=idx)
    assert _normalize(raw, "AAA").index.tz is None


def test_normalize_rejects_missing_columns():
    raw = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError, match="no"):
        _normalize(raw, "AAA")


def test_normalize_handles_an_empty_frame():
    assert _normalize(pd.DataFrame(), "AAA").empty


def test_cache_avoids_a_second_fetch(tmp_path):
    inner = Mock()
    inner.history.return_value = make_bars(n=120, seed=2)
    cached = CachedProvider(inner, str(tmp_path))

    cached.history("AAA", "2019-01-02")
    cached.history("AAA", "2019-01-02")
    assert inner.history.call_count == 1, "second call should hit the cache"


def test_cache_falls_back_when_the_file_is_corrupt(tmp_path):
    inner = Mock()
    inner.history.return_value = make_bars(n=120, seed=2)
    cached = CachedProvider(inner, str(tmp_path))
    cached.history("AAA", "2019-01-02")

    with open(cached._path("AAA", "1d"), "wb") as fh:
        fh.write(b"this is not parquet")

    assert not cached.history("AAA", "2019-01-02").empty


def test_load_universe_skips_bad_symbols(tmp_path):
    good = make_bars(n=200, seed=1)
    good.to_csv(tmp_path / "GOOD.csv")
    provider = CSVProvider(str(tmp_path))
    out = load_universe(provider, ["GOOD", "MISSING"], "2019-01-02")
    assert set(out) == {"GOOD"}


def test_load_universe_raises_when_nothing_loads(tmp_path):
    with pytest.raises(RuntimeError, match="no symbols"):
        load_universe(CSVProvider(str(tmp_path)), ["NOPE"], "2019-01-02")


def test_make_provider_selects_by_name(tmp_path):
    csv_cfg = DataConfig(provider="csv", csv_dir=str(tmp_path))
    assert isinstance(make_provider(csv_cfg), CSVProvider)

    stooq = make_provider(DataConfig(provider="stooq", cache_dir=str(tmp_path)))
    assert isinstance(stooq, CachedProvider)
    assert isinstance(stooq.inner, StooqProvider)

    unknown = make_provider(DataConfig(provider="nonsense", cache_dir=str(tmp_path)))
    assert isinstance(unknown, CachedProvider)


def test_cache_works_without_pyarrow(tmp_path, monkeypatch):
    """Caching must not depend on an optional package.

    Before this fallback existed, a missing pyarrow made every write fail
    silently and the cache never populated — so every run refetched.
    """
    monkeypatch.setattr("taurus.data.providers._parquet_available", lambda: False)
    inner = Mock()
    inner.history.return_value = make_bars(n=120, seed=4)
    cached = CachedProvider(inner, str(tmp_path))
    assert cached.format == "csv"

    cached.history("AAA", "2019-01-02")
    cached.history("AAA", "2019-01-02")
    assert inner.history.call_count == 1


def test_cached_bars_match_what_was_fetched(tmp_path):
    inner = Mock()
    bars = make_bars(n=120, seed=6)
    inner.history.return_value = bars
    cached = CachedProvider(inner, str(tmp_path))

    first = cached.history("AAA", "2019-01-02")
    second = cached.history("AAA", "2019-01-02")
    pd.testing.assert_frame_equal(first, second, check_freq=False)
