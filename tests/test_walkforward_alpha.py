"""The walk-forward strategy must never emit a signal it could not have known.

These tests guard the model-level lookahead hole: bar-level timing can be
perfect while the model itself was trained on the period being backtested.
"""
import pandas as pd
import pytest

from taurus.config import Config, ModelConfig, RiskConfig
from taurus.research.dataset import build_panel
from taurus.research.walkforward import run_walk_forward
from taurus.strategy.base import MarketSnapshot
from taurus.strategy.walkforward_alpha import WalkForwardAlphaStrategy
from tests.synth import make_universe

SYMBOLS = ["SPY", "AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def market():
    cfg = Config()
    cfg.data.universe = SYMBOLS
    cfg.data.benchmark = "SPY"
    bars = make_universe(SYMBOLS, n=1400, seed=53)
    return cfg, bars, build_panel(bars, cfg)


@pytest.fixture(scope="module")
def walk_forward(market):
    _, _, panel = market
    return run_walk_forward(panel, ModelConfig(n_estimators=25, max_depth=3,
                                               train_days=300, test_days=150,
                                               embargo_days=5))


def _snapshot(market, when):
    _, bars, panel = market
    rows = panel.xs(when, level="date")
    prices = {s: float(rows.at[s, "close"]) for s in rows.index}
    return MarketSnapshot(as_of=when.date(), features=rows,
                          bars={s: b.loc[:when] for s, b in bars.items()},
                          prices=prices)


def test_rejects_a_badly_shaped_table():
    flat = pd.DataFrame({"proba": [0.6]}, index=[pd.Timestamp("2020-01-02")])
    with pytest.raises(ValueError):
        WalkForwardAlphaStrategy(flat)


def test_rejects_a_table_without_probabilities():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-02"), "AAA")],
                                    names=["date", "symbol"])
    with pytest.raises(ValueError):
        WalkForwardAlphaStrategy(pd.DataFrame({"other": [0.6]}, index=idx))


def test_silent_before_the_first_test_fold(market, walk_forward):
    """Dates inside the first training window have no honest prediction.

    Emitting anything there would mean inventing a signal the model could not
    have produced at the time.
    """
    cfg, _, panel = market
    strategy = WalkForwardAlphaStrategy(walk_forward.predictions, cfg.risk)

    covered = walk_forward.predictions.index.get_level_values("date")
    all_dates = sorted(panel.index.get_level_values("date").unique())
    before = [d for d in all_dates if d < covered.min()]
    assert before, "fixture should include dates before the first fold"

    for when in before[-20:]:
        assert strategy.generate(_snapshot(market, when)) == []


def test_only_emits_symbols_it_has_predictions_for(market, walk_forward):
    cfg, _, _ = market
    strategy = WalkForwardAlphaStrategy(walk_forward.predictions, cfg.risk)
    preds = walk_forward.predictions

    covered_dates = sorted(set(preds.index.get_level_values("date")))
    for when in covered_dates[:: max(1, len(covered_dates) // 10)]:
        expected = set(preds.xs(when, level="date").index)
        for sig in strategy.generate(_snapshot(market, when)):
            assert sig.symbol in expected


def test_confidence_matches_the_recorded_probability(market, walk_forward):
    cfg, _, _ = market
    strategy = WalkForwardAlphaStrategy(walk_forward.predictions, cfg.risk)
    preds = walk_forward.predictions

    when = sorted(set(preds.index.get_level_values("date")))[len(preds) // 200]
    on_day = preds.xs(when, level="date")["proba"]
    for sig in strategy.generate(_snapshot(market, when)):
        assert sig.confidence == pytest.approx(on_day[sig.symbol])
        assert sig.meta["oos"] is True


def test_honours_the_confidence_floor(market, walk_forward):
    cfg, _, _ = market
    risk = RiskConfig(min_signal_confidence=0.95)
    strategy = WalkForwardAlphaStrategy(walk_forward.predictions, risk)

    covered = sorted(set(walk_forward.predictions.index.get_level_values("date")))
    for when in covered[:: max(1, len(covered) // 15)]:
        for sig in strategy.generate(_snapshot(market, when)):
            assert sig.confidence >= 0.95


def test_predictions_never_precede_their_training_window(walk_forward):
    """Every prediction must fall inside the fold's test window, which starts
    after that fold's training data ends."""
    by_fold = walk_forward.predictions.groupby("fold")
    folds = {f.fold: f for f in walk_forward.folds}
    for fold_id, chunk in by_fold:
        dates = chunk.index.get_level_values("date")
        assert dates.min() >= folds[fold_id].test_start
        assert dates.min() > folds[fold_id].train_end
