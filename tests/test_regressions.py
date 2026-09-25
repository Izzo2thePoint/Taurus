"""Regressions for defects found in code review.

Each test here corresponds to a bug that shipped because nothing checked for
it. Several were dead guardrails: code that existed, read correctly, and did
nothing.
"""
import datetime as dt
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from taurus.backtest.engine import (FORWARD_LOOKING_COLUMNS, BacktestEngine,
                                    TradeRecord, _drop_forward_looking)
from taurus.config import (Config, FeatureConfig, LabelConfig, ModelConfig,
                           RiskConfig)
from taurus.data.providers import CachedProvider
from taurus.execution.broker import Position
from taurus.features.builder import FeatureBuilder
from taurus.research.dataset import build_panel
from taurus.research.walkforward import run_walk_forward
from taurus.risk.allocator import target_weights
from taurus.risk.limits import RiskManager
from taurus.strategy.base import MarketSnapshot, Signal
from taurus.strategy.momentum import MomentumBreakoutStrategy
from taurus.strategy.walkforward_alpha import WalkForwardAlphaStrategy
from tests.synth import make_bars, make_universe


# --- 1. The daily-loss kill switch was dead from day two onward ------------

def test_daily_loss_halt_fires_on_a_later_session():
    """Rebaselining to the equity being checked zeroed the day's loss before
    it could be measured, disabling the limit entirely after day one."""
    rm = RiskManager(RiskConfig(daily_loss_halt=0.06, max_drawdown_halt=0.99))
    rm.reset(100_000)
    rm.check(100_000, dt.date(2026, 1, 5))
    assert rm.check(80_000, dt.date(2026, 1, 6)).halt


def test_daily_baseline_is_the_previous_close():
    rm = RiskManager(RiskConfig(daily_loss_halt=0.99, max_drawdown_halt=0.99))
    rm.reset(100_000)
    rm.check(100_000, dt.date(2026, 1, 5))
    rm.check(90_000, dt.date(2026, 1, 5))       # day one closes at 90k
    rm.check(89_000, dt.date(2026, 1, 6))
    assert rm.state.day_start_equity == pytest.approx(90_000)
    assert rm.daily_loss(89_000) == pytest.approx(1 / 90, abs=1e-4)


def test_intraday_losses_accumulate_within_one_session():
    rm = RiskManager(RiskConfig(daily_loss_halt=0.05, max_drawdown_halt=0.99))
    rm.reset(100_000)
    day = dt.date(2026, 1, 5)
    assert not rm.check(98_000, day).halt
    assert rm.check(94_000, day).halt


# --- 2. A broker blip must not look like a 100% drawdown ------------------

def test_broker_error_raises_rather_than_reporting_zero_equity():
    """Returning 0.0 made the risk manager see a 100% drawdown, latch a
    permanent halt, and market-sell the entire live book."""
    from taurus.execution.alpaca import AlpacaBroker, BrokerUnavailable

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._prices = {}
    broker._client = Mock()
    broker._client.get_account.side_effect = ConnectionError("timeout")

    with pytest.raises(BrokerUnavailable):
        broker.mark_to_market({"AAA": 100.0})


def test_agent_skips_the_cycle_when_the_broker_is_unavailable(tmp_path):
    from taurus.agent.loop import TradingAgent
    from taurus.execution.alpaca import BrokerUnavailable
    from taurus.execution.paper import PaperBroker

    cfg = Config()
    cfg.data.universe = ["SPY", "AAA"]
    cfg.agent.journal_path = str(tmp_path / "j.jsonl")

    bars = make_universe(["SPY", "AAA"], n=400, seed=7)
    broker = PaperBroker(100_000)
    agent = TradingAgent(cfg, broker, strategy=MomentumBreakoutStrategy(cfg.risk))
    agent._bars, agent._panel = bars, build_panel(bars, cfg)

    with patch.object(PaperBroker, "mark_to_market",
                      side_effect=BrokerUnavailable("down")):
        result = agent.run_cycle(dt.date(2026, 1, 5))

    assert result.orders_submitted == 0
    assert not result.halted           # unknown equity is not a drawdown


# --- 3. The cache served a narrow range for a wider request ---------------

def test_cache_refetches_when_it_does_not_reach_back_far_enough(tmp_path):
    inner = Mock()
    full = make_bars(n=600, seed=3, start="2019-01-02")
    inner.history.side_effect = lambda sym, start, end=None, interval="1d": \
        full.loc[str(start):]

    cached = CachedProvider(inner, str(tmp_path))
    recent = cached.history("AAA", "2020-06-01")
    early = cached.history("AAA", "2019-01-02")

    assert early.index.min() < recent.index.min(), \
        "a wider request was served the narrower cached range"
    assert inner.history.call_count == 2


def test_cache_widens_rather_than_replacing(tmp_path):
    inner = Mock()
    full = make_bars(n=600, seed=3, start="2019-01-02")
    inner.history.side_effect = lambda sym, start, end=None, interval="1d": \
        full.loc[str(start):]

    cached = CachedProvider(inner, str(tmp_path))
    cached.history("AAA", "2020-06-01")
    cached.history("AAA", "2019-01-02")
    calls_before = inner.history.call_count
    cached.history("AAA", "2020-06-01")     # narrow again; must hit the cache
    assert inner.history.call_count == calls_before


# --- 4. The configured initial stop did not exist -------------------------

def test_initial_stop_is_actually_applied():
    """`per_trade_stop_atr` had no call site: only a trailing stop ran, and it
    used a flat 2% of entry price instead of the symbol's ATR."""
    cfg = Config()
    cfg.data.universe = ["SPY", "AAA"]
    cfg.risk.per_trade_stop_atr = 0.5      # very tight: must trigger
    cfg.risk.trailing_stop_atr = 50.0      # effectively disabled

    bars = make_universe(["SPY", "AAA"], n=500, seed=13)
    panel = build_panel(bars, cfg)

    class AlwaysLong:
        name = "always_long"
        def generate(self, snapshot):
            return [Signal(symbol=s, direction=1, confidence=0.9,
                           volatility=0.3, price=p, atr=p * 0.02)
                    for s, p in snapshot.prices.items()]
        def on_fill(self, *a):
            pass

    result = BacktestEngine(cfg, AlwaysLong()).run(panel, bars)
    assert any(t.exit_reason == "initial_stop" for t in result.trades)


def test_stop_distance_scales_with_volatility():
    from taurus.risk.sizing import stop_price
    risk = RiskConfig(per_trade_stop_atr=2.0)
    calm = 100.0 - stop_price(100.0, 1.0, risk)
    wild = 100.0 - stop_price(100.0, 5.0, risk)
    assert wild > calm


# --- 5. Hysteresis was unreachable ---------------------------------------

def _snapshot_for(symbol="AAA", price=100.0):
    feats = pd.DataFrame({"atr_pct": [0.02], "vol_21": [0.30],
                          "mom_21": [0.05], "sma_ratio_20_50": [0.02],
                          "breakout_20": [1.0], "volume_ratio": [1.2],
                          "rsi": [0.55]}, index=[symbol])
    return MarketSnapshot(as_of=dt.date(2024, 1, 2), features=feats,
                          bars={}, prices={symbol: price})


def test_strategies_emit_below_the_entry_threshold_for_incumbents():
    """A strategy that filters at min_signal_confidence makes the allocator's
    exit buffer dead code: the marginal incumbent never reaches it."""
    risk = RiskConfig(min_signal_confidence=0.55, exit_confidence_buffer=0.05)
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-02"), "AAA")],
                                    names=["date", "symbol"])
    strategy = WalkForwardAlphaStrategy(
        pd.DataFrame({"proba": [0.52]}, index=idx), risk)
    assert strategy.generate(_snapshot_for()) != []


def test_hysteresis_keeps_the_incumbent_end_to_end():
    cfg = Config()
    cfg.risk.min_signal_confidence = 0.55
    cfg.risk.exit_confidence_buffer = 0.05
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-02"), "AAA")],
                                    names=["date", "symbol"])
    strategy = WalkForwardAlphaStrategy(
        pd.DataFrame({"proba": [0.52]}, index=idx), cfg.risk)
    signals = strategy.generate(_snapshot_for())

    assert target_weights(signals, cfg, incumbents=set()) == {}
    assert "AAA" in target_weights(signals, cfg, incumbents={"AAA"})


# --- 6. min_holding_days was enforced in backtest but not live ------------

def test_live_agent_honours_the_minimum_hold(tmp_path):
    from taurus.agent.loop import TradingAgent
    from taurus.execution.paper import PaperBroker

    cfg = Config()
    cfg.data.universe = ["SPY", "AAA"]
    cfg.risk.min_holding_days = 5
    cfg.agent.journal_path = str(tmp_path / "j.jsonl")

    bars = make_universe(["SPY", "AAA"], n=400, seed=9)
    broker = PaperBroker(100_000)
    agent = TradingAgent(cfg, broker, strategy=MomentumBreakoutStrategy(cfg.risk))
    agent._bars, agent._panel = bars, build_panel(bars, cfg)

    opened = dt.datetime(2026, 1, 5)
    broker._positions["AAA"] = Position("AAA", 100, 100.0, peak_price=100.0,
                                        opened_at=opened)
    orders = agent._orders_from_signals([], {"AAA": 100.0}, 100_000, 1.0,
                                        as_of=dt.date(2026, 1, 6))
    assert orders == [], "position exited inside its minimum hold"

    later = agent._orders_from_signals([], {"AAA": 100.0}, 100_000, 1.0,
                                       as_of=dt.date(2026, 1, 20))
    assert later and later[0].reason == "signal_exit"


# --- 7 & 9. Features must follow their own config ------------------------

def test_feature_builder_respects_custom_volatility_windows():
    """Hardcoded vol_10 / vol_63 raised KeyError the moment vol_windows moved."""
    cfg = FeatureConfig(vol_windows=(5, 30))
    frame = FeatureBuilder(cfg).build(make_bars(n=300, seed=11))
    assert "vol_5" in frame.columns and "vol_30" in frame.columns
    assert "vol_ratio_5_30" in frame.columns


def test_momentum_spread_is_not_silently_constant():
    """mom_21_ex_5 was computed from other feature columns and collapsed to a
    constant 0 whenever momentum_windows lacked 21 or 5."""
    cfg = FeatureConfig(momentum_windows=(3, 7))
    frame = FeatureBuilder(cfg).build(make_bars(n=300, seed=12))
    assert frame["mom_21_ex_5"].dropna().std() > 0


def test_reserved_columns_exclude_the_label_and_price():
    frame = pd.DataFrame(columns=["mom_5", "target", "close", "label",
                                  "atr_abs", "holding_days"])
    cols = FeatureBuilder.feature_columns(frame)
    assert cols == ["mom_5"]


# --- 8. The purge must follow the label horizon --------------------------

def test_purge_widens_with_the_label_horizon():
    """Deriving the purge from embargo_days leaks resolved labels whenever the
    label horizon is longer."""
    cfg = Config()
    cfg.data.universe = ["SPY", "AAA", "BBB"]
    cfg.labels = LabelConfig(max_holding_days=30)
    panel = build_panel(make_universe(["SPY", "AAA", "BBB"], n=1200, seed=17), cfg)
    model_cfg = ModelConfig(n_estimators=20, max_depth=3, train_days=300,
                            test_days=150, embargo_days=2)

    narrow = run_walk_forward(panel, model_cfg, label_horizon=2)
    wide = run_walk_forward(panel, model_cfg, label_horizon=30)
    assert wide.folds[0].metrics.n_train < narrow.folds[0].metrics.n_train


# --- 10. The snapshot carried the answers --------------------------------

def test_snapshot_drops_forward_looking_columns():
    rows = pd.DataFrame({"mom_5": [0.1], "target": [1], "label": [1.0],
                         "label_return": [0.05], "holding_days": [3.0],
                         "close": [100.0]}, index=["AAA"])
    out = _drop_forward_looking(rows)
    for column in FORWARD_LOOKING_COLUMNS:
        assert column not in out.columns
    assert "mom_5" in out.columns and "close" in out.columns


def test_engine_never_hands_a_strategy_the_label():
    cfg = Config()
    cfg.data.universe = ["SPY", "AAA"]
    bars = make_universe(["SPY", "AAA"], n=400, seed=19)
    seen: list[set] = []

    class Peeker:
        name = "peeker"
        def generate(self, snapshot):
            seen.append(set(snapshot.features.columns))
            return []
        def on_fill(self, *a):
            pass

    BacktestEngine(cfg, Peeker()).run(build_panel(bars, cfg), bars)
    assert seen
    for columns in seen:
        assert not columns & set(FORWARD_LOOKING_COLUMNS)


# --- 11. Unpriced positions vanished from the trade log ------------------

def test_unpriced_close_keeps_the_trade_record():
    """Dropping the record removed losing trades from win rate and profit
    factor rather than reporting them."""
    trades: list[TradeRecord] = []
    open_trades = {"AAA": TradeRecord(
        symbol="AAA", entry_date=pd.Timestamp("2024-01-02"), exit_date=None,
        entry_price=100.0, exit_price=None, quantity=50, return_pct=None,
        realized_pnl=-500.0, peak_quantity=50)}

    BacktestEngine._close_trade(trades, open_trades, "AAA",
                                pd.Timestamp("2024-02-01"), 0.0, "risk_halt")
    assert len(trades) == 1
    assert trades[0].return_pct is not None
    assert trades[0].exit_reason.endswith("_unpriced")
    assert not open_trades
