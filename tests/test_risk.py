"""Risk limits are the part that must never quietly stop working."""
import pytest

from taurus.config import Config, RiskConfig
from taurus.execution.broker import Position
from taurus.risk.allocator import orders_to_reach, target_weights
from taurus.risk.limits import RiskManager
from taurus.risk.sizing import (apply_portfolio_caps, kelly_fraction,
                                position_weight, shares_for_weight,
                                volatility_scalar)
from taurus.strategy.base import Signal


def test_kelly_is_zero_without_edge():
    assert kelly_fraction(0.3, 2.0) == 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert kelly_fraction(0.6, 2.0) > 0


def test_low_confidence_gets_no_capital():
    cfg = RiskConfig(min_signal_confidence=0.55)
    assert position_weight(0.54, 0.30, cfg) == 0.0
    assert position_weight(0.70, 0.30, cfg) > 0


def test_position_weight_respects_the_cap():
    cfg = RiskConfig(aggression=50.0, max_position_weight=0.25)
    # Absurd aggression must still not breach the per-name cap.
    assert position_weight(0.99, 0.10, cfg) <= 0.25


def test_volatility_scalar_shrinks_for_volatile_names():
    calm = volatility_scalar(0.10, 0.35)
    wild = volatility_scalar(0.90, 0.35)
    assert calm > wild > 0


def test_volatility_scalar_handles_zero_vol():
    assert volatility_scalar(0.0, 0.35) == 0.0


def test_portfolio_caps_enforce_leverage_and_count():
    cfg = RiskConfig(max_positions=5, max_gross_leverage=1.5,
                     max_position_weight=0.5)
    capped = apply_portfolio_caps({f"S{i}": 0.4 for i in range(20)}, cfg)
    assert len(capped) == 5
    assert sum(abs(v) for v in capped.values()) == pytest.approx(1.5)


def test_incumbent_bonus_affects_ranking_not_weight():
    cfg = RiskConfig(max_positions=1, max_gross_leverage=10.0)
    weights = {"NEW": 0.20, "HELD": 0.19}
    assert "NEW" in apply_portfolio_caps(weights, cfg)
    kept = apply_portfolio_caps(weights, cfg, incumbents={"HELD"})
    assert "HELD" in kept
    assert kept["HELD"] == pytest.approx(0.19)  # bonus did not inflate the size


def test_drawdown_halt_fires_and_latches():
    rm = RiskManager(RiskConfig(max_drawdown_halt=0.20, daily_loss_halt=0.99))
    rm.reset(100_000)
    assert rm.check(95_000).allowed
    assert rm.check(79_000).halt
    # A recovery must not silently un-halt: a human decides that.
    assert rm.check(100_000).halt


def test_daily_loss_halt_clears_next_session():
    import datetime as dt
    rm = RiskManager(RiskConfig(daily_loss_halt=0.05, max_drawdown_halt=0.99))
    rm.reset(100_000)
    day1 = dt.date(2026, 1, 5)
    assert rm.check(93_000, day1).halt
    assert rm.check(93_000, dt.date(2026, 1, 6)).allowed


def test_sizing_scales_down_as_drawdown_deepens():
    rm = RiskManager(RiskConfig(max_drawdown_halt=0.20, daily_loss_halt=0.99))
    rm.reset(100_000)
    rm.check(100_000)
    mild = rm.check(88_000).scale
    severe = rm.check(83_000).scale
    assert 0.25 <= severe < mild <= 1.0


def test_shares_for_weight_rejects_bad_prices():
    assert shares_for_weight(0.25, 100_000, 0.0) == 0
    assert shares_for_weight(0.25, 100_000, 50.0) == 500


def _signal(symbol, conf, vol=0.30):
    return Signal(symbol=symbol, direction=1, confidence=conf,
                  volatility=vol, price=100.0, atr=2.0)


def test_hysteresis_keeps_a_marginal_incumbent():
    cfg = Config()
    cfg.risk.min_signal_confidence = 0.55
    cfg.risk.exit_confidence_buffer = 0.05
    marginal = [_signal("AAA", 0.52)]
    assert "AAA" not in target_weights(marginal, cfg, incumbents=set())
    assert "AAA" in target_weights(marginal, cfg, incumbents={"AAA"})


def test_allocator_suppresses_tiny_adjustments():
    current = {"AAA": Position("AAA", 100, 100.0)}
    orders = orders_to_reach({"AAA": 0.1005}, current, {"AAA": 100.0},
                             equity=100_000)
    assert orders == []


def test_allocator_exits_dropped_names():
    current = {"AAA": Position("AAA", 100, 100.0)}
    orders = orders_to_reach({}, current, {"AAA": 100.0}, equity=100_000)
    assert len(orders) == 1 and orders[0].reason == "signal_exit"


def test_allocator_respects_protected_positions():
    current = {"AAA": Position("AAA", 100, 100.0)}
    orders = orders_to_reach({}, current, {"AAA": 100.0}, equity=100_000,
                             skip_exit=["AAA"])
    assert orders == []
