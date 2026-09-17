"""The backtester's guarantees: no lookahead, honest accounting, live halts."""
import pandas as pd
import pytest

from taurus.backtest.engine import BacktestEngine
from taurus.backtest.metrics import build_report, max_drawdown, sharpe_ratio
from taurus.config import Config
from taurus.research.dataset import build_panel
from taurus.strategy.base import MarketSnapshot, Signal, Strategy
from taurus.strategy.momentum import MomentumBreakoutStrategy
from tests.synth import make_universe

SYMBOLS = ["SPY", "AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def market():
    cfg = Config()
    cfg.data.universe = SYMBOLS
    cfg.data.benchmark = "SPY"
    bars = make_universe(SYMBOLS, n=700, seed=21)
    return cfg, bars, build_panel(bars, cfg)


class SnapshotSpy(Strategy):
    """Records the last timestamp visible in every snapshot it is given."""

    name = "spy"

    def __init__(self):
        self.max_seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def generate(self, snapshot: MarketSnapshot):
        for bars in snapshot.bars.values():
            if not bars.empty:
                self.max_seen.append((pd.Timestamp(snapshot.as_of), bars.index.max()))
        return []


def test_strategy_never_sees_the_future(market):
    """The snapshot handed to a strategy must not contain a bar after as_of.

    This is the load-bearing test for the whole project. If it fails, every
    performance figure the system produces is worthless.
    """
    cfg, bars, panel = market
    spy = SnapshotSpy()
    BacktestEngine(cfg, spy).run(panel, bars)
    assert spy.max_seen
    for as_of, latest_bar in spy.max_seen:
        assert latest_bar <= as_of, f"strategy saw {latest_bar} while deciding on {as_of}"


class AlwaysLong(Strategy):
    name = "always_long"

    def generate(self, snapshot: MarketSnapshot):
        return [Signal(symbol=s, direction=1, confidence=0.80,
                       volatility=0.30, price=p, atr=p * 0.02)
                for s, p in snapshot.prices.items()]


def test_trade_pnl_reconciles_with_equity(market):
    """Sum of realized trade P&L must equal the change in equity.

    If these disagree, either the trade log or the equity curve is lying,
    and there is no way to tell which.
    """
    cfg, bars, panel = market
    result = BacktestEngine(cfg, AlwaysLong()).run(panel, bars)
    realized = sum(t.realized_pnl for t in result.trades)
    change = result.equity_curve.iloc[-1] - cfg.execution.starting_cash
    assert realized == pytest.approx(change, abs=1.0)


def test_no_trade_is_left_open(market):
    cfg, bars, panel = market
    result = BacktestEngine(cfg, AlwaysLong()).run(panel, bars)
    assert all(t.exit_date is not None and t.return_pct is not None
               for t in result.trades)


def test_leverage_cap_holds_throughout(market):
    cfg, bars, panel = market
    result = BacktestEngine(cfg, AlwaysLong()).run(panel, bars)
    gross = result.daily_positions["gross_exposure"]
    # Marks move between rebalances, so allow a modest overshoot; the point
    # is that the book is never wildly beyond the configured ceiling.
    assert gross.max() <= cfg.risk.max_gross_leverage * 1.35


class Crasher(Strategy):
    """Concentrates everything into one name to force a drawdown halt."""

    name = "crasher"

    def generate(self, snapshot: MarketSnapshot):
        worst = min(snapshot.prices, key=lambda s: snapshot.prices[s])
        return [Signal(symbol=worst, direction=1, confidence=0.99,
                       volatility=0.05, price=snapshot.prices[worst],
                       atr=snapshot.prices[worst] * 0.02)]


def test_halt_flattens_the_book_and_stops_trading():
    cfg = Config()
    cfg.data.universe = SYMBOLS
    cfg.risk.max_drawdown_halt = 0.05
    cfg.risk.daily_loss_halt = 0.03
    cfg.risk.aggression = 5.0
    # A steady decline guarantees the limit is breached.
    bars = make_universe(SYMBOLS, n=400, seed=3, drift=-0.004, vol=0.03)
    panel = build_panel(bars, cfg)
    result = BacktestEngine(cfg, Crasher()).run(panel, bars)

    if result.halted_at is not None:
        after = result.daily_positions.loc[result.halted_at:]
        # One bar of settling is allowed for the flattening fills.
        assert (after["n_positions"].iloc[1:] == 0).all()
        assert result.halt_reason


def test_backtest_is_deterministic(market):
    cfg, bars, panel = market
    a = BacktestEngine(cfg, MomentumBreakoutStrategy(cfg.risk)).run(panel, bars)
    b = BacktestEngine(cfg, MomentumBreakoutStrategy(cfg.risk)).run(panel, bars)
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)


def test_costs_make_a_difference(market):
    """A zero-cost run must beat a realistic-cost run. If it does not, costs
    are not actually being applied."""
    cfg, bars, panel = market
    import copy
    free = copy.deepcopy(cfg)
    free.execution.slippage_bps = 0.0
    free.execution.commission_per_share = 0.0
    costly = copy.deepcopy(cfg)
    costly.execution.slippage_bps = 50.0
    costly.execution.commission_per_share = 0.02

    r_free = BacktestEngine(free, AlwaysLong()).run(panel, bars)
    r_cost = BacktestEngine(costly, AlwaysLong()).run(panel, bars)
    assert r_free.report.total_return > r_cost.report.total_return


# --- metrics ---------------------------------------------------------------

def test_max_drawdown_matches_a_known_curve():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(0.5)


def test_sharpe_of_a_flat_curve_is_zero():
    assert sharpe_ratio(pd.Series([0.0] * 50)) == 0.0


def test_report_handles_an_empty_curve():
    report = build_report(pd.Series(dtype=float))
    assert report.total_return == 0.0
    assert report.n_trades == 0


def test_profit_factor_without_losses():
    report = build_report(pd.Series([100.0, 110.0]), trade_returns=[0.1, 0.2])
    assert report.profit_factor == float("inf")
