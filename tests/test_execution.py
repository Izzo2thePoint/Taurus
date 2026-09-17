"""The paper broker must cost money, refuse what it cannot fund, and
never invent a fill."""
import pytest

from taurus.config import Config
from taurus.execution.alpaca import LiveTradingBlocked, make_broker
from taurus.execution.broker import Order, OrderSide, OrderType
from taurus.execution.paper import PaperBroker


@pytest.fixture
def broker():
    b = PaperBroker(100_000, commission_per_share=0.005, slippage_bps=10)
    b.set_prices({"AAA": 100.0, "BBB": 50.0})
    return b


def test_slippage_always_works_against_us(broker):
    buy = broker.submit(Order("AAA", 10, OrderSide.BUY))
    assert buy.price > 100.0
    sell = broker.submit(Order("BBB", 10, OrderSide.SELL))
    assert sell.price < 50.0


def test_round_trip_realizes_pnl(broker):
    broker.submit(Order("AAA", 100, OrderSide.BUY))
    broker.set_prices({"AAA": 120.0})
    broker.submit(Order("AAA", 100, OrderSide.SELL))
    assert broker.realized_pnl > 0
    assert "AAA" not in broker.positions()


def test_scaling_in_averages_the_basis(broker):
    broker.submit(Order("AAA", 100, OrderSide.BUY))
    broker.set_prices({"AAA": 200.0})
    broker.submit(Order("AAA", 100, OrderSide.BUY))
    pos = broker.positions()["AAA"]
    assert pos.quantity == 200
    assert 140 < pos.avg_price < 160


def test_leverage_ceiling_is_enforced(broker):
    assert broker.submit(Order("AAA", 100_000, OrderSide.BUY)) is None
    assert broker.rejected


def test_no_fill_without_a_mark(broker):
    assert broker.submit(Order("UNKNOWN", 10, OrderSide.BUY)) is None


def test_limit_order_does_not_fill_through_the_limit(broker):
    order = Order("AAA", 10, OrderSide.BUY, OrderType.LIMIT, limit_price=90.0)
    assert broker.submit(order) is None


def test_close_all_flattens_everything(broker):
    broker.submit(Order("AAA", 50, OrderSide.BUY))
    broker.submit(Order("BBB", 50, OrderSide.BUY))
    broker.close_all({"AAA": 100.0, "BBB": 50.0})
    assert broker.positions() == {}


def test_equity_tracks_marks(broker):
    broker.submit(Order("AAA", 100, OrderSide.BUY))
    before = broker.mark_to_market({"AAA": 100.0})
    after = broker.mark_to_market({"AAA": 110.0})
    assert after - before == pytest.approx(1000.0, abs=1.0)


def test_costs_accumulate(broker):
    broker.submit(Order("AAA", 100, OrderSide.BUY))
    assert broker.total_commission > 0
    assert broker.total_slippage > 0


def test_default_broker_is_the_simulator():
    assert isinstance(make_broker(Config()), PaperBroker)


def test_live_requires_both_switches(monkeypatch):
    from taurus.execution.alpaca import AlpacaBroker
    cfg = Config()
    cfg.execution.broker = "alpaca"
    cfg.execution.allow_live = True          # one switch only
    monkeypatch.delenv("TAURUS_ALLOW_LIVE", raising=False)
    assert cfg.live_enabled() is False
    with pytest.raises(LiveTradingBlocked):
        AlpacaBroker(cfg, paper=False)


def test_unknown_broker_falls_back_to_paper():
    cfg = Config()
    cfg.execution.broker = "definitely-not-a-broker"
    assert isinstance(make_broker(cfg), PaperBroker)
