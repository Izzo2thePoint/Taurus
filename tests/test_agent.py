"""Agent loop, journal, strategies, and config gates."""
from datetime import date

import pandas as pd
import pytest

from taurus.agent.journal import Journal
from taurus.agent.loop import TradingAgent
from taurus.config import Config
from taurus.data.providers import CSVProvider, _normalize
from taurus.execution.paper import PaperBroker
from taurus.research.dataset import build_panel
from taurus.strategy.base import MarketSnapshot, Signal, Strategy
from taurus.strategy.ensemble import EnsembleStrategy
from taurus.strategy.momentum import MomentumBreakoutStrategy
from tests.synth import make_universe

SYMBOLS = ["SPY", "AAA", "BBB"]


@pytest.fixture(scope="module")
def market():
    cfg = Config()
    cfg.data.universe = SYMBOLS
    cfg.data.benchmark = "SPY"
    bars = make_universe(SYMBOLS, n=600, seed=41)
    return cfg, bars, build_panel(bars, cfg)


# --- journal ---------------------------------------------------------------

def test_journal_round_trips(tmp_path):
    j = Journal(str(tmp_path / "j.jsonl"))
    j.log_equity(date(2026, 1, 5), 100_000, 50_000, 2, 1.2, 0.0)
    j.log_decision(date(2026, 1, 5), "AAA", "buy", 10, 100.0, 0.7, "because")
    assert len(j.recent("decision")) == 1
    assert j.performance_summary()["current_equity"] == 100_000


def test_journal_survives_a_torn_line(tmp_path):
    path = tmp_path / "j.jsonl"
    j = Journal(str(path))
    j.log_equity(date(2026, 1, 5), 100_000, 0, 0, 0.0, 0.0)
    with open(path, "a") as fh:
        fh.write('{"truncated')            # killed mid-write
    assert len(list(j.read())) == 1


def test_journal_reports_drawdown(tmp_path):
    j = Journal(str(tmp_path / "j.jsonl"))
    for day, equity in enumerate([100_000, 120_000, 90_000], start=1):
        j.log_equity(date(2026, 1, day), equity, 0, 0, 0.0, 0.0)
    assert j.performance_summary()["current_drawdown"] == pytest.approx(0.25)


# --- strategies ------------------------------------------------------------

def _snapshot(market, offset=-1):
    cfg, bars, panel = market
    latest = sorted(panel.index.get_level_values("date").unique())[offset]
    rows = panel.xs(latest, level="date")
    prices = {s: float(rows.at[s, "close"]) for s in rows.index}
    return MarketSnapshot(as_of=latest.date(), features=rows,
                          bars={s: b.loc[:latest] for s, b in bars.items()},
                          prices=prices)


def test_momentum_returns_well_formed_signals(market):
    cfg, _, _ = market
    for sig in MomentumBreakoutStrategy(cfg.risk).generate(_snapshot(market)):
        assert sig.direction in (-1, 0, 1)
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.rationale


class Fixed(Strategy):
    def __init__(self, direction, confidence, symbols=("AAA",)):
        self.direction, self.confidence, self.symbols = direction, confidence, symbols

    def generate(self, snapshot):
        return [Signal(symbol=s, direction=self.direction,
                       confidence=self.confidence, volatility=0.3,
                       price=100.0, atr=2.0, rationale="fixed")
                for s in self.symbols]


def test_ensemble_drops_contradictions(market):
    ensemble = EnsembleStrategy([Fixed(1, 0.8), Fixed(-1, 0.8)])
    assert ensemble.generate(_snapshot(market)) == []


def test_ensemble_rewards_agreement(market):
    alone = EnsembleStrategy([Fixed(1, 0.70)]).generate(_snapshot(market))
    agreed = EnsembleStrategy([Fixed(1, 0.70), Fixed(1, 0.70)]).generate(_snapshot(market))
    assert agreed[0].confidence > alone[0].confidence


def test_ensemble_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        EnsembleStrategy([Fixed(1, 0.8)], weights=[0.5, 0.5])


def test_ensemble_rejects_empty():
    with pytest.raises(ValueError):
        EnsembleStrategy([])


# --- agent -----------------------------------------------------------------

def test_agent_cycle_places_orders_and_journals(market, tmp_path):
    cfg, bars, panel = market
    cfg = Config.load(None)
    cfg.data.universe = SYMBOLS
    cfg.agent.journal_path = str(tmp_path / "j.jsonl")

    broker = PaperBroker(100_000)
    agent = TradingAgent(cfg, broker, strategy=Fixed(1, 0.85, SYMBOLS))
    agent._bars, agent._panel = bars, panel

    latest = sorted(panel.index.get_level_values("date").unique())[-1].date()
    result = agent.run_cycle(latest)

    assert result.orders_submitted > 0
    assert result.fills > 0
    assert broker.positions()
    kinds = {r["kind"] for r in Journal(cfg.agent.journal_path).read()}
    assert {"equity", "decision"}.issubset(kinds)


def test_agent_halts_and_flattens(market, tmp_path):
    cfg, bars, panel = market
    cfg = Config.load(None)
    cfg.data.universe = SYMBOLS
    cfg.agent.journal_path = str(tmp_path / "j.jsonl")

    broker = PaperBroker(100_000)
    agent = TradingAgent(cfg, broker, strategy=Fixed(1, 0.85, SYMBOLS))
    agent._bars, agent._panel = bars, panel
    latest = sorted(panel.index.get_level_values("date").unique())[-1].date()
    agent.run_cycle(latest)
    assert broker.positions()

    agent.risk.force_halt("test kill switch")
    result = agent.run_cycle(latest)
    assert result.halted
    assert broker.positions() == {}


def test_agent_requires_observation_first():
    agent = TradingAgent(Config(), PaperBroker(100_000), strategy=Fixed(1, 0.8))
    with pytest.raises(RuntimeError):
        agent.run_cycle(date(2026, 1, 5))


def test_agent_snapshot_never_exceeds_as_of(market):
    cfg, bars, panel = market
    agent = TradingAgent(cfg, PaperBroker(100_000), strategy=Fixed(1, 0.8))
    agent._bars, agent._panel = bars, panel

    dates = sorted(panel.index.get_level_values("date").unique())
    as_of = dates[len(dates) // 2].date()
    snapshot = agent._latest_snapshot(as_of)
    assert snapshot.as_of <= as_of
    for frame in snapshot.bars.values():
        assert frame.index.max() <= pd.Timestamp(as_of)


# --- config + data ---------------------------------------------------------

def test_live_needs_flag_and_env(monkeypatch):
    cfg = Config()
    cfg.execution.allow_live = True
    monkeypatch.delenv("TAURUS_ALLOW_LIVE", raising=False)
    assert cfg.live_enabled() is False
    monkeypatch.setenv("TAURUS_ALLOW_LIVE", "1")
    assert cfg.live_enabled() is True
    cfg.execution.allow_live = False
    assert cfg.live_enabled() is False


def test_config_yaml_overrides_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  aggression: 4.0\n  max_positions: 3\n"
                    "data:\n  universe: [AAA, BBB]\n")
    cfg = Config.load(str(path))
    assert cfg.risk.aggression == 4.0
    assert cfg.risk.max_positions == 3
    assert cfg.data.universe == ["AAA", "BBB"]
    assert cfg.risk.max_drawdown_halt == 0.20     # untouched default survives


def test_missing_config_falls_back_to_defaults():
    assert Config.load("/nonexistent/path.yaml").risk.aggression == 2.0


def test_normalize_applies_adjusted_close():
    raw = pd.DataFrame({
        "Open": [100.0], "High": [110.0], "Low": [90.0],
        "Close": [100.0], "Adj Close": [50.0], "Volume": [1e6],
    }, index=pd.to_datetime(["2024-01-02"]))
    out = _normalize(raw, "AAA")
    assert out["close"].iloc[0] == 50.0
    assert out["high"].iloc[0] == pytest.approx(55.0)   # scaled by the same ratio


def test_csv_provider_reads_bars(tmp_path):
    bars = make_universe(["AAA"], n=100, seed=1)["AAA"]
    bars.to_csv(tmp_path / "AAA.csv")
    out = CSVProvider(str(tmp_path)).history("AAA", "2019-01-02")
    assert len(out) > 50
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_csv_provider_raises_for_unknown_symbol(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVProvider(str(tmp_path)).history("NOPE", "2020-01-01")


def test_rejected_model_is_not_traded(tmp_path, monkeypatch):
    """A gate checked at training time and ignored at trading time is no gate.

    A model that failed must not be silently picked up by `backtest`,
    `signals`, or `trade`.
    """
    from taurus.cli import _load_gated_model
    from taurus.research.model import AlphaModel, ModelMetrics

    cfg = Config()
    cfg.model.model_dir = str(tmp_path)

    model = AlphaModel(cfg.model)
    model.features = ["mom_21"]
    model.metrics = ModelMetrics(accuracy=0.68, auc=0.49, brier=0.22,
                                 n_train=100, n_test=50, positive_rate=0.32,
                                 tradeable=False)
    model.save(str(tmp_path))

    assert _load_gated_model(cfg) is None
    assert _load_gated_model(cfg, force=True) is not None   # research escape hatch


def test_accepted_model_is_loaded(tmp_path):
    from taurus.cli import _load_gated_model
    from taurus.research.model import AlphaModel, ModelMetrics

    cfg = Config()
    cfg.model.model_dir = str(tmp_path)
    model = AlphaModel(cfg.model)
    model.features = ["mom_21"]
    model.metrics = ModelMetrics(accuracy=0.74, auc=0.58, brier=0.19,
                                 n_train=100, n_test=50, positive_rate=0.32,
                                 tradeable=True)
    model.save(str(tmp_path))
    assert _load_gated_model(cfg) is not None


def test_missing_model_is_not_an_error(tmp_path):
    from taurus.cli import _load_gated_model
    cfg = Config()
    cfg.model.model_dir = str(tmp_path / "nothing-here")
    assert _load_gated_model(cfg) is None
