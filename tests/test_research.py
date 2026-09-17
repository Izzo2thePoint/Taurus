"""Model training, walk-forward splits, and the tradeability gate."""
import numpy as np
import pytest

from taurus.config import Config, ModelConfig
from taurus.research.dataset import build_panel, feature_matrix, training_rows
from taurus.research.model import AlphaModel, ModelMetrics
from taurus.research.walkforward import run_walk_forward, train_production_model
from tests.synth import make_universe

SYMBOLS = ["SPY", "AAA", "BBB", "CCC"]


@pytest.fixture(scope="module")
def panel():
    cfg = Config()
    cfg.data.universe = SYMBOLS
    cfg.data.benchmark = "SPY"
    return build_panel(make_universe(SYMBOLS, n=1400, seed=31), cfg)


def test_panel_is_indexed_by_date_and_symbol(panel):
    assert list(panel.index.names) == ["date", "symbol"]
    assert {"target", "label", "atr_abs", "close"}.issubset(panel.columns)


def test_feature_matrix_excludes_labels(panel):
    X, cols = feature_matrix(panel)
    for leak in ("target", "label", "label_return", "close", "atr_abs"):
        assert leak not in cols, f"{leak} would leak the answer into the model"


def test_training_rows_drop_incomplete_data(panel):
    rows = training_rows(panel)
    X, cols = feature_matrix(rows)
    assert not X.isna().any().any()
    assert rows["target"].notna().all()


def test_model_trains_and_predicts_probabilities(panel):
    rows = training_rows(panel)
    X, _ = feature_matrix(rows)
    y = rows["target"].astype(int)
    split = int(len(X) * 0.7)

    model = AlphaModel(ModelConfig(n_estimators=40, max_depth=3))
    metrics = model.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

    proba = model.predict_proba(X.iloc[split:])
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.n_test == len(X) - split


def test_model_round_trips_through_disk(panel, tmp_path):
    rows = training_rows(panel).head(1500)
    X, _ = feature_matrix(rows)
    y = rows["target"].astype(int)

    model = AlphaModel(ModelConfig(n_estimators=30, max_depth=3))
    model.fit(X, y)
    model.save(str(tmp_path))

    reloaded = AlphaModel.load(str(tmp_path))
    assert reloaded.features == model.features
    np.testing.assert_allclose(reloaded.predict_proba(X), model.predict_proba(X))


def test_tradeability_gate_rejects_a_coin_flip():
    assert not ModelMetrics(0.51, 0.50, 0.25, 100, 50, 0.5).is_tradeable(0.52)
    assert not ModelMetrics(0.60, 0.48, 0.25, 100, 50, 0.5).is_tradeable(0.52)  # AUC below chance
    assert ModelMetrics(0.58, 0.61, 0.24, 100, 50, 0.5).is_tradeable(0.52)


def test_walk_forward_folds_never_overlap(panel):
    cfg = ModelConfig(n_estimators=25, max_depth=3, train_days=300,
                      test_days=120, embargo_days=5)
    result = run_walk_forward(panel, cfg)
    assert result.folds

    for fold in result.folds:
        assert fold.train_end < fold.test_start, "train window runs into the test window"
        gap = (fold.test_start - fold.train_end).days
        assert gap >= cfg.embargo_days, f"embargo of {gap}d is shorter than configured"

    for earlier, later in zip(result.folds, result.folds[1:]):
        assert earlier.test_end <= later.test_start, "test windows overlap"


def test_walk_forward_predictions_cover_the_test_windows(panel):
    cfg = ModelConfig(n_estimators=25, max_depth=3, train_days=300,
                      test_days=120, embargo_days=5)
    result = run_walk_forward(panel, cfg)
    preds = result.predictions
    assert not preds.empty
    assert ((preds["proba"] >= 0) & (preds["proba"] <= 1)).all()
    assert list(preds.index.names) == ["date", "symbol"]


def test_walk_forward_refuses_insufficient_history(panel):
    tiny = panel.loc[panel.index.get_level_values("date") < "2019-06-01"]
    with pytest.raises(ValueError):
        run_walk_forward(tiny, ModelConfig(train_days=756, test_days=63))


def test_production_model_holds_out_a_tail(panel):
    cfg = ModelConfig(n_estimators=25, max_depth=3, test_days=100, embargo_days=5)
    model, metrics = train_production_model(panel, cfg)
    assert model.model is not None
    assert metrics.n_test > 0, "production model reported no holdout"


def test_gate_rejects_a_majority_class_predictor():
    """68% accuracy against a 68% base rate is no edge at all.

    This is the failure mode the gate exists for: a model that never predicts
    a winner scores well on accuracy while being worthless.
    """
    majority = ModelMetrics(accuracy=0.68, auc=0.52, brier=0.21,
                            n_train=1000, n_test=500, positive_rate=0.32)
    assert majority.baseline_accuracy == pytest.approx(0.68)
    assert not majority.is_tradeable(0.52)

    real_edge = ModelMetrics(accuracy=0.73, auc=0.58, brier=0.19,
                             n_train=1000, n_test=500, positive_rate=0.32)
    assert real_edge.accuracy_edge == pytest.approx(0.05)
    assert real_edge.is_tradeable(0.52)


def test_gate_rejects_sub_chance_auc():
    metrics = ModelMetrics(accuracy=0.90, auc=0.45, brier=0.10,
                           n_train=1000, n_test=500, positive_rate=0.50)
    assert not metrics.is_tradeable(0.52)


def test_production_model_records_its_gate_verdict(panel):
    """The verdict must be persisted, or a consumer loading the model from
    disk has no way to know it was rejected."""
    cfg = ModelConfig(n_estimators=25, max_depth=3, test_days=100, embargo_days=5)
    _, metrics = train_production_model(panel, cfg)
    assert metrics.tradeable == metrics.is_tradeable(cfg.min_oos_accuracy)


def test_gate_verdict_survives_a_disk_round_trip(panel, tmp_path):
    cfg = ModelConfig(n_estimators=25, max_depth=3, test_days=100, embargo_days=5)
    model, metrics = train_production_model(panel, cfg)
    model.metrics = metrics
    model.save(str(tmp_path))
    assert AlphaModel.load(str(tmp_path)).metrics.tradeable == metrics.tradeable
