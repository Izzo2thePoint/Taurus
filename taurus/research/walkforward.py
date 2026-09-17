"""Purged walk-forward validation.

The only honest way to evaluate a trading model: train on the past, predict
the next block, roll forward, never look back. Two details make it correct
and are easy to get wrong:

  * Purging — a bar's triple-barrier label resolves up to `max_holding_days`
    later, so bars near the end of a training window contain information from
    the test window. They are dropped.
  * Embargo — the first bars after the split are skipped as well, since they
    overlap the same resolution horizon in the other direction.

Without these, out-of-sample scores are inflated and the strategy looks
profitable right up until it trades real money.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ModelConfig
from .dataset import feature_matrix, training_rows
from .model import AlphaModel, ModelMetrics

log = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: ModelMetrics


@dataclass
class WalkForwardResult:
    """Aggregate of every fold, plus the out-of-sample prediction series.

    `predictions` is the honest signal history: each value was produced by a
    model that had never seen that bar. The backtester consumes exactly this.
    """
    folds: list[WalkForwardFold]
    predictions: pd.DataFrame  # index (date, symbol), columns [proba, target]

    @property
    def mean_accuracy(self) -> float:
        return float(np.nanmean([f.metrics.accuracy for f in self.folds])) if self.folds else float("nan")

    @property
    def mean_auc(self) -> float:
        return float(np.nanmean([f.metrics.auc for f in self.folds])) if self.folds else float("nan")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"fold": f.fold, "train_end": f.train_end.date(),
             "test_start": f.test_start.date(), "test_end": f.test_end.date(),
             "n_train": f.metrics.n_train, "n_test": f.metrics.n_test,
             "accuracy": round(f.metrics.accuracy, 4),
             "auc": round(f.metrics.auc, 4),
             "brier": round(f.metrics.brier, 4)}
            for f in self.folds
        ])


def run_walk_forward(panel: pd.DataFrame, model_config: ModelConfig,
                     embargo_days: int | None = None) -> WalkForwardResult:
    """Roll a train/test window through the panel and collect OOS predictions."""
    cfg = model_config
    embargo = embargo_days if embargo_days is not None else cfg.embargo_days

    rows = training_rows(panel)
    if rows.empty:
        raise ValueError("no complete rows in panel; check warm-up and label horizon")

    X_all, feature_cols = feature_matrix(rows)
    y_all = rows["target"].astype(int)
    dates = rows.index.get_level_values("date")
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))

    folds: list[WalkForwardFold] = []
    pred_chunks: list[pd.DataFrame] = []
    start = 0
    fold_id = 0

    while start + cfg.train_days + embargo + cfg.test_days <= len(unique_dates):
        train_lo = unique_dates[start]
        train_hi = unique_dates[start + cfg.train_days - 1]
        test_lo = unique_dates[start + cfg.train_days + embargo]
        test_hi_idx = min(start + cfg.train_days + embargo + cfg.test_days - 1,
                          len(unique_dates) - 1)
        test_hi = unique_dates[test_hi_idx]

        # Purge: drop training bars whose labels resolve after the train
        # window ends and therefore peek into the test period.
        purge_cutoff = train_hi - pd.Timedelta(days=int(cfg.embargo_days * 1.5))
        train_mask = (dates >= train_lo) & (dates <= purge_cutoff)
        test_mask = (dates >= test_lo) & (dates <= test_hi)

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_te, y_te = X_all[test_mask], y_all[test_mask]

        if len(X_tr) < 200 or len(X_te) == 0 or y_tr.nunique() < 2:
            log.warning("fold %d skipped: train=%d test=%d", fold_id, len(X_tr), len(X_te))
            start += cfg.test_days
            fold_id += 1
            continue

        model = AlphaModel(cfg)
        metrics = model.fit(X_tr, y_tr, X_te, y_te)
        folds.append(WalkForwardFold(
            fold=fold_id, train_start=train_lo, train_end=train_hi,
            test_start=test_lo, test_end=test_hi, metrics=metrics,
        ))

        chunk = pd.DataFrame(
            {"proba": model.predict_proba(X_te), "target": y_te.to_numpy()},
            index=X_te.index,
        )
        chunk["fold"] = fold_id
        pred_chunks.append(chunk)

        log.info("fold %d [%s -> %s] acc=%.3f auc=%.3f",
                 fold_id, test_lo.date(), test_hi.date(),
                 metrics.accuracy, metrics.auc)

        start += cfg.test_days
        fold_id += 1

    if not folds:
        raise ValueError(
            f"no usable folds: need >= {cfg.train_days + embargo + cfg.test_days} "
            f"trading days, panel has {len(unique_dates)}"
        )

    predictions = pd.concat(pred_chunks).sort_index()
    return WalkForwardResult(folds=folds, predictions=predictions)


def train_production_model(panel: pd.DataFrame, model_config: ModelConfig
                           ) -> tuple[AlphaModel, ModelMetrics]:
    """Fit the model that will actually trade.

    Trained on all history bar a final holdout, which is what the returned
    metrics describe. Walk-forward tells you whether the *approach* works;
    this is the single fitted artifact the live agent loads.
    """
    rows = training_rows(panel)
    X_all, _ = feature_matrix(rows)
    y_all = rows["target"].astype(int)
    dates = rows.index.get_level_values("date")
    unique_dates = pd.DatetimeIndex(sorted(dates.unique()))

    if len(unique_dates) <= model_config.test_days + model_config.embargo_days:
        model = AlphaModel(model_config)
        metrics = model.fit(X_all, y_all)
        # No holdout means no evidence; an unmeasured model is not tradeable.
        metrics.tradeable = False
        return model, metrics

    holdout_start = unique_dates[-model_config.test_days]
    purge_cutoff = holdout_start - pd.Timedelta(days=int(model_config.embargo_days * 1.5))
    train_mask = dates <= purge_cutoff
    test_mask = dates >= holdout_start

    model = AlphaModel(model_config)
    metrics = model.fit(X_all[train_mask], y_all[train_mask],
                        X_all[test_mask], y_all[test_mask])
    metrics.tradeable = metrics.is_tradeable(model_config.min_oos_accuracy)
    return model, metrics
