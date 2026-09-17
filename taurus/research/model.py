"""The learning component: a gradient-boosted classifier over the feature panel.

The model estimates P(profit barrier is hit before the stop). That probability
feeds both the entry decision and the position size, so calibration matters
more than raw accuracy — a model that is 60% accurate but always says 0.99 is
useless for sizing.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from ..config import ModelConfig

log = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Out-of-sample scorecard for one trained model."""
    accuracy: float
    auc: float
    brier: float
    n_train: int
    n_test: int
    positive_rate: float
    trained_at: str = ""
    # Verdict recorded at training time, so a consumer that loads the model
    # from disk does not have to re-derive whether it may be traded.
    tradeable: bool = False

    @property
    def baseline_accuracy(self) -> float:
        """Accuracy of always predicting the majority class.

        With a 2:1 barrier ratio only ~1 bar in 3 is a winner, so a model that
        always says "no" scores ~0.67. Any accuracy figure is meaningless
        unless it is read against this number.
        """
        if not np.isfinite(self.positive_rate):
            return 0.5
        return max(self.positive_rate, 1.0 - self.positive_rate)

    @property
    def accuracy_edge(self) -> float:
        """How much the model beats the always-say-no baseline by."""
        return self.accuracy - self.baseline_accuracy

    def is_tradeable(self, min_accuracy: float, min_edge: float = 0.005) -> bool:
        """Whether this model has earned the right to size real positions.

        Three gates, all of which must pass:

          * AUC above 0.5 — below that the model ranks worse than chance, so
            its probabilities cannot be used for sizing.
          * Accuracy above the configured floor.
          * Accuracy above the majority-class baseline. This is the gate that
            matters most and the one most often missed: a model can look 68%
            accurate while being strictly worse than never trading at all.
        """
        if not np.isfinite(self.auc) or self.auc <= 0.5:
            return False
        if not np.isfinite(self.accuracy) or self.accuracy < min_accuracy:
            return False
        return self.accuracy_edge >= min_edge


class AlphaModel:
    """Wraps the classifier, its feature list, and its out-of-sample metrics."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model: CalibratedClassifierCV | None = None
        self.features: list[str] = []
        self.metrics: ModelMetrics | None = None

    def _make_estimator(self) -> HistGradientBoostingClassifier:
        cfg = self.config
        return HistGradientBoostingClassifier(
            max_iter=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            min_samples_leaf=cfg.min_samples_leaf,
            l2_regularization=1.0,
            random_state=cfg.random_state,
            early_stopping=False,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_test: pd.DataFrame | None = None, y_test: pd.Series | None = None
            ) -> ModelMetrics:
        """Train on (X, y); score on the held-out set when one is supplied."""
        if X.empty or y.nunique() < 2:
            raise ValueError("training set is empty or single-class")

        self.features = list(X.columns)
        # Isotonic calibration needs a decent sample; fall back to sigmoid on
        # small folds where isotonic would just overfit the calibration curve.
        method = "isotonic" if len(X) >= 2000 else "sigmoid"
        self.model = CalibratedClassifierCV(
            self._make_estimator(), method=method, cv=3,
        )
        self.model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=int))

        if X_test is not None and y_test is not None and len(X_test) > 0:
            proba = self.predict_proba(X_test)
            preds = (proba >= 0.5).astype(int)
            auc = (roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan"))
            metrics = ModelMetrics(
                accuracy=float(accuracy_score(y_test, preds)),
                auc=float(auc),
                brier=float(brier_score_loss(y_test, proba)),
                n_train=len(X), n_test=len(X_test),
                positive_rate=float(y_test.mean()),
            )
        else:
            metrics = ModelMetrics(
                accuracy=float("nan"), auc=float("nan"), brier=float("nan"),
                n_train=len(X), n_test=0, positive_rate=float(y.mean()),
            )

        metrics.trained_at = datetime.utcnow().isoformat(timespec="seconds")
        self.metrics = metrics
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(profit barrier first) for each row."""
        if self.model is None:
            raise RuntimeError("model is not trained")
        aligned = X[self.features] if self.features else X
        return self.model.predict_proba(aligned.to_numpy(dtype=float))[:, 1]

    def feature_importance(self, X: pd.DataFrame, y: pd.Series,
                           n_repeats: int = 5) -> pd.Series:
        """Permutation importance — what the model is actually leaning on.

        Worth checking after every retrain: if a single feature dominates,
        that is usually a leak rather than an edge.
        """
        from sklearn.inspection import permutation_importance

        if self.model is None:
            raise RuntimeError("model is not trained")
        result = permutation_importance(
            self.model, X[self.features].to_numpy(dtype=float),
            y.to_numpy(dtype=int), n_repeats=n_repeats,
            random_state=self.config.random_state, scoring="roc_auc",
        )
        return pd.Series(result.importances_mean, index=self.features).sort_values(ascending=False)

    # --- persistence -------------------------------------------------------

    def save(self, directory: str | None = None, name: str = "alpha") -> str:
        import joblib

        directory = directory or self.config.model_dir
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.joblib")
        joblib.dump({"model": self.model, "features": self.features,
                     "config": asdict(self.config)}, path)
        if self.metrics:
            with open(os.path.join(directory, f"{name}.metrics.json"), "w") as fh:
                json.dump(asdict(self.metrics), fh, indent=2)
        log.info("saved model to %s", path)
        return path

    @classmethod
    def load(cls, directory: str = "models", name: str = "alpha") -> "AlphaModel":
        import joblib

        path = os.path.join(directory, f"{name}.joblib")
        payload = joblib.load(path)
        obj = cls(ModelConfig(**payload["config"]))
        obj.model = payload["model"]
        obj.features = payload["features"]
        metrics_path = os.path.join(directory, f"{name}.metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as fh:
                obj.metrics = ModelMetrics(**json.load(fh))
        return obj
