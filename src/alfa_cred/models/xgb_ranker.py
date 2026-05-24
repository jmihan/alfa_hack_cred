"""Обёртка над XGBoost rank:ndcg для diversification ансамбля."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


DEFAULT_PARAMS: dict[str, object] = {
    "objective": "rank:ndcg",
    "eval_metric": ["ndcg@5"],
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": 42,
    "lambdarank_pair_method": "topk",
    "lambdarank_num_pair_per_sample": 8,
}


@dataclass
class XgbRanker:
    """Тонкая обёртка над `xgb.XGBRanker` с group-API."""

    params: dict[str, object] = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    n_estimators: int = 2000
    feature_columns: list[str] = field(default_factory=list)
    early_stopping_rounds: int = 100
    booster: xgb.Booster | None = None
    best_iteration: int | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Sequence[int],
        groups_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: Sequence[int] | None = None,
        groups_val: np.ndarray | None = None,
    ) -> "XgbRanker":
        dtrain = xgb.DMatrix(
            X_train[self.feature_columns],
            label=y_train,
            feature_names=self.feature_columns,
        )
        dtrain.set_group(groups_train)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None and groups_val is not None:
            dval = xgb.DMatrix(
                X_val[self.feature_columns],
                label=y_val,
                feature_names=self.feature_columns,
            )
            dval.set_group(groups_val)
            evals.append((dval, "valid"))

        booster = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=0,
        )
        self.booster = booster
        self.best_iteration = booster.best_iteration or self.n_estimators
        LOG.info("XGB обучен: best_iter=%s, num_features=%d", self.best_iteration, len(self.feature_columns))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Модель не обучена.")
        dmat = xgb.DMatrix(X[self.feature_columns], feature_names=self.feature_columns)
        return self.booster.predict(dmat, iteration_range=(0, self.best_iteration + 1))

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self.booster is None:
            raise RuntimeError("Модель не обучена.")
        raw = self.booster.get_score(importance_type=importance_type)
        importance = pd.Series(0.0, index=self.feature_columns)
        for k, v in raw.items():
            if k in importance.index:
                importance[k] = v
        return importance.sort_values(ascending=False)
