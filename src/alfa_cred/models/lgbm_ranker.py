"""Обёртка над LightGBM LambdaRank."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


DEFAULT_PARAMS: dict[str, object] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "lambdarank_truncation_level": 5,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}


@dataclass
class LgbmRanker:
    """Лёгкая обёртка над `lightgbm.LGBMRanker` с groups-API.

    Хранит модель, имена фичей и категориальные колонки, чтобы при
    инференсе подавать тот же набор и тип.
    """

    params: dict[str, object] = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    n_estimators: int = 2000
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    early_stopping_rounds: int = 100
    log_evaluation_period: int = 100
    booster: lgb.Booster | None = None
    best_iteration: int | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Sequence[int],
        groups_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: Sequence[int] | None = None,
        groups_val: np.ndarray | None = None,
    ) -> "LgbmRanker":
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            group=groups_train,
            feature_name=self.feature_columns,
            categorical_feature=self.categorical_columns or "auto",
            free_raw_data=False,
        )
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=self.log_evaluation_period)]
        if X_val is not None and y_val is not None and groups_val is not None:
            val_set = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val,
                feature_name=self.feature_columns,
                categorical_feature=self.categorical_columns or "auto",
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets.append(val_set)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        booster = lgb.train(
            params={**self.params, "num_iterations": self.n_estimators},
            train_set=train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.booster = booster
        self.best_iteration = booster.best_iteration or self.n_estimators
        LOG.info("LGBM обучен: best_iter=%s, num_features=%d", self.best_iteration, len(self.feature_columns))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")
        return self.booster.predict(
            X[self.feature_columns],
            num_iteration=self.best_iteration,
        )

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self.booster is None:
            raise RuntimeError("Модель не обучена.")
        importance = self.booster.feature_importance(importance_type=importance_type)
        return pd.Series(importance, index=self.feature_columns).sort_values(ascending=False)
