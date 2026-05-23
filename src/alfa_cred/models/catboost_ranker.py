"""Обёртка над CatBoost для задач ранжирования."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool

from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


DEFAULT_PARAMS: dict[str, object] = {
    "loss_function": "YetiRank",
    "eval_metric": "NDCG:top=5;type=Exp",
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "task_type": "GPU",
    "devices": "0",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "verbose": False,
    "allow_writing_files": False,
}


@dataclass
class CatBoostRanker:
    """Тонкая обёртка над `catboost.CatBoost` с group_id-API.

    CatBoost требует, чтобы данные были отсортированы по `group_id`. В
    отличие от LightGBM, он не использует длины групп — нужен сам
    идентификатор группы для каждой строки.
    """

    params: dict[str, object] = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    n_estimators: int = 2000
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    early_stopping_rounds: int = 100
    log_period: int = 100
    model: CatBoost | None = None
    best_iteration: int | None = None

    def _pool(
        self,
        X: pd.DataFrame,
        y: Sequence[int] | None,
        group_ids: np.ndarray,
    ) -> Pool:
        cat_idx = [self.feature_columns.index(c) for c in self.categorical_columns if c in self.feature_columns]
        data = X[self.feature_columns].copy()
        # CatBoost ожидает целочисленные коды или строки для категориальных
        for col in self.categorical_columns:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(-1).astype("int32")
        return Pool(
            data=data,
            label=y,
            group_id=group_ids,
            cat_features=cat_idx,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Sequence[int],
        group_ids_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: Sequence[int] | None = None,
        group_ids_val: np.ndarray | None = None,
    ) -> "CatBoostRanker":
        train_pool = self._pool(X_train, y_train, group_ids_train)
        eval_pool = None
        if X_val is not None and y_val is not None and group_ids_val is not None:
            eval_pool = self._pool(X_val, y_val, group_ids_val)

        model = CatBoost({
            **self.params,
            "iterations": self.n_estimators,
            "od_type": "Iter" if self.early_stopping_rounds else None,
            "od_wait": self.early_stopping_rounds or None,
        })
        model.fit(
            train_pool,
            eval_set=eval_pool,
            verbose=self.log_period if self.params.get("verbose") is False else False,
            use_best_model=eval_pool is not None,
        )
        self.model = model
        self.best_iteration = model.get_best_iteration() or self.n_estimators
        LOG.info(
            "CatBoost обучен: best_iter=%s, num_features=%d",
            self.best_iteration, len(self.feature_columns),
        )
        return self

    def predict(self, X: pd.DataFrame, group_ids: np.ndarray | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите fit().")
        data = X[self.feature_columns].copy()
        for col in self.categorical_columns:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(-1).astype("int32")
        cat_idx = [self.feature_columns.index(c) for c in self.categorical_columns if c in self.feature_columns]
        # Для inference group_id не обязателен, но Pool в CatBoost его принимает
        if group_ids is not None:
            pool = Pool(data=data, group_id=group_ids, cat_features=cat_idx)
        else:
            pool = Pool(data=data, cat_features=cat_idx)
        return self.model.predict(pool)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Модель не обучена.")
        importance = self.model.get_feature_importance()
        return pd.Series(importance, index=self.feature_columns).sort_values(ascending=False)
