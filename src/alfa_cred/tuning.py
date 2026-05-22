"""Подбор гиперпараметров LightGBM LambdaRank через Optuna.

Целевая метрика для оптимизации — NDCG@5 на подзадаче B (запросы без
`pil1mtrx_offer=1`). Подзадача A фактически закрыта hard-rule'ом, и
оптимизация общей метрики смазывала бы сигнал.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from alfa_cred.config import RANDOM_STATE, REQUEST_ID, TARGET
from alfa_cred.io_utils import make_groups, sort_by_request
from alfa_cred.metrics import mean_ndcg_at_5
from alfa_cred.models.lgbm_ranker import LgbmRanker
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


@dataclass
class TuneResult:
    best_params: dict[str, object]
    best_value: float
    study: optuna.Study


def _suggest_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "lambdarank_truncation_level": 5,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "verbose": -1,
        "n_jobs": -1,
        "seed": RANDOM_STATE,
    }


def _cv_score_subtask_b(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict[str, object],
    n_estimators: int,
    early_stopping_rounds: int,
    n_splits: int = 3,
) -> float:
    """Кросс-валидация по request_id, метрика только на подзадаче B.

    Подзадача A покрывается hard-rule'ом и в оптимизации не участвует.
    """
    df_sorted = sort_by_request(df)
    groups = df_sorted[REQUEST_ID].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    scores: list[float] = []

    for train_idx, val_idx in splitter.split(np.zeros(len(df_sorted)), groups=groups):
        df_tr = sort_by_request(df_sorted.iloc[train_idx])
        df_va = sort_by_request(df_sorted.iloc[val_idx])

        # Оптимизируем именно подзадачу B: на валидации оставляем только
        # запросы без pil1mtrx_offer=1, на обучении даём всё (модель должна
        # учиться на полном распределении, но оцениваем на сложной части).
        req_has_pil = df_va.groupby(REQUEST_ID)["pil1mtrx_offer"].transform("max")
        b_mask = req_has_pil == 0
        if not b_mask.any():
            continue

        model = LgbmRanker(
            params=params,
            n_estimators=n_estimators,
            feature_columns=feature_cols,
            categorical_columns=categorical_cols,
            early_stopping_rounds=early_stopping_rounds,
            log_evaluation_period=10_000,  # практически без логов
        )
        model.fit(
            X_train=df_tr[feature_cols],
            y_train=df_tr[TARGET].values,
            groups_train=make_groups(df_tr),
            X_val=df_va[feature_cols],
            y_val=df_va[TARGET].values,
            groups_val=make_groups(df_va),
        )
        preds = model.predict(df_va[feature_cols])
        df_va_b = df_va[b_mask].assign(score=preds[b_mask.to_numpy()])
        if df_va_b.empty:
            continue
        score = mean_ndcg_at_5(df_va_b, score_col="score")
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def run_optuna(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    n_trials: int = 30,
    n_splits: int = 3,
    n_estimators: int = 1500,
    early_stopping_rounds: int = 75,
    seed: int = RANDOM_STATE,
    callbacks: list[Callable] | None = None,
) -> TuneResult:
    """Запускает TPE-оптимизацию с pruning по NDCG@5 на подзадаче B."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        score = _cv_score_subtask_b(
            df=df,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            params=params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            n_splits=n_splits,
        )
        LOG.info("Trial %d: B-NDCG@5 = %.5f", trial.number, score)
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)
    LOG.info("Лучший результат: %.5f", study.best_value)
    LOG.info("Лучшие параметры: %s", study.best_params)
    return TuneResult(best_params=study.best_params, best_value=study.best_value, study=study)
