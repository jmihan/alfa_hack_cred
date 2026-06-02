"""Универсальный CV-loop с логированием в MLflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mlflow
import numpy as np
import pandas as pd

from alfa_cred.config import OOF_DIR, REQUEST_ID, TARGET
from alfa_cred.io_utils import make_groups, sort_by_request
from alfa_cred.metrics import mean_ndcg_at_5
from alfa_cred.models.lgbm_ranker import LgbmRanker
from alfa_cred.utils import get_logger, timer

LOG = get_logger(__name__)


@dataclass
class CvResult:
    fold_scores: list[float]
    fold_scores_subtask_a: list[float]
    fold_scores_subtask_b: list[float]
    oof_predictions: pd.DataFrame
    models: list[LgbmRanker]
    feature_importance: pd.Series

    @property
    def mean_ndcg(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std_ndcg(self) -> float:
        return float(np.std(self.fold_scores))

    @property
    def mean_ndcg_subtask_a(self) -> float:
        return float(np.mean(self.fold_scores_subtask_a))

    @property
    def mean_ndcg_subtask_b(self) -> float:
        return float(np.mean(self.fold_scores_subtask_b))


def _score_subtasks(
    df_fold: pd.DataFrame,
    score_col: str = "score",
) -> tuple[float, float, float]:
    """Считает NDCG@5 на фолде: общий, A (есть pil1mtrx=1), B (нет)."""
    request_has_pil = df_fold.groupby(REQUEST_ID)["pil1mtrx_offer"].transform("max")
    overall = mean_ndcg_at_5(df_fold, score_col=score_col)
    a_mask = request_has_pil == 1
    b_mask = ~a_mask
    a_score = (
        mean_ndcg_at_5(df_fold[a_mask], score_col=score_col) if a_mask.any() else float("nan")
    )
    b_score = (
        mean_ndcg_at_5(df_fold[b_mask], score_col=score_col) if b_mask.any() else float("nan")
    )
    return overall, a_score, b_score


def train_cv(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    params: dict[str, object],
    n_estimators: int = 2000,
    early_stopping_rounds: int = 100,
    run_name: str = "lgbm_baseline",
    apply_pil1mtrx_rule: bool = True,
    log_to_mlflow: bool = True,
) -> CvResult:
    """5-fold GroupKFold-обучение LightGBM LambdaRank.

    Логирует pre-fold и среднее значение NDCG@5 в MLflow, считает
    метрики отдельно для подзадачи A (запросы с `pil1mtrx_offer=1`) и
    подзадачи B (остальные).
    """
    df_sorted = sort_by_request(df_train)
    oof = pd.DataFrame(
        {
            REQUEST_ID: df_sorted[REQUEST_ID].values,
            TARGET: df_sorted[TARGET].values,
            "pil1mtrx_offer": df_sorted["pil1mtrx_offer"].values,
            "score": np.zeros(len(df_sorted), dtype=np.float64),
            "fold": np.full(len(df_sorted), -1, dtype=np.int8),
        }
    )

    fold_scores: list[float] = []
    fold_scores_a: list[float] = []
    fold_scores_b: list[float] = []
    models: list[LgbmRanker] = []
    importance_acc = pd.Series(0.0, index=feature_cols)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        with timer(f"fold {fold_idx}"):
            df_tr = df_sorted.iloc[train_idx]
            df_va = df_sorted.iloc[val_idx]
            df_tr = sort_by_request(df_tr)
            df_va = sort_by_request(df_va)

            model = LgbmRanker(
                params=params,
                n_estimators=n_estimators,
                feature_columns=feature_cols,
                categorical_columns=categorical_cols,
                early_stopping_rounds=early_stopping_rounds,
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

            df_va_scored = df_va.assign(score=preds)
            if apply_pil1mtrx_rule:
                df_va_scored = _apply_pil1mtrx_rule(df_va_scored)

            overall, a_score, b_score = _score_subtasks(df_va_scored)
            fold_scores.append(overall)
            fold_scores_a.append(a_score)
            fold_scores_b.append(b_score)

            oof.loc[df_va_scored.index, "score"] = df_va_scored["score"].values
            oof.loc[df_va_scored.index, "fold"] = fold_idx

            importance_acc += model.feature_importance(importance_type="gain")
            models.append(model)

            LOG.info(
                "Fold %d: NDCG@5=%.4f (A=%.4f, B=%.4f), best_iter=%d",
                fold_idx, overall, a_score, b_score, model.best_iteration,
            )
            if log_to_mlflow:
                mlflow.log_metric("fold_ndcg5", overall, step=fold_idx)
                mlflow.log_metric("fold_ndcg5_a", a_score, step=fold_idx)
                mlflow.log_metric("fold_ndcg5_b", b_score, step=fold_idx)
                mlflow.log_metric("fold_best_iter", model.best_iteration, step=fold_idx)

    importance_avg = (importance_acc / max(len(models), 1)).sort_values(ascending=False)

    result = CvResult(
        fold_scores=fold_scores,
        fold_scores_subtask_a=fold_scores_a,
        fold_scores_subtask_b=fold_scores_b,
        oof_predictions=oof,
        models=models,
        feature_importance=importance_avg,
    )

    LOG.info(
        "CV NDCG@5 = %.4f ± %.4f (A=%.4f, B=%.4f)",
        result.mean_ndcg, result.std_ndcg,
        result.mean_ndcg_subtask_a, result.mean_ndcg_subtask_b,
    )

    if log_to_mlflow:
        mlflow.log_metric("cv_ndcg5_mean", result.mean_ndcg)
        mlflow.log_metric("cv_ndcg5_std", result.std_ndcg)
        mlflow.log_metric("cv_ndcg5_subtask_a", result.mean_ndcg_subtask_a)
        mlflow.log_metric("cv_ndcg5_subtask_b", result.mean_ndcg_subtask_b)
        _log_artifacts(result, run_name=run_name)

    return result


def _apply_pil1mtrx_rule(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Post-processing: офферы с `pil1mtrx_offer=1` получают +LARGE_CONST к скору.

    Гарантирует, что после сортировки в каждом запросе оффер с pil1mtrx=1
    окажется первым (если он есть). Если pil1mtrx=1 на нескольких офферах
    одного запроса — между ними сохранится порядок по модельному скору.
    """
    LARGE_CONST = 1e6
    df = df.copy()
    df[score_col] = df[score_col] + LARGE_CONST * df["pil1mtrx_offer"].astype(float)
    return df


def _log_artifacts(result: CvResult, run_name: str) -> None:
    """Сохраняет OOF-предсказания и важность фичей как артефакты MLflow."""
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = OOF_DIR / f"{run_name}.parquet"
    result.oof_predictions.to_parquet(oof_path, index=False)
    mlflow.log_artifact(str(oof_path), artifact_path="oof")

    importance_path = OOF_DIR / f"{run_name}_importance.csv"
    result.feature_importance.to_csv(importance_path, header=["gain"])
    mlflow.log_artifact(str(importance_path), artifact_path="importance")


def predict_full(
    models: list[LgbmRanker],
    df_test: pd.DataFrame,
    feature_cols: list[str],
    apply_pil1mtrx_rule: bool = True,
) -> np.ndarray:
    """Усредняет предсказания CV-моделей по test и применяет hard-rule."""
    preds = np.zeros(len(df_test), dtype=np.float64)
    for model in models:
        preds += model.predict(df_test[feature_cols])
    preds /= max(len(models), 1)
    if apply_pil1mtrx_rule:
        preds = preds + 1e6 * df_test["pil1mtrx_offer"].astype(float).to_numpy()
    return preds
