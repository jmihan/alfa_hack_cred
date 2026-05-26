"""Метрика NDCG@k для оценки качества ранжирования предложений."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from alfa_cred.config import REQUEST_ID, TARGET


def ndcg_at_k(relevances: Sequence[int], k: int = 5) -> float:
    """NDCG@k для бинарной релевантности (0/1).

    Параметры
    ----------
    relevances : Sequence[int]
        Список меток релевантности предложений в порядке, выданном моделью
        (от наиболее до наименее релевантного).
    k : int, по умолчанию 5
        Глубина усечения.

    Возвращает
    ----------
    float
        Значение NDCG@k или NaN, если идеальный DCG равен нулю
        (в группе нет позитивных меток).
    """
    top_k = list(relevances[:k])
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(top_k) if rel == 1)

    ideal_top_k = sorted(relevances, reverse=True)[:k]
    idcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(ideal_top_k) if rel == 1)

    return dcg / idcg if idcg > 0 else np.nan


def mean_ndcg_at_5(
    df: pd.DataFrame,
    request_col: str = REQUEST_ID,
    score_col: str = "score",
    target_col: str = TARGET,
) -> float:
    """Средний NDCG@5 по всем запросам в датафрейме.

    Параметры
    ----------
    df : pd.DataFrame
        Должен содержать колонки `request_col`, `score_col`, `target_col`.
    request_col : str
        Имя колонки с идентификатором запроса.
    score_col : str
        Имя колонки со скорами модели (сортировка по убыванию).
    target_col : str
        Имя колонки с бинарным таргетом (0/1).

    Возвращает
    ----------
    float
        Среднее значение NDCG@5 по всем запросам (NaN-группы игнорируются).
    """
    df_sorted = df.sort_values(
        [request_col, score_col],
        ascending=[True, False],
    )
    ndcg_per_request = (
        df_sorted.groupby(request_col, sort=False)[target_col]
        .apply(lambda x: ndcg_at_k(x.tolist(), k=5))
    )
    return float(np.nanmean(ndcg_per_request))


def fast_mean_ndcg_at_5(
    request_ids: np.ndarray,
    scores: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Векторизированная версия NDCG@5 на чистом NumPy.

    Использовать при многократных вычислениях в цикле CV.
    """
    df = pd.DataFrame(
        {
            REQUEST_ID: request_ids,
            "score": scores,
            TARGET: targets,
        }
    )
    return mean_ndcg_at_5(df)


def compute_b_ndcg5(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = TARGET,
    request_col: str = REQUEST_ID,
    pil_col: str = "pil1mtrx_offer",
) -> float:
    """NDCG@5 на подзадаче B (запросы без `pil1mtrx_offer=1`).

    На подзадаче A метрика всегда близка к 1.0 за счёт hard-rule, поэтому
    она «скрывает» реальный прирост модели. Этот хелпер фильтрует df на
    «трудные» запросы и считает NDCG@5 только на них — это даёт честную
    оценку B-only модели.
    """
    # Локальный импорт, чтобы не создавать цикл metrics <-> subtasks.
    from alfa_cred.subtasks import filter_subtask_b

    df_b = filter_subtask_b(df, pil_col=pil_col)
    if score_col != "score":
        df_b = df_b.rename(columns={score_col: "score"})
    return mean_ndcg_at_5(
        df_b,
        request_col=request_col,
        score_col="score",
        target_col=target_col,
    )
