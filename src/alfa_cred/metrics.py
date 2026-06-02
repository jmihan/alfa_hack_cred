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
        Метки релевантности офферов в порядке, выданном моделью
        (от наиболее к наименее релевантному).
    k : int, по умолчанию 5
        Глубина усечения.

    Возвращает
    ----------
    float
        Значение NDCG@k или NaN, если идеальный DCG равен нулю (в группе нет
        позитивных меток).
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

    Датафрейм должен содержать колонки `request_col`, `score_col`, `target_col`;
    офферы внутри запроса сортируются по `score_col` (убывание), NaN-группы
    (без позитива) игнорируются.
    """
    df_sorted = df.sort_values([request_col, score_col], ascending=[True, False])
    ndcg_per_request = (
        df_sorted.groupby(request_col, sort=False)[target_col]
        .apply(lambda x: ndcg_at_k(x.tolist(), k=5))
    )
    return float(np.nanmean(ndcg_per_request))
