"""Утилиты для смешивания скоров нескольких моделей.

Для нашей задачи лучший blend получился rank-averaging: внутри каждого
запроса считаем перцентильные ранги (`rank(pct=True)`), усредняем по
моделям, а итог уже идёт в сабмит. Такой подход устойчив к разным
масштабам скоров (LGBM, XGBoost, CatBoost дают несравнимые величины).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alfa_cred.config import REQUEST_ID, VARIANT_ID
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


def _read_ranks(path: Path, base_keys: pd.DataFrame) -> np.ndarray:
    """Читает test_scores, выравнивает по base_keys, возвращает перцентильные ранги."""
    df = pd.read_parquet(path)
    df[REQUEST_ID] = df[REQUEST_ID].astype(str)
    df[VARIANT_ID] = df[VARIANT_ID].astype("int32")
    merged = base_keys.merge(
        df[[REQUEST_ID, VARIANT_ID, "score_raw"]],
        on=[REQUEST_ID, VARIANT_ID],
        how="left",
    )
    if merged["score_raw"].isna().any():
        n_missing = int(merged["score_raw"].isna().sum())
        LOG.warning("Пропуски при merge с %s: %d строк", path.name, n_missing)
    return (
        merged.groupby(REQUEST_ID, sort=False)["score_raw"]
        .rank(pct=True)
        .fillna(0.5)
        .to_numpy()
    )


def rank_avg_blend(
    test_score_paths: Iterable[Path],
    base: pd.DataFrame,
    weights: Iterable[float] | None = None,
    divide_after_sum: bool = True,
) -> np.ndarray:
    """Считает blend перцентильных рангов нескольких моделей.

    Для каждого файла из `test_score_paths` читается parquet с колонкой
    `score_raw`, скоры внутри каждого `request_id` переводятся в
    перцентильные ранги, затем рангы усредняются по моделям. Пропуски
    после merge заполняются нейтральным значением 0.5.

    Параметры
    ----------
    test_score_paths : Iterable[Path]
        Список путей к файлам вида `*_test_scores.parquet`, в каждом
        ожидаются колонки `REQUEST_ID`, `VARIANT_ID`, `score_raw`.
    base : pd.DataFrame
        Базовый порядок строк (request_id × variant_no). Длина результата
        совпадает с `len(base)`.
    weights : Iterable[float] | None
        Веса моделей. По умолчанию — равные (uniform). Веса нормализуются
        к сумме 1.
    divide_after_sum : bool
        Численный режим. При `True` (по умолчанию для uniform) делает
        sum(ranks) и в конце делит на число моделей. Это устойчиво
        совпадает с эталоном record_blend. При `False` накапливает
        sum(w * ranks) с нормализованными весами — режим для blend
        с явно заданными весами. Для uniform два режима математически
        эквивалентны, но различаются на 1 ULP из-за float-арифметики.

    Возвращает
    ----------
    np.ndarray
        Массив длины `len(base)` с усреднёнными перцентильными рангами.
    """
    paths = [Path(p) for p in test_score_paths]
    if not paths:
        raise ValueError("test_score_paths пуст")

    base_keys = base[[REQUEST_ID, VARIANT_ID]].copy()
    base_keys[REQUEST_ID] = base_keys[REQUEST_ID].astype(str)
    base_keys[VARIANT_ID] = base_keys[VARIANT_ID].astype("int32")

    if weights is None and divide_after_sum:
        rank_sum = np.zeros(len(base_keys), dtype=np.float64)
        for path in paths:
            rank_sum += _read_ranks(path, base_keys)
        return rank_sum / len(paths)

    if weights is None:
        weights_arr = np.full(len(paths), 1.0 / len(paths))
    else:
        weights_arr = np.asarray(list(weights), dtype=np.float64)
        if len(weights_arr) != len(paths):
            raise ValueError(
                f"Длина weights ({len(weights_arr)}) не совпадает с числом моделей ({len(paths)})"
            )
        weights_arr = weights_arr / weights_arr.sum()

    rank_sum = np.zeros(len(base_keys), dtype=np.float64)
    for path, w in zip(paths, weights_arr):
        rank_sum += w * _read_ranks(path, base_keys)
    return rank_sum
