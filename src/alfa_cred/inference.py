"""Формирование сабмит-файла в формате Яндекс Контеста.

Содержит как обычный однослойный сабмит (`make_submission`), так и
two-stage схему (`build_two_stage_submission`): разные модели на
подзадачах A (есть `pil1mtrx_offer=1`) и B (нет такого флага).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alfa_cred.blending import rank_avg_blend
from alfa_cred.config import (
    REQUEST_ID,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSION_COLUMNS,
    SUBMISSION_SCORE_DECIMALS,
    SUBMISSION_SEPARATOR,
    VARIANT_ID,
)
from alfa_cred.subtasks import is_subtask_b_mask
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)

PIL_HARD_RULE_BOOST = 1e6
PIL_RANK_BOOST = 1.0


def apply_pil1mtrx_hard_rule(
    df: pd.DataFrame,
    scores: np.ndarray,
    pil_col: str = "pil1mtrx_offer",
    boost: float = PIL_HARD_RULE_BOOST,
) -> np.ndarray:
    """Прибавляет большую константу к скору, если `pil1mtrx_offer = 1`.

    После такого «буста» оффер с pil1mtrx=1 гарантированно окажется на
    первом месте в своей группе при сортировке по убыванию score.
    Используется как страховка поверх модели: согласно EDA, при
    pil1mtrx=1 target rate = 99.73%.
    """
    if pil_col not in df.columns:
        return scores
    return scores + boost * df[pil_col].astype(float).to_numpy()


def make_submission(
    df_test: pd.DataFrame,
    scores: np.ndarray,
    out_path: Path | str,
    score_decimals: int = SUBMISSION_SCORE_DECIMALS,
) -> Path:
    """Собирает и сохраняет сабмит, сверяясь со схемой commit.csv."""
    if len(scores) != len(df_test):
        raise ValueError(
            f"Длина scores ({len(scores)}) не совпадает с df_test ({len(df_test)})"
        )

    submission = pd.DataFrame(
        {
            REQUEST_ID: df_test[REQUEST_ID].to_numpy(),
            VARIANT_ID: df_test[VARIANT_ID].to_numpy(),
            "score": np.round(scores, score_decimals),
        }
    )

    if submission["score"].isna().any() or np.isinf(submission["score"]).any():
        raise ValueError("В скорах обнаружены NaN или Inf — проверьте модель")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(
        out_path,
        sep=SUBMISSION_SEPARATOR,
        index=False,
        columns=list(SUBMISSION_COLUMNS),
    )
    LOG.info("Сабмит сохранён: %s (rows=%d)", out_path, len(submission))
    return out_path


def verify_submission(submission_path: Path, sample_path: Path) -> None:
    """Сверяет шейп и набор (request_id, variant_no) с эталонным `commit.csv`."""
    submitted = pd.read_csv(submission_path, sep=SUBMISSION_SEPARATOR)
    sample = pd.read_csv(sample_path, sep=SUBMISSION_SEPARATOR)
    if len(submitted) != len(sample):
        raise ValueError(
            f"Размер сабмита {len(submitted)} != {len(sample)} в эталоне"
        )
    sub_keys = set(map(tuple, submitted[[REQUEST_ID, VARIANT_ID]].astype(str).values))
    ref_keys = set(map(tuple, sample[[REQUEST_ID, VARIANT_ID]].astype(str).values))
    if sub_keys != ref_keys:
        diff = len(sub_keys ^ ref_keys)
        raise ValueError(f"Ключи (request_id, variant_no) различаются: {diff} элементов")
    LOG.info("Сабмит соответствует эталону по структуре")


def build_two_stage_submission(
    b_test_paths: Iterable[Path],
    record_test_paths: Iterable[Path],
    test_sorted: pd.DataFrame,
    out_path: Path | str,
    b_blend_weights: Iterable[float] | None = None,
    pil_col: str = "pil1mtrx_offer",
) -> Path:
    """Собирает two-stage сабмит и сохраняет CSV.

    Логика чистого replacement-а:
    - Для запросов подзадачи A (есть `pil1mtrx_offer=1` в группе) скор
      берётся из rank-averaging blend моделей `record_test_paths` и
      сверху добавляется hard-rule «+1.0 за pil1mtrx_offer=1».
    - Для запросов подзадачи B (нет pil1mtrx-флага) скор берётся из
      rank-averaging blend моделей `b_test_paths` с весами
      `b_blend_weights` (либо uniform, если None).

    Параметры
    ----------
    b_test_paths : Iterable[Path]
        Пути к `*_test_scores.parquet` B-only моделей.
    record_test_paths : Iterable[Path]
        Пути к `*_test_scores.parquet` основного blend для подзадачи A.
    test_sorted : pd.DataFrame
        Тестовая выборка, отсортированная по `request_id`. Должна
        содержать колонки `REQUEST_ID`, `VARIANT_ID`, `pil_col`.
    out_path : Path | str
        Куда сохранить итоговый CSV.
    b_blend_weights : Iterable[float] | None
        Веса для B-only blend. По умолчанию — uniform.
    pil_col : str
        Имя бизнес-флага, по умолчанию `pil1mtrx_offer`.

    Возвращает
    ----------
    Path
        Путь к сохранённому CSV.
    """
    b_paths = list(b_test_paths)
    record_paths = list(record_test_paths)

    base = test_sorted[[REQUEST_ID, VARIANT_ID, pil_col]].copy()
    base[REQUEST_ID] = base[REQUEST_ID].astype(str)
    base[VARIANT_ID] = base[VARIANT_ID].astype("int32")

    is_subtask_b = is_subtask_b_mask(base, pil_col=pil_col)

    # Обе подзадачи — чистое sum(ranks)/N (uniform). Так делается в
    # эталонном two-stage сабмите, который собирался ad-hoc скриптом
    # на основе make_record_blend.py.
    record_blend = rank_avg_blend(record_paths, base, weights=None, divide_after_sum=True)
    if b_blend_weights is None:
        b_blend = rank_avg_blend(b_paths, base, weights=None, divide_after_sum=True)
    else:
        b_blend = rank_avg_blend(b_paths, base, weights=b_blend_weights, divide_after_sum=False)

    scores_final = np.where(is_subtask_b, b_blend, record_blend)
    scores_final = scores_final + PIL_RANK_BOOST * base[pil_col].astype(float).to_numpy()

    out_path = Path(out_path)
    make_submission(base, scores_final, out_path)
    verify_submission(out_path, SAMPLE_SUBMISSION_PATH)

    n_b = int(is_subtask_b.sum())
    LOG.info(
        "two-stage сабмит: %d/%d строк из B-blend (%d моделей), %d из record-blend (%d моделей) -> %s",
        n_b, len(base), len(b_paths),
        len(base) - n_b, len(record_paths),
        out_path.name,
    )
    return out_path
