"""Формирование сабмит-файла в формате Яндекс Контеста."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alfa_cred.config import (
    REQUEST_ID,
    SUBMISSION_COLUMNS,
    SUBMISSION_SCORE_DECIMALS,
    SUBMISSION_SEPARATOR,
    VARIANT_ID,
)
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)

PIL_HARD_RULE_BOOST = 1e6


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
