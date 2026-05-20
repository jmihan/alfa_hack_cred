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
