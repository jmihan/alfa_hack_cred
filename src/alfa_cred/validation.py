"""Стратегии валидации: GroupKFold по `request_id` и time-based split."""

from __future__ import annotations

from typing import Iterator, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from alfa_cred.config import DATE_PART, RANDOM_STATE, REQUEST_ID

CvScheme = Literal["group", "time"]


def make_cv_splits(
    df: pd.DataFrame,
    scheme: CvScheme = "group",
    n_splits: int = 5,
    request_col: str = REQUEST_ID,
    date_col: str = DATE_PART,
    random_state: int = RANDOM_STATE,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Возвращает итератор (train_idx, val_idx) согласно выбранной схеме."""
    if scheme == "group":
        yield from _group_kfold(df, n_splits=n_splits, request_col=request_col)
    elif scheme == "time":
        yield from _time_split(df, n_splits=n_splits, date_col=date_col)
    else:
        raise ValueError(f"Неизвестная схема CV: {scheme!r}")


def _group_kfold(
    df: pd.DataFrame,
    n_splits: int,
    request_col: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    groups = df[request_col].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    for tr, va in splitter.split(np.zeros(len(df)), groups=groups):
        yield tr, va


def _time_split(
    df: pd.DataFrame,
    n_splits: int,
    date_col: str,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Хронологический split: фолд k обучается на первых k частях, валидируется на (k+1).

    В отличие от sklearn TimeSeriesSplit, дробит по уникальным датам, чтобы
    запросы одного дня целиком попадали в одну часть.
    """
    dates = np.sort(df[date_col].unique())
    fold_dates = np.array_split(dates, n_splits + 1)
    date_to_fold = {d: i for i, fold in enumerate(fold_dates) for d in fold}
    fold_assignment = df[date_col].map(date_to_fold).to_numpy()
    for k in range(1, n_splits + 1):
        train_idx = np.where(fold_assignment < k)[0]
        val_idx = np.where(fold_assignment == k)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx
