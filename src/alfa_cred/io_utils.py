"""Загрузка и подготовка исходных parquet-таблиц."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alfa_cred.config import (
    APP_ID,
    DATE_PART,
    FEATURES_PATH,
    REQUEST_ID,
    TEST_PATH,
    TRAIN_PATH,
)
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Читает parquet и логирует размер."""
    df = pd.read_parquet(path)
    LOG.info("Прочитан %s: shape=%s, ~%.1f MB", path, df.shape, df.memory_usage(deep=True).sum() / 1024 ** 2)
    return df


def load_raw(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    features_path: Path = FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает три исходные таблицы хакатона."""
    return load_parquet(train_path), load_parquet(test_path), load_parquet(features_path)


def merge_features(
    df: pd.DataFrame,
    features: pd.DataFrame,
    on: Iterable[str] = (APP_ID, DATE_PART),
    how: str = "left",
) -> pd.DataFrame:
    """Соединяет основную таблицу с признаками клиента и проверяет покрытие."""
    keys = list(on)
    merged = df.merge(features, on=keys, how=how)
    if how == "left":
        missing = merged.iloc[:, len(df.columns):].isna().all(axis=1).mean()
        if missing > 0:
            LOG.warning("После merge %.2f%% строк не имеют признаков клиента", missing * 100)
    return merged


def downcast_numeric(df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
    """Сокращает разрядность числовых типов без потери точности."""
    out = df.copy() if copy else df
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def make_groups(df: pd.DataFrame, request_col: str = REQUEST_ID) -> np.ndarray:
    """Возвращает массив длин групп по `request_col` для LambdaRank.

    Датафрейм должен быть предварительно отсортирован по `request_col`.
    """
    return df.groupby(request_col, sort=False).size().to_numpy()
