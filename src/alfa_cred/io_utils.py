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
    LOG.info(
        "Прочитан %s: shape=%s, ~%.1f MB",
        path, df.shape, df.memory_usage(deep=True).sum() / 1024 ** 2,
    )
    return df


def normalize_request_id(df: pd.DataFrame, col: str = REQUEST_ID) -> pd.DataFrame:
    """Приводит `request_id` к строке.

    В исходных данных колонка хранится в train как object (UUID-подобная
    hex-строка), в test как int64. При merge'е и сравнениях это даёт
    ошибки. Приводим оба к строке.
    """
    if col not in df.columns:
        return df
    if df[col].dtype != object:
        df[col] = df[col].astype(str)
    return df


DECIMAL_COLUMNS_TO_FLOAT = ("rate", "eva", "eva_perc", "ncl")


def coerce_decimal_columns(df: pd.DataFrame, columns: Iterable[str] = DECIMAL_COLUMNS_TO_FLOAT) -> pd.DataFrame:
    """Приводит `decimal.Decimal`-колонки к float32.

    В исходных parquet поля `rate`, `eva`, `eva_perc`, `ncl` хранятся как
    `object` (Decimal с произвольной точностью). Для арифметики и моделей
    нужен численный тип.
    """
    for col in columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


def load_raw(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    features_path: Path = FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает три исходные таблицы, нормализует ключи и dtype."""
    train = coerce_decimal_columns(normalize_request_id(load_parquet(train_path)))
    test = coerce_decimal_columns(normalize_request_id(load_parquet(test_path)))
    features = coerce_decimal_columns(load_parquet(features_path))
    return train, test, features


def merge_features(
    df: pd.DataFrame,
    features: pd.DataFrame,
    on: Iterable[str] = (APP_ID, DATE_PART),
    how: str = "left",
) -> pd.DataFrame:
    """Соединяет основную таблицу с признаками клиента и проверяет покрытие."""
    keys = list(on)
    n_before = df.shape[1]
    merged = df.merge(features, on=keys, how=how)
    if how == "left":
        new_cols = merged.columns[n_before:]
        if len(new_cols):
            missing = merged[new_cols].isna().all(axis=1).mean()
            if missing > 0:
                LOG.warning(
                    "После merge %.2f%% строк не имеют признаков клиента", missing * 100
                )
    return merged


def downcast_numeric(df: pd.DataFrame, copy: bool = False) -> pd.DataFrame:
    """Сокращает разрядность числовых типов без потери точности."""
    out = df.copy() if copy else df
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def filter_features_by_fill_rate(
    features: pd.DataFrame,
    min_fill_rate: float = 0.5,
    keep_keys: Iterable[str] = (APP_ID, DATE_PART),
) -> pd.DataFrame:
    """Оставляет колонки с заполненностью не ниже порога.

    Ключевые колонки (`app_id`, `date_part`) всегда остаются, даже если
    у них fill_rate < min_fill_rate (что маловероятно).
    """
    keep_keys = list(keep_keys)
    fill_rate = 1 - features.isna().mean()
    keep_cols = fill_rate[fill_rate >= min_fill_rate].index.tolist()
    for k in keep_keys:
        if k in features.columns and k not in keep_cols:
            keep_cols.insert(0, k)
    LOG.info(
        "Отфильтровано колонок features по fill_rate >= %.2f: %d из %d",
        min_fill_rate, len(keep_cols), features.shape[1],
    )
    return features[keep_cols]


def make_groups(df: pd.DataFrame, request_col: str = REQUEST_ID) -> np.ndarray:
    """Возвращает массив длин групп по `request_col` для LambdaRank.

    Датафрейм должен быть предварительно отсортирован по `request_col`.
    """
    return df.groupby(request_col, sort=False).size().to_numpy()


def sort_by_request(df: pd.DataFrame, request_col: str = REQUEST_ID) -> pd.DataFrame:
    """Стабильная сортировка по `request_id`. Нужна перед `make_groups`."""
    return df.sort_values(request_col, kind="stable").reset_index(drop=True)
