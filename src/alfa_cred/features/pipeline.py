"""Сборка финального feature set из исходных таблиц.

Собирает все трансформации в один поток: базовые фичи, кросс-фичи,
индикаторы, basket-multihot, временные, внутригрупповые ранги/агрегаты,
клиентские фичи через merge.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from alfa_cred.config import APP_ID, DATE_PART, REQUEST_ID, TARGET
from alfa_cred.features.basic import (
    OFFER_CATEGORICAL,
    add_cross_features,
    add_indicator_features,
)
from alfa_cred.features.basket import add_basket_features
from alfa_cred.features.client import (
    add_client_offer_interactions,
    select_client_features,
)
from alfa_cred.features.group import (
    add_group_aggregates,
    add_group_ranks,
    add_group_size,
)
from alfa_cred.features.time import add_time_features
from alfa_cred.io_utils import (
    downcast_numeric,
    filter_features_by_fill_rate,
    merge_features,
    sort_by_request,
)
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


def build_feature_table(
    df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    min_fill_rate: float = 0.5,
    is_train: bool = True,
) -> pd.DataFrame:
    """Собирает финальную таблицу с фичами.

    Шаги:
    1. Merge с отфильтрованными клиентскими признаками.
    2. Сортировка по `request_id` (нужно для группировки LambdaRank).
    3. Внутригрупповые ранги, агрегаты, размер группы.
    4. Кросс-фичи и индикаторы оффера.
    5. Multi-hot из `basket_name`.
    6. Временные признаки из `request_received`.
    7. Interaction-фичи клиент × оффер.
    8. Downcast числовых типов.
    """
    LOG.info("Build features: shape=%s, is_train=%s", df.shape, is_train)

    features_slim = filter_features_by_fill_rate(features, min_fill_rate=min_fill_rate)
    features_slim = select_client_features(features_slim)
    LOG.info("Колонок клиентских после отбора: %d", features_slim.shape[1])

    merged = merge_features(df, features_slim, on=(APP_ID, DATE_PART), how="left")
    merged = sort_by_request(merged)

    merged = add_group_ranks(merged)
    merged = add_group_aggregates(merged)
    merged = add_group_size(merged)
    merged = add_cross_features(merged)
    merged = add_indicator_features(merged)
    merged = add_basket_features(merged)
    merged = add_time_features(merged)
    merged = add_client_offer_interactions(merged)

    merged = downcast_numeric(merged)
    LOG.info("Финальная таблица: shape=%s", merged.shape)
    return merged


def feature_columns(
    df: pd.DataFrame,
    extra_drop: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Возвращает (feature_columns, categorical_columns) для модели.

    Дропаем все ID-колонки, таргет и сырые datetime/object-поля, которые
    модель не умеет переваривать напрямую.
    """
    drop = {
        APP_ID, REQUEST_ID, "offer_id", "request_received", DATE_PART, TARGET,
        "basket_name",
        *extra_drop,
    }
    feature_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        if col in drop:
            continue
        dtype = df[col].dtype
        if dtype == "object" or str(dtype) == "category":
            categorical_cols.append(col)
            feature_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            feature_cols.append(col)
            if col in OFFER_CATEGORICAL and col != "pil1mtrx_offer":
                categorical_cols.append(col)
        else:
            # datetime, timedelta — пропускаем
            continue
    return feature_cols, categorical_cols
