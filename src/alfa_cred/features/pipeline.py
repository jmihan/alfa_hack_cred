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
    add_cross_offer_features,
    add_growth_features,
    add_group_aggregates,
    add_group_ranks,
    add_group_size,
    add_offer_rank_features,
    add_pairwise_diffs,
    add_subgroup_ranks,
    add_variant_position_features,
)
from alfa_cred.features.match import (
    MATCH_FEATURE_COLUMNS,
    add_match_features,
    add_pareto_features,
)
from alfa_cred.features.time import add_time_features
from alfa_cred.io_utils import (
    coerce_decimal_columns,
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
    merged = add_subgroup_ranks(merged)
    merged = add_pairwise_diffs(merged)
    merged = add_variant_position_features(merged)
    merged = add_cross_features(merged)
    merged = add_match_features(merged)
    merged = add_pareto_features(merged)
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


def _wide_feature_list(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Список фич и категориальных колонок (object-типа) для широкого набора."""
    drop = {APP_ID, "request_received", DATE_PART, REQUEST_ID, TARGET, "offer_id"}
    feature_cols = [c for c in df.columns if c not in drop]
    cat_cols = [c for c in feature_cols if df[c].dtype == object]
    return feature_cols, cat_cols


def _encode_wide_categoricals(
    train: pd.DataFrame, test: pd.DataFrame, cat_cols: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Общая категориальная кодировка train+test (pandas Categorical)."""
    for c in cat_cols:
        s = pd.concat([train[c].astype("string"), test[c].astype("string")], ignore_index=True)
        cats = pd.Categorical(s).categories
        train[c] = pd.Categorical(train[c].astype("string"), categories=cats)
        test[c] = pd.Categorical(test[c].astype("string"), categories=cats)
    return train, test


def build_wide_feature_table(
    train: pd.DataFrame, test: pd.DataFrame, feats: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Широкий offer-набор для B-бленда подзадачи B.

    В отличие от `build_feature_table`, клиентские признаки мерджатся целиком
    (минус служебные `*_date`), без фильтра по заполненности — это даёт более
    широкий контекст и оказалось сильнее на подзадаче B. Дальше — расширенные
    внутригрупповые ранги/нормализации, кросс-офферные сравнения внутри типа и
    уровня риска, Парето-доминирование и ask-match стек с `is_best_both`.

    Порядок трансформаций зафиксирован — он влияет на состав/порядок колонок и,
    как следствие, на воспроизводимость рекордного сабмита. Возвращает
    `(train, test, feature_cols, cat_cols)`.
    """
    train = coerce_decimal_columns(train)
    test = coerce_decimal_columns(test)
    drop_feat = [c for c in feats.columns if c.endswith("_date")]
    if drop_feat:
        feats = feats.drop(columns=drop_feat)
    train = train.merge(feats, on=[APP_ID, DATE_PART], how="left")
    test = test.merge(feats, on=[APP_ID, DATE_PART], how="left")

    train = add_offer_rank_features(train)
    test = add_offer_rank_features(test)
    train, test = add_cross_offer_features(train, test)
    train = add_growth_features(train)
    test = add_growth_features(test)
    train = add_match_features(train)
    test = add_match_features(test)

    feature_cols, cat_cols = _wide_feature_list(train)
    feature_cols = feature_cols + [c for c in MATCH_FEATURE_COLUMNS if c not in feature_cols]
    train, test = _encode_wide_categoricals(train, test, cat_cols)
    return train, test, feature_cols, cat_cols
