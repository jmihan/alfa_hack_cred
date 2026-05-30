"""Внутригрупповые признаки (ранги, отклонения от среднего по request_id).

Эти признаки — самый прямой способ дать модели LambdaRank информацию
о «положении» оффера внутри карусели запроса.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alfa_cred.config import REQUEST_ID, VARIANT_ID
from alfa_cred.features.match import add_pareto_features

GROUP_RANK_COLUMNS = ("rate", "term", "limit", "eva", "eva_perc", "ncl")


def add_group_ranks(
    df: pd.DataFrame,
    columns: tuple[str, ...] = GROUP_RANK_COLUMNS,
    request_col: str = REQUEST_ID,
) -> pd.DataFrame:
    """Добавляет внутригрупповые ранги и нормированные ранги.

    Для каждой колонки:
    - `<col>_rank` — целочисленный ранг (method='first');
    - `<col>_pct_rank` — нормированный ранг в [0, 1];
    - `<col>_zscore` — отклонение от среднего по группе в std-единицах;
    - `<col>_gap_to_mean` — простая разница со средним группы;
    - `<col>_gap_to_min` — разница с минимумом группы.
    """
    grouped = df.groupby(request_col, sort=False)
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        df[f"{col}_rank"] = grouped[col].rank(method="first").astype("float32")
        df[f"{col}_pct_rank"] = grouped[col].rank(pct=True).astype("float32")
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0, np.nan)
        df[f"{col}_zscore"] = ((series - mean) / std).astype("float32")
        df[f"{col}_gap_to_mean"] = (series - mean).astype("float32")
        df[f"{col}_gap_to_min"] = (series - grouped[col].transform("min")).astype("float32")
    return df


def add_group_aggregates(
    df: pd.DataFrame,
    columns: tuple[str, ...] = GROUP_RANK_COLUMNS,
    request_col: str = REQUEST_ID,
) -> pd.DataFrame:
    """Добавляет агрегаты по группе (mean/min/max/range).

    Полезно для модели — она видит, насколько данный оффер
    «выделяется» в карусели.
    """
    grouped = df.groupby(request_col, sort=False)
    for col in columns:
        if col not in df.columns:
            continue
        df[f"{col}_grp_min"] = grouped[col].transform("min").astype("float32")
        df[f"{col}_grp_max"] = grouped[col].transform("max").astype("float32")
        df[f"{col}_grp_range"] = (df[f"{col}_grp_max"] - df[f"{col}_grp_min"]).astype("float32")
    return df


def add_group_size(df: pd.DataFrame, request_col: str = REQUEST_ID) -> pd.DataFrame:
    """Размер запроса (число вариантов в карусели)."""
    df["group_size"] = df.groupby(request_col, sort=False)[request_col].transform("size").astype("int16")
    return df


SUBGROUP_RANK_KEYS = ("offer_type", "risk_level_map")


def add_subgroup_ranks(
    df: pd.DataFrame,
    columns: tuple[str, ...] = GROUP_RANK_COLUMNS,
    subgroup_keys: tuple[str, ...] = SUBGROUP_RANK_KEYS,
    request_col: str = REQUEST_ID,
) -> pd.DataFrame:
    """Ранги признака внутри пары `(request_id, subgroup_key)`.

    Например, какое место у оффера среди других RA-офферов этого же
    запроса. Это даёт модели сигнал «лучший из RA», «лучший из
    высокорисковых» и т. п. — намного информативнее, чем просто ранг
    среди всех вариантов.
    """
    for sub_key in subgroup_keys:
        if sub_key not in df.columns:
            continue
        grouped = df.groupby([request_col, sub_key], sort=False)
        for col in columns:
            if col not in df.columns:
                continue
            df[f"{col}_rank_in_{sub_key}"] = grouped[col].rank(pct=True).astype("float32")
            df[f"{col}_gap_to_min_in_{sub_key}"] = (
                df[col] - grouped[col].transform("min")
            ).astype("float32")
    return df


def add_pairwise_diffs(
    df: pd.DataFrame,
    columns: tuple[str, ...] = GROUP_RANK_COLUMNS,
    request_col: str = REQUEST_ID,
) -> pd.DataFrame:
    """Разницы признака между текущим оффером и экстремумами группы.

    Идея — дать модели сигнал «насколько данный оффер отличается от
    самого выгодного / самого рискованного в карусели».
    """
    grouped = df.groupby(request_col, sort=False)
    for col in columns:
        if col not in df.columns:
            continue
        col_max = grouped[col].transform("max")
        col_min = grouped[col].transform("min")
        df[f"{col}_diff_to_max"] = (df[col] - col_max).astype("float32")
        df[f"{col}_diff_to_min"] = (df[col] - col_min).astype("float32")
        # Относительная разница (нормализованная диапазоном группы)
        span = (col_max - col_min).replace(0, np.nan)
        df[f"{col}_rel_position"] = ((df[col] - col_min) / span).astype("float32")
    return df


def add_variant_position_features(df: pd.DataFrame, request_col: str = REQUEST_ID) -> pd.DataFrame:
    """Нормированная позиция variant_no внутри карусели.

    `variant_no` — это бизнес-порядок, но в одном запросе их может быть
    1-50. Нормированная позиция в [0, 1] — более стабильный сигнал.
    """
    if "variant_no" not in df.columns:
        return df
    grouped = df.groupby(request_col, sort=False)
    max_variant = grouped["variant_no"].transform("max")
    df["variant_no_norm"] = (df["variant_no"] / max_variant).astype("float32")
    df["variant_no_pct_rank"] = grouped["variant_no"].rank(pct=True).astype("float32")
    return df


# ---------------------------------------------------------------------------
# Широкий offer-набор для подзадачи B.
#
# Расширенные внутригрупповые сравнения оффера: ранги/нормализации, кросс-офферные
# сравнения внутри типа и уровня риска, отклонения от экстремумов группы. Здесь
# `<col>_rank` — это перцентильный ранг (`pct=True`), в отличие от `add_group_ranks`
# выше, где `<col>_rank` — целочисленный ранг (`method="first"`); это два разных
# набора признаков для двух разных feature-таблиц (`build_feature_table` против
# `build_wide_feature_table`), поэтому имена совпадают, а смысл — нет.
# ---------------------------------------------------------------------------

OFFER_MINMAX_COLUMNS = ("rate", "eva", "eva_perc", "ncl", "limit", "term")


def add_offer_rank_features(df: pd.DataFrame, request_col: str = REQUEST_ID) -> pd.DataFrame:
    """Внутригрупповые ранги, min-max нормализации и индикаторы оффера.

    Создаёт размер запроса/позицию `variant_no`, абсолютные и относительные
    разрывы суммы/срока, для ключевых числовых колонок — min-max, отклонение от
    среднего и перцентильный ранг, и индикаторы «лучший по rate/eva/limit».
    """
    g = df.groupby(request_col, sort=False)
    df["req_n_offers"] = g[VARIANT_ID].transform("size").astype("int16")
    df["req_max_variant"] = g[VARIANT_ID].transform("max").astype("int16")
    df["variant_no_norm"] = (df[VARIANT_ID] / df["req_max_variant"]).astype("float32")
    df["variant_no_inv"] = (1.0 / df[VARIANT_ID].astype("float32"))

    df["limit_gap"] = (df["limit"] - df["req_loan_amount"]).astype("float32")
    df["limit_ratio"] = (df["limit"] / (df["req_loan_amount"] + 1)).astype("float32")
    df["term_gap"] = (df["term"] - df["req_term"]).astype("float32")
    df["term_ratio"] = (df["term"] / (df["req_term"] + 1)).astype("float32")

    for col in OFFER_MINMAX_COLUMNS:
        gmin = g[col].transform("min")
        gmax = g[col].transform("max")
        gmean = g[col].transform("mean")
        rng = (gmax - gmin).replace(0, np.nan)
        df[f"{col}_minmax"] = ((df[col] - gmin) / rng).astype("float32").fillna(0.5)
        df[f"{col}_dev_mean"] = (df[col] - gmean).astype("float32")
        df[f"{col}_rank"] = g[col].rank(method="average", pct=True).astype("float32")

    df["same_type_cnt"] = g["offer_type"].transform("count").astype("int16")
    df["is_lowest_rate"] = (df["rate_rank"] <= (1.0 / df["req_n_offers"]) + 1e-6).astype("int8")
    df["is_highest_eva"] = (df["eva_rank"] >= 1 - 1e-6).astype("int8")
    df["is_max_limit"] = (df["limit_rank"] >= 1 - 1e-6).astype("int8")
    return df


def add_cross_offer_features(train: pd.DataFrame, test: pd.DataFrame, request_col: str = REQUEST_ID):
    """Кросс-офферные сравнения внутри запроса, типа оффера и уровня риска."""
    for df in (train, test):
        g = df.groupby(request_col, sort=False)
        df["req_unique_offer_types"] = g["offer_type"].transform("nunique").astype("int8")
        df["req_unique_risks"] = g["risk_level_map"].transform("nunique").astype("int8")
        df["req_unique_baskets"] = g["basket_name"].transform("nunique").astype("int8")
        df["req_unique_terms"] = g["term"].transform("nunique").astype("int8")
        df["req_unique_rates"] = g["rate"].transform("nunique").astype("int8")
        for col in ["rate", "eva", "eva_perc", "limit", "ncl"]:
            df[f"{col}_rank_in_type"] = df.groupby([request_col, "offer_type"])[col].rank(method="average", pct=True).astype("float32")
            df[f"{col}_rank_in_risk"] = df.groupby([request_col, "risk_level_map"])[col].rank(method="average", pct=True).astype("float32")
        for col in ["rate", "ncl"]:
            df[f"{col}_share_lower"] = df.groupby(request_col)[col].rank(pct=True, ascending=True).astype("float32")
        for col in ["eva", "eva_perc", "limit"]:
            df[f"{col}_share_better"] = df.groupby(request_col)[col].rank(pct=True, ascending=False).astype("float32")
    return train, test


def add_growth_features(df: pd.DataFrame, request_col: str = REQUEST_ID) -> pd.DataFrame:
    """Отклонения от экстремумов группы + Парето-доминирование (rate↓, limit↑, ncl↓).

    Относительные отношения к экстремумам и Парето-признаки вынесены в
    `match.add_pareto_features` — переиспользуем его, чтобы не дублировать логику.
    """
    g = df.groupby(request_col, sort=False)
    df["rate_gap_to_min"] = (df["rate"] - g["rate"].transform("min")).astype("float32")
    df["ncl_gap_to_min"] = (df["ncl"] - g["ncl"].transform("min")).astype("float32")
    df["limit_gap_to_max"] = (g["limit"].transform("max") - df["limit"]).astype("float32")
    df["eva_gap_to_max"] = (g["eva"].transform("max") - df["eva"]).astype("float32")
    df["evaperc_gap_to_max"] = (g["eva_perc"].transform("max") - df["eva_perc"]).astype("float32")
    df = add_pareto_features(df, request_col=request_col)
    return df
