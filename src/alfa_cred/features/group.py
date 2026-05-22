"""Внутригрупповые признаки (ранги, отклонения от среднего по request_id).

Эти признаки — самый прямой способ дать модели LambdaRank информацию
о «положении» оффера внутри карусели запроса.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alfa_cred.config import REQUEST_ID

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
