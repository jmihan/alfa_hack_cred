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
