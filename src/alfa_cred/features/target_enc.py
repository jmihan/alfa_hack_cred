"""Out-of-fold target encoding для категориальных признаков.

Считаем сглаженное среднее таргета по категории, обучая на K-1 фолдах и
применяя на оставшемся фолде. Это снимает leakage и при этом работает
лучше, чем глобальный target rate.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from alfa_cred.config import RANDOM_STATE, TARGET


def smoothed_target_mean(
    series: pd.Series,
    target: pd.Series,
    smoothing: float = 10.0,
) -> pd.Series:
    """Сглаженное среднее таргета по категории.

    `enc(c) = (n_c * mean_c + m * global_mean) / (n_c + m)`,
    где `m` — параметр сглаживания.
    """
    global_mean = float(target.mean())
    stats = target.groupby(series).agg(["count", "mean"])
    weighted = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    return series.map(weighted).fillna(global_mean)


def oof_target_encode(
    df: pd.DataFrame,
    columns: Iterable[str],
    target_col: str = TARGET,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Считает OOF target encoding на train и возвращает full-data maps для test.

    Возвращает:
    - словарь новых колонок `{col_name: encoded_series}` для добавления в train;
    - словарь полных энкодингов `{col: series_by_category}` для применения к test.
    """
    target = df[target_col]
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    encoded_columns: dict[str, pd.Series] = {}
    full_maps: dict[str, pd.Series] = {}

    for col in columns:
        if col not in df.columns:
            continue
        oof = pd.Series(np.nan, index=df.index, dtype="float64")
        for train_idx, val_idx in folds.split(df):
            enc = smoothed_target_mean(
                df[col].iloc[train_idx], target.iloc[train_idx], smoothing=smoothing
            )
            oof.iloc[val_idx] = df[col].iloc[val_idx].map(
                enc.groupby(df[col].iloc[train_idx]).first()
            ).fillna(target.iloc[train_idx].mean()).values
        encoded_columns[f"{col}_te"] = oof.astype("float32")
        # Полный энкодинг — обучен на всех train-фолдах, применяется к test
        full_enc = smoothed_target_mean(df[col], target, smoothing=smoothing)
        full_maps[col] = full_enc.groupby(df[col]).first()

    return pd.DataFrame(encoded_columns, index=df.index), full_maps


def apply_target_encoding(
    df: pd.DataFrame,
    full_maps: dict[str, pd.Series],
    global_mean: float,
) -> pd.DataFrame:
    """Применяет сохранённые энкодинги к test (или новому датасету)."""
    out = pd.DataFrame(index=df.index)
    for col, mapping in full_maps.items():
        if col in df.columns:
            out[f"{col}_te"] = df[col].map(mapping).fillna(global_mean).astype("float32")
    return out
