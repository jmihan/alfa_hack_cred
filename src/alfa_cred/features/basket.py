"""Парсинг `basket_name` в multi-hot бизнес-правил.

Поле хранит comma-separated список правил, под которые подходит оффер
(`desiredparams`, `minannuity`, `maxlimit`, `noverifdocmaxlimit`).
Согласно EDA, `desiredparams` и `minannuity` дают target rate ~49%,
поэтому это сильный сигнал, несмотря на 95% пропусков.
"""

from __future__ import annotations

import pandas as pd

BASKET_TOKENS = ("desiredparams", "minannuity", "maxlimit", "noverifdocmaxlimit")


def add_basket_features(
    df: pd.DataFrame,
    col: str = "basket_name",
    tokens: tuple[str, ...] = BASKET_TOKENS,
) -> pd.DataFrame:
    """Multi-hot колонки для каждого известного токена + флаг наличия."""
    if col not in df.columns:
        for t in tokens:
            df[f"basket_{t}"] = 0
        df["basket_present"] = 0
        return df

    df["basket_present"] = df[col].notna().astype("int8")
    series = df[col].fillna("")
    for t in tokens:
        df[f"basket_{t}"] = series.str.contains(t, regex=False).astype("int8")
    return df
