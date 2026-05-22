"""Временные признаки из `request_received`."""

from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame, col: str = "request_received") -> pd.DataFrame:
    """Извлекает час, день недели, флаг выходного.

    Если колонка `request_received` отсутствует или не парсится — пропускает.
    """
    if col not in df.columns:
        return df
    ts = pd.to_datetime(df[col], errors="coerce")
    if ts.isna().all():
        return df
    df["req_hour"] = ts.dt.hour.astype("Int8")
    df["req_dow"] = ts.dt.dayofweek.astype("Int8")
    df["req_is_weekend"] = (ts.dt.dayofweek >= 5).astype("Int8")
    df["req_is_business_hours"] = ((ts.dt.hour >= 9) & (ts.dt.hour < 19)).astype("Int8")
    return df
