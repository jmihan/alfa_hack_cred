"""Базовые признаки оффера и заявки.

Простые трансформации над исходными колонками: производные суммы,
отношения, индикаторы. Без агрегаций и target encoding — они вынесены
в отдельные модули.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OFFER_NUMERIC = ("rate", "term", "limit", "eva", "eva_perc", "ncl")
REQUEST_NUMERIC = ("req_loan_amount", "req_term")
OFFER_CATEGORICAL = (
    "offer_type",
    "risk_level_map",
    "channel",
    "verif_compl",
    "verif_need",
    "need_2ndfl",
    "pil1mtrx_offer",
)


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Производные признаки на пересечении полей оффера и заявки."""
    if {"eva", "limit"}.issubset(df.columns):
        df["eva_per_limit"] = df["eva"] / df["limit"].replace(0, np.nan)
    if {"ncl", "rate"}.issubset(df.columns):
        df["ncl_x_rate"] = df["ncl"] * df["rate"]
    if {"term", "rate"}.issubset(df.columns):
        df["term_x_rate"] = df["term"] * df["rate"]
    if {"req_loan_amount", "limit"}.issubset(df.columns):
        df["req_to_limit_ratio"] = df["req_loan_amount"] / df["limit"].replace(0, np.nan)
        df["req_minus_limit"] = df["req_loan_amount"] - df["limit"]
    if {"req_term", "term"}.issubset(df.columns):
        df["req_minus_term"] = df["req_term"] - df["term"]
    return df


def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Бинарные индикаторы по очевидным паттернам из EDA."""
    if "variant_no" in df.columns:
        df["is_first_variant"] = (df["variant_no"] == 1).astype("int8")
        df["variant_le_5"] = (df["variant_no"] <= 5).astype("int8")
        df["variant_le_10"] = (df["variant_no"] <= 10).astype("int8")
    if "need_2ndfl" in df.columns:
        df["need_2ndfl_y"] = (df["need_2ndfl"] == "Y").astype("int8")
    return df
