"""Расширенный offer-feature набор для рекордного B-бленда (LB ≈ 92.18).

Этот модуль собирает «широкий» набор признаков оффера для подзадачи B —
внутригрупповые ранги/нормализации, кросс-офферные сравнения внутри типа и
уровня риска, Парето-доминирование и ask-match признаки (`is_best_both`).
В отличие от `features/pipeline.py:build_feature_table`, здесь клиентские
признаки берутся целиком (минус служебные `*_date`), без фильтра по
заполненности — это даёт более широкий контекст и оказалось сильнее на
подзадаче B.

Порядок трансформаций зафиксирован — он влияет на состав/имена колонок и, как
следствие, на воспроизводимость рекордного сабмита.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from alfa_cred.config import APP_ID, DATE_PART, REQUEST_ID, TARGET, VARIANT_ID

# Decimal-колонки исходных данных (хранятся как object) — приводим к float32.
DECIMAL_COLS = ("rate", "eva", "eva_perc", "ncl")

# ask-match признаки (главный из них — `is_best_both`).
REQMATCH_COLS = (
    "lim_match", "term_match", "both_match", "dlim_abs", "dterm_abs",
    "n_both", "is_uniq_both", "min_vn_both", "is_best_both", "vn_rank_in_both",
)


def cast_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит decimal-колонки оффера к float32."""
    for c in DECIMAL_COLS:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: float(x) if isinstance(x, Decimal) else x).astype("float32")
    return df


def add_offer_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Внутригрупповые ранги, нормализации и индикаторы оффера."""
    g = df.groupby(REQUEST_ID, sort=False)
    df["req_n_offers"] = g[VARIANT_ID].transform("size").astype("int16")
    df["req_max_variant"] = g[VARIANT_ID].transform("max").astype("int16")
    df["variant_no_norm"] = (df[VARIANT_ID] / df["req_max_variant"]).astype("float32")
    df["variant_no_inv"] = (1.0 / df[VARIANT_ID].astype("float32"))

    df["limit_gap"] = (df["limit"] - df["req_loan_amount"]).astype("float32")
    df["limit_ratio"] = (df["limit"] / (df["req_loan_amount"] + 1)).astype("float32")
    df["term_gap"] = (df["term"] - df["req_term"]).astype("float32")
    df["term_ratio"] = (df["term"] / (df["req_term"] + 1)).astype("float32")

    for col in ["rate", "eva", "eva_perc", "ncl", "limit", "term"]:
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


def add_cross_offer_features(train: pd.DataFrame, test: pd.DataFrame):
    """Кросс-офферные сравнения внутри запроса, типа оффера и уровня риска."""
    for df in (train, test):
        g = df.groupby(REQUEST_ID, sort=False)
        df["req_unique_offer_types"] = g["offer_type"].transform("nunique").astype("int8")
        df["req_unique_risks"] = g["risk_level_map"].transform("nunique").astype("int8")
        df["req_unique_baskets"] = g["basket_name"].transform("nunique").astype("int8")
        df["req_unique_terms"] = g["term"].transform("nunique").astype("int8")
        df["req_unique_rates"] = g["rate"].transform("nunique").astype("int8")
        for col in ["rate", "eva", "eva_perc", "limit", "ncl"]:
            df[f"{col}_rank_in_type"] = df.groupby([REQUEST_ID, "offer_type"])[col].rank(method="average", pct=True).astype("float32")
            df[f"{col}_rank_in_risk"] = df.groupby([REQUEST_ID, "risk_level_map"])[col].rank(method="average", pct=True).astype("float32")
        for col in ["rate", "ncl"]:
            df[f"{col}_share_lower"] = df.groupby(REQUEST_ID)[col].rank(pct=True, ascending=True).astype("float32")
        for col in ["eva", "eva_perc", "limit"]:
            df[f"{col}_share_better"] = df.groupby(REQUEST_ID)[col].rank(pct=True, ascending=False).astype("float32")
    return train, test


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Отклонения от экстремумов группы и Парето-доминирование (rate↓, limit↑, ncl↓)."""
    g = df.groupby(REQUEST_ID, sort=False)
    df["rate_gap_to_min"] = (df["rate"] - g["rate"].transform("min")).astype("float32")
    df["ncl_gap_to_min"] = (df["ncl"] - g["ncl"].transform("min")).astype("float32")
    df["limit_gap_to_max"] = (g["limit"].transform("max") - df["limit"]).astype("float32")
    df["eva_gap_to_max"] = (g["eva"].transform("max") - df["eva"]).astype("float32")
    df["evaperc_gap_to_max"] = (g["eva_perc"].transform("max") - df["eva_perc"]).astype("float32")
    df["rate_ratio_to_min"] = (df["rate"] / (g["rate"].transform("min") + 1e-6)).astype("float32")
    df["limit_ratio_to_max"] = (df["limit"] / (g["limit"].transform("max") + 1e-6)).astype("float32")

    def _dominated(sub: pd.DataFrame) -> pd.Series:
        r = sub["rate"].values; l = sub["limit"].values; n = sub["ncl"].values
        weak = (r[None, :] <= r[:, None]) & (l[None, :] >= l[:, None]) & (n[None, :] <= n[:, None])
        strict = (r[None, :] < r[:, None]) | (l[None, :] > l[:, None]) | (n[None, :] < n[:, None])
        dom = weak & strict
        np.fill_diagonal(dom, False)
        return pd.Series(dom.sum(axis=1), index=sub.index)

    df["pareto_dominated_cnt"] = (
        g[["rate", "limit", "ncl"]].apply(_dominated).reset_index(level=0, drop=True).astype("float32")
    )
    df["is_pareto_optimal"] = (df["pareto_dominated_cnt"] == 0).astype("int8")
    return df


def add_reqmatch_features(df: pd.DataFrame) -> pd.DataFrame:
    """ask-match признаки: точное совпадение оффера с заявкой + `is_best_both`.

    `is_best_both` — оффер, у которого `limit == req_loan_amount` И
    `term == req_term`, и он первый (минимальный `variant_no`) среди таких.
    Deal-rate ≈ 52% против ≈2.8% — главный сигнал подзадачи B.
    """
    df["lim_match"] = (df["limit"] == df["req_loan_amount"]).astype("int8")
    df["term_match"] = (df["term"] == df["req_term"]).astype("int8")
    df["both_match"] = (df["lim_match"] * df["term_match"]).astype("int8")
    df["dlim_abs"] = (df["limit"] - df["req_loan_amount"]).abs().astype("float32")
    df["dterm_abs"] = (df["term"] - df["req_term"]).abs().astype("float32")
    g = df.groupby(REQUEST_ID, sort=False)
    df["n_both"] = g["both_match"].transform("sum").astype("int16")
    df["is_uniq_both"] = ((df["n_both"] == 1) & (df["both_match"] == 1)).astype("int8")
    df["_vn_if_both"] = np.where(df["both_match"] == 1, df[VARIANT_ID], np.inf)
    df["min_vn_both"] = df.groupby(REQUEST_ID)["_vn_if_both"].transform("min")
    df["is_best_both"] = ((df["both_match"] == 1) & (df[VARIANT_ID] == df["min_vn_both"])).astype("int8")
    df["vn_rank_in_both"] = np.where(
        df["both_match"] == 1, df.groupby(REQUEST_ID)["_vn_if_both"].rank(method="first"), 0
    ).astype("float32")
    df["min_vn_both"] = df["min_vn_both"].replace(np.inf, 0).astype("float32")
    df.drop(columns=["_vn_if_both"], inplace=True)
    return df


def build_feature_list(df: pd.DataFrame):
    """Список фич и категориальных колонок (object-типа)."""
    drop = {APP_ID, "request_received", DATE_PART, REQUEST_ID, TARGET, "offer_id"}
    feature_cols = [c for c in df.columns if c not in drop]
    cat_cols = [c for c in feature_cols if df[c].dtype == object]
    return feature_cols, cat_cols


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame, cat_cols):
    """Общая категориальная кодировка train+test (pandas Categorical)."""
    for c in cat_cols:
        s = pd.concat([train[c].astype("string"), test[c].astype("string")], ignore_index=True)
        cats = pd.Categorical(s).categories
        train[c] = pd.Categorical(train[c].astype("string"), categories=cats)
        test[c] = pd.Categorical(test[c].astype("string"), categories=cats)
    return train, test


def build_record_feature_table(train: pd.DataFrame, test: pd.DataFrame, feats: pd.DataFrame):
    """Полная сборка широкого offer-набора для рекордного B-бленда.

    Возвращает (train, test, feature_cols, cat_cols). Клиентские признаки
    мерджатся целиком (минус `*_date`), категориальные кодируются Categorical.
    """
    train = cast_decimals(train); test = cast_decimals(test)
    drop_feat = [c for c in feats.columns if c.endswith("_date")]
    if drop_feat:
        feats = feats.drop(columns=drop_feat)
    train = train.merge(feats, on=[APP_ID, DATE_PART], how="left")
    test = test.merge(feats, on=[APP_ID, DATE_PART], how="left")

    train = add_offer_rank_features(train); test = add_offer_rank_features(test)
    train, test = add_cross_offer_features(train, test)
    train = add_growth_features(train); test = add_growth_features(test)
    train = add_reqmatch_features(train); test = add_reqmatch_features(test)

    feature_cols, cat_cols = build_feature_list(train)
    feature_cols = feature_cols + [c for c in REQMATCH_COLS if c not in feature_cols]
    train, test = encode_categoricals(train, test, cat_cols)
    return train, test, feature_cols, cat_cols
