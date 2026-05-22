"""Отбор клиентских признаков из `features_small.pq`.

По EDA значимые сигналы концентрируются вокруг BKI/HDB BKI (кредитная
активность, recency) и счётчиков сторонних кредитов. Демография/доход
сами по себе слабее. Здесь — короткий курируемый список фичей плюс
утилита для расширения по fill-rate.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Курируемый список: топ-сигналы по Spearman с pos_rate клиента
# (см. notebooks/EDA_FINDINGS.md, секция "Клиентские признаки").
PRIORITY_NUMERIC = (
    "age",
    "life_time_days",
    "other_credits_count",
    "bki_last_product_days",
    "bki_total_active_products",
    "bki_total_products",
    "bki_total_cnt",
    "bki_total_il_cnt",
    "bki_total_cc_cnt",
    "bki_total_micro_cnt",
    "bki_total_max_limit",
    "bki_active_max_limit",
    "salary_last_days",
    "socialparamchange_days",
    "winback_cnt",
    "hdb_relend_active_max_psk",
    "hdb_other_active_max_psk",
    "hdb_bki_total_active_products",
    "hdb_bki_active_pil_cnt",
    "hdb_bki_total_products",
    "hdb_bki_last_product_days",
    "hdb_other_active_credits_count",
    "hdb_other_credits_count",
    "hdb_relend_outstand_sum",
    "hdb_bki_active_micro_cnt",
    "hdb_bki_total_cnt",
    "hdb_bki_total_micro_cnt",
    "hdb_bki_total_pil_cnt",
    "hdb_bki_total_pil_last_days",
    "hdb_bki_total_cc_last_days",
    "hdb_relend_active_mean_psk",
    "hdb_other_active_mean_psk",
    "loan_cnt",
    "term_utilization",
)

PRIORITY_CATEGORICAL = (
    "gender",
    "country",
    "clientsegment",
    "clientsegment_new",
    "clientsegment_prd",
    "clientsegment_prd_out",
    "clientsegment_out",
    "clientgroup",
    "clienttype",
    "clientoutflowstatus",
    "srvpackage",
    "srvpackage_fst",
    "stratsegfactor",
    "adminarea",
    "city_smart_name",
    "tenor_name",
    "lp_client_group_name",
    "n2b_group",
    "profuct_upsell_flag",
    "realty_flag",
    "otherbankdeposit_flag",
    "otherbankaccount_flag",
    "izk_sts",
    "izk_sts_leave",
)

PRIORITY_FLAGS = (
    "vip_flag",
    "staff_flag",
    "dead_flag",
    "blacklist_flag",
    "blacklist_employer_flag",
    "client_active_flag",
    "account_active_flag",
    "accountsalary_flag",
    "accountsalary_out_flag",
    "nonresident_flag",
    "primarybank_out_flag",
    "izk_in_past_2y",
    "asmart_sub_active_flag",
    "asmart_extra_flag",
    "r200_flag",
    "r700_flag",
    "r800_flag",
)


def select_client_features(
    features: pd.DataFrame,
    extra_numeric: Iterable[str] = (),
    extra_categorical: Iterable[str] = (),
) -> pd.DataFrame:
    """Возвращает обрезанную копию `features_small.pq` с интересными колонками."""
    keys = [c for c in ("app_id", "date_part") if c in features.columns]
    numeric = [c for c in (*PRIORITY_NUMERIC, *PRIORITY_FLAGS, *extra_numeric) if c in features.columns]
    categorical = [c for c in (*PRIORITY_CATEGORICAL, *extra_categorical) if c in features.columns]
    cols = keys + numeric + categorical
    seen: set[str] = set()
    deduped = [c for c in cols if not (c in seen or seen.add(c))]
    return features[deduped].copy()


def add_client_offer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Несколько вручную выбранных interaction-фичей клиент × оффер.

    Идея — дать модели сигнал «насколько данный оффер совпадает с
    кредитным профилем клиента».
    """
    interactions: dict[str, pd.Series] = {}
    if {"ncl", "bki_total_active_products"}.issubset(df.columns):
        interactions["ncl_x_bki_total_active_products"] = df["ncl"] * df["bki_total_active_products"]
    if {"rate", "age"}.issubset(df.columns):
        interactions["rate_x_age"] = df["rate"] * df["age"]
    if {"limit", "salary_last_days"}.issubset(df.columns):
        interactions["limit_div_salary_last_days"] = df["limit"] / df["salary_last_days"].replace(0, np.nan)
    if {"req_loan_amount", "other_credits_count"}.issubset(df.columns):
        interactions["req_loan_x_other_credits"] = df["req_loan_amount"] * df["other_credits_count"].fillna(0)
    if {"term", "hdb_bki_last_product_days"}.issubset(df.columns):
        interactions["term_minus_hdb_recency"] = df["term"] - df["hdb_bki_last_product_days"].fillna(0)
    for name, series in interactions.items():
        df[name] = series.astype("float32")
    return df
