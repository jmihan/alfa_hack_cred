"""Признаки соответствия оффера запросу клиента (ask-matching).

Находка из разрезов по подзадаче B: оффер, который точно повторяет заявку
клиента (сумма И срок совпадают) и стоит у банка первым среди таких,
становится сделкой почти в половине случаев — deal-rate ≈ 52% против ≈2.8%
у остальных, покрытие ≈ 64% «трудных» запросов подзадачи B. Это «мягкий»
аналог hard-rule по `pil1mtrx_offer`, но для подзадачи B.

Главный признак — `is_best_both` — трёхуровневый композит:
`limit == req_loan_amount` И `term == req_term` И минимальный `variant_no`
среди совпадающих. GBDT не строит такой композит дёшево из непрерывных
разниц (`req_minus_limit`, `req_minus_term` и group-рангов), поэтому его
нужно подавать явной фичей. Эффект значим в полном ансамбле, а не в
одиночной модели.

Все признаки детерминированы из pre-decision атрибутов оффера/заявки и
`variant_no` — не зависят от таргета (не лик) и не дрейфуют между train и
test (offer-rate `is_best_both` train-B ≈ 0.0236 ↔ test-B ≈ 0.0248).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alfa_cred.config import REQUEST_ID, VARIANT_ID

# Колонки, по которым проверяется точное соответствие оффера заявке.
MATCH_PAIRS = (("limit", "req_loan_amount"), ("term", "req_term"))
# Колонки для Парето-доминирования (меньше rate, больше limit, меньше ncl — лучше).
PARETO_COLUMNS = ("rate", "limit", "ncl")


def add_match_features(df: pd.DataFrame, request_col: str = REQUEST_ID) -> pd.DataFrame:
    """Признаки точного соответствия оффера заявке и «лучшего из совпадающих».

    Создаёт:
    - `lim_match`, `term_match`, `both_match` — точные равенства сумма/срок;
    - `dlim_abs`, `dterm_abs` — модуль отклонения предложенного от запрошенного;
    - `n_both` — число ask-matching офферов в запросе;
    - `is_uniq_both` — единственный ли ask-matching оффер в запросе;
    - `is_best_both` — ГЛАВНЫЙ: ask-matching оффер с минимальным `variant_no`;
    - `min_vn_both` — минимальный `variant_no` среди совпадающих в запросе;
    - `vn_rank_in_both` — ранг `variant_no` внутри подмножества совпадающих.

    Дизамбигуация multi-both по eva/rate проверена и не работает
    (`variant_no` — уже оптимальный дизамбигуатор), поэтому здесь её нет.
    """
    required = {"limit", "req_loan_amount", "term", "req_term", VARIANT_ID}
    if required - set(df.columns):
        return df

    df["lim_match"] = (df["limit"] == df["req_loan_amount"]).astype("int8")
    df["term_match"] = (df["term"] == df["req_term"]).astype("int8")
    df["both_match"] = (df["lim_match"] * df["term_match"]).astype("int8")
    df["dlim_abs"] = (df["limit"] - df["req_loan_amount"]).abs().astype("float32")
    df["dterm_abs"] = (df["term"] - df["req_term"]).abs().astype("float32")

    grouped = df.groupby(request_col, sort=False)
    df["n_both"] = grouped["both_match"].transform("sum").astype("int16")
    df["is_uniq_both"] = ((df["n_both"] == 1) & (df["both_match"] == 1)).astype("int8")

    # `variant_no` только среди ask-matching офферов; у несовпадающих — +inf,
    # чтобы они не попадали в минимум и в ранги внутри both-подмножества.
    df["_vn_both"] = np.where(
        df["both_match"].to_numpy() == 1, df[VARIANT_ID].to_numpy(), np.inf
    )
    grouped_vn = df.groupby(request_col, sort=False)["_vn_both"]
    min_vn = grouped_vn.transform("min")
    df["is_best_both"] = (
        (df["both_match"] == 1) & (df[VARIANT_ID] == min_vn)
    ).astype("int8")
    df["vn_rank_in_both"] = np.where(
        df["both_match"].to_numpy() == 1,
        grouped_vn.rank(method="first").to_numpy(),
        0.0,
    ).astype("float32")
    df["min_vn_both"] = min_vn.replace(np.inf, 0).astype("float32")
    df.drop(columns=["_vn_both"], inplace=True)
    return df


def add_pareto_features(df: pd.DataFrame, request_col: str = REQUEST_ID) -> pd.DataFrame:
    """Признаки Парето-доминирования и относительные отношения к экстремумам.

    Добавляет только то, чего нет во внутригрупповых рангах (`features/group.py`):
    - `rate_ratio_to_min`, `limit_ratio_to_max` — отношения к экстремуму группы;
    - `pareto_dominated_cnt` — сколько офферов запроса доминируют данный
      (ниже rate, выше limit, ниже ncl при хотя бы одном строгом неравенстве);
    - `is_pareto_optimal` — флаг недоминируемого оффера.

    Дублирующие gap-признаки (`*_gap_to_min`, `*_diff_to_max`, `rel_position`)
    здесь намеренно не считаются — они уже есть в `features/group.py`.
    """
    if not set(PARETO_COLUMNS).issubset(df.columns):
        return df

    grouped = df.groupby(request_col, sort=False)
    df["rate_ratio_to_min"] = (
        df["rate"] / (grouped["rate"].transform("min") + 1e-6)
    ).astype("float32")
    df["limit_ratio_to_max"] = (
        df["limit"] / (grouped["limit"].transform("max") + 1e-6)
    ).astype("float32")

    def _dominated_count(sub: pd.DataFrame) -> pd.Series:
        rate = sub["rate"].to_numpy()
        limit = sub["limit"].to_numpy()
        ncl = sub["ncl"].to_numpy()
        weak = (
            (rate[None, :] <= rate[:, None])
            & (limit[None, :] >= limit[:, None])
            & (ncl[None, :] <= ncl[:, None])
        )
        strict = (
            (rate[None, :] < rate[:, None])
            | (limit[None, :] > limit[:, None])
            | (ncl[None, :] < ncl[:, None])
        )
        dominated = weak & strict
        np.fill_diagonal(dominated, False)
        return pd.Series(dominated.sum(axis=1), index=sub.index)

    df["pareto_dominated_cnt"] = (
        grouped[list(PARETO_COLUMNS)]
        .apply(_dominated_count)
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    df["is_pareto_optimal"] = (df["pareto_dominated_cnt"] == 0).astype("int8")
    return df
