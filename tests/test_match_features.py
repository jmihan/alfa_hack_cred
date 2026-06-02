"""Тест ключевой фичи `is_best_both` (ask-matching) — главный сигнал подзадачи B.

`is_best_both` = оффер, точно совпавший с заявкой по сумме И сроку, и минимальный по
`variant_no` среди таких. Это «мягкий hard-rule» подзадачи B (deal-rate ≈52%),
поэтому его логику фиксируем тестом.
"""

import pandas as pd

from alfa_cred.features.match import add_match_features


def test_is_best_both_picks_min_variant_among_full_matches():
    df = pd.DataFrame({
        "request_id":      ["r1", "r1", "r1", "r2", "r2"],
        "variant_no":      [3,    1,    2,    1,    2],
        "limit":           [100,  100,  50,   100,  50],
        "req_loan_amount": [100,  100,  100,  100,  100],
        "term":            [12,   12,   12,   6,    12],
        "req_term":        [12,   12,   12,   12,   12],
    })
    res = add_match_features(df.copy()).set_index(["request_id", "variant_no"])

    # r1: vn=1 и vn=3 совпадают по сумме И сроку; vn=2 — нет (сумма 50 != 100)
    assert res.loc[("r1", 1), "both_match"] == 1
    assert res.loc[("r1", 3), "both_match"] == 1
    assert res.loc[("r1", 2), "both_match"] == 0

    # is_best_both только у минимального variant_no среди совпавших (vn=1)
    assert res.loc[("r1", 1), "is_best_both"] == 1
    assert res.loc[("r1", 3), "is_best_both"] == 0
    assert res.loc[("r1", 2), "is_best_both"] == 0

    # r2: ни один не совпал И по сумме, И по сроку -> is_best_both нигде
    assert res.loc[("r2", 1), "is_best_both"] == 0
    assert res.loc[("r2", 2), "is_best_both"] == 0


def test_is_best_both_unique_per_request_block():
    df = pd.DataFrame({
        "request_id":      ["r1", "r1", "r1", "r2", "r2"],
        "variant_no":      [3,    1,    2,    1,    2],
        "limit":           [100,  100,  50,   100,  50],
        "req_loan_amount": [100,  100,  100,  100,  100],
        "term":            [12,   12,   12,   6,    12],
        "req_term":        [12,   12,   12,   12,   12],
    })
    out = add_match_features(df.copy())
    # ровно один is_best_both на весь набор (только в r1)
    assert int(out["is_best_both"].sum()) == 1
    assert out.groupby("request_id")["is_best_both"].sum().max() <= 1
