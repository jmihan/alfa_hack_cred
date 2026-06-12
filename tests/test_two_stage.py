"""Тесты сборки two-stage сабмита: перцентильные ранги и склейка A/B + hard-rule."""

import numpy as np
import pandas as pd
import pytest

from alfa_cred import two_stage
from alfa_cred.two_stage import assemble_submission, pct_rank


def test_pct_rank_within_request():
    df = pd.DataFrame({"request_id": ["r1", "r1", "r1", "r2", "r2"]})
    scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    ranks = pct_rank(df, scores)
    # r1: 3 оффера -> 1/3, 2/3, 3/3 ; r2: 2 оффера -> 1/2, 2/2
    np.testing.assert_allclose(ranks, [1 / 3, 2 / 3, 1.0, 0.5, 1.0])


def test_assemble_two_stage_hard_rule_and_b_override(tmp_path, monkeypatch):
    # verify_submission читает data/commit.csv — в тесте данных нет, отключаем сверку.
    monkeypatch.setattr(two_stage, "verify_submission", lambda *a, **k: None)

    # A1 — есть pil1 (подзадача A), B1 — нет pil1 (подзадача B).
    a_keys = pd.DataFrame({
        "request_id":     ["A1", "A1", "B1", "B1"],
        "variant_no":     [1,    2,    1,    2],
        "pil1mtrx_offer": [1,    0,    0,    0],
    })
    a_pct = np.array([0.50, 0.50, 0.90, 0.10])           # для B-строк будет перекрыт
    b_keys_score = pd.DataFrame({
        "request_id": ["B1", "B1"],
        "variant_no": [1, 2],
        "b_score":    [0.30, 0.80],
    })

    out = tmp_path / "sub.csv"
    assemble_submission(a_keys, a_pct, b_keys_score, out)
    res = pd.read_csv(out, sep=";").set_index(["request_id", "variant_no"])["score"]

    # A1: оффер с pil1=1 получает +1.0 (hard-rule) и встаёт выше
    assert res.loc[("A1", 1)] == pytest.approx(1.5)   # 0.5 + 1.0
    assert res.loc[("A1", 2)] == pytest.approx(0.5)
    assert res.loc[("A1", 1)] > res.loc[("A1", 2)]
    # B1: скор берётся из b_score (a_pct проигнорирован)
    assert res.loc[("B1", 1)] == pytest.approx(0.30)
    assert res.loc[("B1", 2)] == pytest.approx(0.80)
