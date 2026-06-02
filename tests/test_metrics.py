"""Тесты корректности метрики NDCG@5 — ядро оценки ранжирования (метрика хакатона).

Реализация `ndcg_at_k`/`mean_ndcg_at_5` адаптирована из бейзлайна организаторов;
по ней принимаются все решения, поэтому её корректность проверяем явно.
"""

import math

import pandas as pd
import pytest

from alfa_cred.metrics import mean_ndcg_at_5, ndcg_at_k


def test_perfect_ranking_is_one():
    assert ndcg_at_k([1, 0, 0, 0, 0], k=5) == pytest.approx(1.0)


def test_single_positive_second_place():
    # DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert ndcg_at_k([0, 1, 0, 0, 0], k=5) == pytest.approx(1.0 / math.log2(3))


def test_no_positive_is_nan():
    assert math.isnan(ndcg_at_k([0, 0, 0, 0, 0], k=5))


def test_positive_beyond_k_is_zero():
    # позитив на 6-й позиции: в топ-5 релевантных нет, но IDCG>0 -> 0.0
    assert ndcg_at_k([0, 0, 0, 0, 0, 1], k=5) == pytest.approx(0.0)


def test_two_positives_suboptimal_order():
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)   # позиции 0 и 2
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # идеал: позиции 0 и 1
    assert ndcg_at_k([1, 0, 1, 0, 0], k=5) == pytest.approx(dcg / idcg)


def test_mean_ndcg_ignores_groups_without_positive():
    # r1: позитив с максимальным score -> 1.0; r2: без позитива -> NaN -> игнор
    df = pd.DataFrame({
        "request_id": ["r1", "r1", "r1", "r2", "r2"],
        "score": [0.9, 0.5, 0.1, 0.8, 0.2],
        "is_deal": [1, 0, 0, 0, 0],
    })
    assert mean_ndcg_at_5(df) == pytest.approx(1.0)


def test_mean_ndcg_sorts_by_score_descending():
    # позитив имеет НЕ самый высокий score -> NDCG@5 < 1
    df = pd.DataFrame({
        "request_id": ["r1", "r1", "r1"],
        "score": [0.9, 0.5, 0.1],
        "is_deal": [0, 1, 0],
    })
    assert mean_ndcg_at_5(df) == pytest.approx(1.0 / math.log2(3))
