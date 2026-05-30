"""Two-stage пайплайн: подготовка признаков и сборка сабмита (train и inference).

Подготовка признаков (A — расширенный набор, B — широкий) и сборка финального
сабмита из готовых компонентов. Сами модели обучаются (`scripts/fit_pipeline.py`)
или загружаются (`scripts/predict.py`) — здесь только feature engineering и
two-stage сборка, общие для обоих режимов.

Архитектура:
- A (есть pil1-оффер): rank-avg 5-модельного A-бленда + hard-rule (pil1 → верх).
- B (нет pil1): 0.70·b_blend + 0.30·pointwise-MLP (перцентильные ранги).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch импортируем ПЕРВЫМ (до lightgbm/catboost через model-модули): иначе на
# Windows c10.dll падает с WinError 1114 при уже загруженном OpenMP.
import torch  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alfa_cred.config import REQUEST_ID, SAMPLE_SUBMISSION_PATH, VARIANT_ID  # noqa: E402
from alfa_cred.features.pipeline import (  # noqa: E402
    build_feature_table,
    build_wide_feature_table,
    feature_columns,
)
from alfa_cred.inference import make_submission, verify_submission  # noqa: E402
from alfa_cred.io_utils import encode_categoricals_inplace, load_raw, sort_by_request  # noqa: E402

PIL_COL = "pil1mtrx_offer"
B_BLEND_WEIGHT = 0.70  # вес b_blend в B-стороне; 0.30 — на pointwise-MLP


def pct_rank(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores, index=df.index).groupby(df[REQUEST_ID].values).rank(pct=True).values


def prepare_a_features():
    """Расширенный набор для A-бленда. Возвращает (train_sorted, test_sorted, fc, cat)."""
    train_raw, test_raw, feats = load_raw()
    train = build_feature_table(train_raw, feats, is_train=True)
    test = build_feature_table(test_raw, feats, is_train=False)
    fc, cat = feature_columns(train)
    for c in fc:
        if c not in test.columns:
            test[c] = 0
    train, test = encode_categoricals_inplace(train, test, cat)
    return sort_by_request(train), sort_by_request(test), fc, cat


def prepare_b_features():
    """Широкий offer-набор. Возвращает (train_b, test_sorted, is_b, test_b, fc, cat)."""
    train, test, fc, cat = build_wide_feature_table(*load_raw())
    cat = [c for c in cat if c in fc]
    for d in (train, test):
        d["req_has_pil1"] = d.groupby(REQUEST_ID, sort=False)[PIL_COL].transform("max").astype("int8")
    train_b = train[train["req_has_pil1"] == 0]
    test_sorted = sort_by_request(test)
    is_b = (test_sorted["req_has_pil1"] == 0).to_numpy()
    test_b = test_sorted[is_b].reset_index(drop=True)
    return train_b, test_sorted, is_b, test_b, fc, cat


def assemble_submission(a_keys: pd.DataFrame, a_pct: np.ndarray, b_keys_score: pd.DataFrame, out_path) -> None:
    """Two-stage сборка: A = a_pct + hard-rule, B = b_keys_score; round(6) + verify."""
    base = a_keys.copy()
    base[REQUEST_ID] = base[REQUEST_ID].astype(str)
    base[VARIANT_ID] = base[VARIANT_ID].astype("int32")
    base["req_has_pil1"] = base.groupby(REQUEST_ID, sort=False)[PIL_COL].transform("max").astype("int8")

    b_keys_score = b_keys_score.copy()
    b_keys_score[REQUEST_ID] = b_keys_score[REQUEST_ID].astype(str)
    b_keys_score[VARIANT_ID] = b_keys_score[VARIANT_ID].astype("int32")
    b_map = base.merge(b_keys_score, on=[REQUEST_ID, VARIANT_ID], how="left")["b_score"].to_numpy()

    is_b = (base["req_has_pil1"] == 0).to_numpy()
    score = a_pct.copy()
    score[is_b] = b_map[is_b]
    score = score + 1.0 * base[PIL_COL].astype(float).to_numpy()  # hard-rule: pil1 → верх

    out = base[[REQUEST_ID, VARIANT_ID]].copy()
    make_submission(out, np.round(score, 6), out_path)
    verify_submission(out_path, SAMPLE_SUBMISSION_PATH)
