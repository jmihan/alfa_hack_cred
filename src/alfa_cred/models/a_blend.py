"""Компактный A-бленд для подзадачи A (заявки с pil1mtrx-оффером).

5 моделей на расширенном наборе (`build_feature_table`): 3×LightGBM LambdaRank
(tuned-параметры, сиды 42/123/777) + 2×CatBoost YetiRank (сиды 42/123). Скоры
усредняются как перцентильные ранги внутри `request_id`. Поверх в сборке сабмита
применяется hard-rule (pil1-оффер → верх).

API разделён на обучение/сохранение (`fit_a_models`/`save_a_models`) и
загрузку/инференс (`load_a_models`/`predict_a_blend`) — для раздельных режимов
train/inference. Категориальные ожидаются закодированными целочисленными кодами
(общими train+test), `train_sorted` — отсортированным по `request_id`.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from alfa_cred.config import REQUEST_ID, TARGET

LGB_A_PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[5], lambdarank_truncation_level=5,
    learning_rate=0.0138, num_leaves=374, min_data_in_leaf=388, feature_fraction=0.78,
    bagging_fraction=0.89, bagging_freq=5, lambda_l1=0.0005, lambda_l2=7.0e-05,
    min_gain_to_split=0.013, max_depth=0, verbose=-1, n_jobs=-1,
)
LGB_A_SEEDS = (42, 123, 777)
LGB_A_N_ESTIMATORS = 2500

CB_A_PARAMS = dict(
    loss_function="YetiRank", eval_metric="NDCG:top=5;type=Exp", learning_rate=0.05,
    depth=6, l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.85,
    verbose=0, allow_writing_files=False, thread_count=-1,
)
CB_A_SEEDS = (42, 123)
CB_A_N_ESTIMATORS = 1000


def _pct_rank(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores, index=df.index).groupby(df[REQUEST_ID].values).rank(pct=True).values


def fit_a_models(train_sorted: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> list[tuple[str, object]]:
    """Обучает 5 A-моделей. Возвращает список `(kind, model)`, kind ∈ {lgb, cb}."""
    group = train_sorted.groupby(REQUEST_ID, sort=False).size().to_numpy()
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    models: list[tuple[str, object]] = []
    for seed in LGB_A_SEEDS:
        m = lgb.LGBMRanker(random_state=seed, n_estimators=LGB_A_N_ESTIMATORS, **LGB_A_PARAMS)
        m.fit(train_sorted[feature_cols], train_sorted[TARGET].astype(int), group=group, categorical_feature=cat_cols)
        models.append(("lgb", m.booster_))
    pool = Pool(train_sorted[feature_cols], label=train_sorted[TARGET].astype(int).to_numpy(),
                group_id=train_sorted[REQUEST_ID].to_numpy(), cat_features=cat_idx)
    for seed in CB_A_SEEDS:
        m = CatBoostRanker(iterations=CB_A_N_ESTIMATORS, random_seed=seed, **CB_A_PARAMS)
        m.fit(pool)
        models.append(("cb", m))
    return models


def predict_a_blend(models: list[tuple[str, object]], feature_cols: list[str], predict_df: pd.DataFrame) -> np.ndarray:
    """Rank-avg перцентильных рангов 5 A-моделей для строк `predict_df`."""
    ranks = [_pct_rank(predict_df, model.predict(predict_df[feature_cols])) for _, model in models]
    return np.mean(ranks, axis=0)


def save_a_models(models: list[tuple[str, object]], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (kind, model) in enumerate(models):
        if kind == "lgb":
            path = out_dir / f"a_{i:02d}_lgb.txt"
            model.save_model(str(path))
        else:
            path = out_dir / f"a_{i:02d}_cb.cbm"
            model.save_model(str(path))
        manifest.append({"kind": kind, "file": path.name})
    (out_dir / "a_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def load_a_models(in_dir: Path) -> list[tuple[str, object]]:
    in_dir = Path(in_dir)
    manifest = json.loads((in_dir / "a_manifest.json").read_text(encoding="utf-8"))
    models: list[tuple[str, object]] = []
    for entry in manifest:
        path = in_dir / entry["file"]
        if entry["kind"] == "lgb":
            models.append(("lgb", lgb.Booster(model_file=str(path))))
        else:
            m = CatBoostRanker()
            m.load_model(str(path))
            models.append(("cb", m))
    return models


def build_a_blend(train_sorted: pd.DataFrame, feature_cols: list[str], cat_cols: list[str], predict_df: pd.DataFrame) -> np.ndarray:
    """Обучить и сразу предсказать (all-in-one): rank-avg A-бленд для `predict_df`."""
    return predict_a_blend(fit_a_models(train_sorted, feature_cols, cat_cols), feature_cols, predict_df)
