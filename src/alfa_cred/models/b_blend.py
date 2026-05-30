"""B-бленд для подзадачи B (заявки без pil1mtrx-оффера).

8 моделей: LightGBM LambdaRank ×3 + XGBoost rank:ndcg ×3 + CatBoost YetiRank ×2.
Каждая обучается один раз на всех train-B заявках, предсказывает test-B, скоры
переводятся в перцентильные ранги внутри `request_id` и усредняются (rank-avg
устойчив к разным шкалам моделей). Сиды и гиперпараметры зафиксированы.

API разделён на обучение/сохранение и загрузку/инференс (`fit_b_models`/
`save_b_models`/`load_b_models`/`predict_b_blend`) — для раздельных режимов
train/inference. `build_b_blend` — обучить-и-сразу-предсказать (all-in-one).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRanker, Pool

from alfa_cred.config import REQUEST_ID, TARGET

# Сиды по типам моделей (зафиксированы).
LGB_B_SEEDS = (42, 123, 7)
XGB_B_SEEDS = (42, 137, 314)
CB_B_SEEDS = (42, 777)
CB_N_ITER = 500

LGB_B_PARAMS = dict(
    objective="lambdarank", metric="ndcg", n_estimators=500, learning_rate=0.03,
    num_leaves=31, min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=5, lambdarank_truncation_level=5, lambda_l2=1.0, n_jobs=-1, verbose=-1,
)
XGB_B_PARAMS = dict(
    objective="rank:ndcg", eval_metric="ndcg@5", tree_method="hist",
    n_estimators=500, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1,
)
CB_B_PARAMS = dict(
    loss_function="YetiRank", learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    eval_metric="NDCG:top=5;type=Exp", verbose=0, thread_count=-1,
)


def _pct_rank(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Перцентильные ранги скоров внутри каждого `request_id`."""
    return pd.Series(scores, index=df.index).groupby(df[REQUEST_ID].values).rank(pct=True).values


def _xgb_frame(df: pd.DataFrame, feature_cols, cat_cols) -> pd.DataFrame:
    """Для XGBoost категориальные колонки переводим в целочисленные коды."""
    d = df[feature_cols].copy()
    for c in cat_cols:
        if str(d[c].dtype) == "category":
            d[c] = d[c].cat.codes.astype("int32")
    return d


def _cb_frame(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    """Для CatBoost категориальные колонки переводим в строки с заполнением NA."""
    d = df.copy()
    for c in cat_cols:
        d[c] = d[c].astype("string").fillna("__NA__")
    return d


def _fit_lgb(train_b, feature_cols, cat_cols, seed):
    t = train_b.sort_values(REQUEST_ID)
    group = t.groupby(REQUEST_ID, sort=False).size().values
    model = lgb.LGBMRanker(random_state=seed, **LGB_B_PARAMS)
    model.fit(t[feature_cols], t[TARGET].astype(int), group=group, categorical_feature=cat_cols)
    return model.booster_


def _fit_xgb(train_b, feature_cols, cat_cols, seed):
    t = train_b.sort_values(REQUEST_ID)
    qid = pd.factorize(t[REQUEST_ID])[0].astype("int32")  # подряд после сортировки
    model = xgb.XGBRanker(random_state=seed, **XGB_B_PARAMS)
    model.fit(_xgb_frame(t, feature_cols, cat_cols), t[TARGET].astype(int), qid=qid)
    return model


def _fit_cb(train_b, feature_cols, cat_cols, seed, n_iter):
    t = _cb_frame(train_b, cat_cols).sort_values(REQUEST_ID)
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    pool = Pool(t[feature_cols], label=t[TARGET].astype(int).values,
                group_id=t[REQUEST_ID].values, cat_features=cat_idx)
    model = CatBoostRanker(iterations=n_iter, random_seed=seed, **CB_B_PARAMS)
    model.fit(pool)
    return model


def _predict_one(kind: str, model, feature_cols, cat_cols, predict_df) -> np.ndarray:
    if kind == "lgb":
        return model.predict(predict_df[feature_cols])
    if kind == "xgb":
        return model.predict(_xgb_frame(predict_df, feature_cols, cat_cols))
    return model.predict(_cb_frame(predict_df, cat_cols)[feature_cols])


def fit_b_models(train_b, feature_cols, cat_cols, n_iter_cb: int = CB_N_ITER) -> list[tuple[str, object]]:
    """Обучает 8 B-моделей. Возвращает список `(kind, model)`, kind ∈ {lgb, xgb, cb}."""
    models: list[tuple[str, object]] = []
    for s in LGB_B_SEEDS:
        models.append(("lgb", _fit_lgb(train_b, feature_cols, cat_cols, s)))
    for s in XGB_B_SEEDS:
        models.append(("xgb", _fit_xgb(train_b, feature_cols, cat_cols, s)))
    for s in CB_B_SEEDS:
        models.append(("cb", _fit_cb(train_b, feature_cols, cat_cols, s, n_iter_cb)))
    return models


def predict_b_blend(models, feature_cols, cat_cols, predict_df) -> np.ndarray:
    """Rank-avg перцентильных рангов 8 B-моделей для строк `predict_df`."""
    ranks = [_pct_rank(predict_df, _predict_one(k, m, feature_cols, cat_cols, predict_df)) for k, m in models]
    return np.mean(ranks, axis=0)


def save_b_models(models, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (kind, model) in enumerate(models):
        ext = {"lgb": "txt", "xgb": "json", "cb": "cbm"}[kind]
        path = out_dir / f"b_{i:02d}_{kind}.{ext}"
        model.save_model(str(path))
        manifest.append({"kind": kind, "file": path.name})
    (out_dir / "b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def load_b_models(in_dir: Path) -> list[tuple[str, object]]:
    in_dir = Path(in_dir)
    manifest = json.loads((in_dir / "b_manifest.json").read_text(encoding="utf-8"))
    models: list[tuple[str, object]] = []
    for entry in manifest:
        path = in_dir / entry["file"]
        if entry["kind"] == "lgb":
            models.append(("lgb", lgb.Booster(model_file=str(path))))
        elif entry["kind"] == "xgb":
            m = xgb.XGBRanker()
            m.load_model(str(path))
            models.append(("xgb", m))
        else:
            m = CatBoostRanker()
            m.load_model(str(path))
            models.append(("cb", m))
    return models


def build_b_blend(train_b, feature_cols, cat_cols, predict_df, n_iter_cb=CB_N_ITER) -> np.ndarray:
    """Обучить и сразу предсказать (all-in-one): rank-avg B-бленд для `predict_df`."""
    models = fit_b_models(train_b, feature_cols, cat_cols, n_iter_cb)
    return predict_b_blend(models, feature_cols, cat_cols, predict_df)
