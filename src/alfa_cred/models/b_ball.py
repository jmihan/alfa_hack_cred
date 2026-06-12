"""B-сторона bAllL — лучший на приватном лидерборде ансамбль подзадачи B.

Rank-avg перцентильных рангов 6 GBDT на расширенном наборе (`build_feature_table`,
тот же, что у A-бленда, B-only) — точный рецепт исходного прогона (Pipeline L):
- 5× XGBoost (`rank:ndcg`, Optuna-параметры, `lambdarank_pair_method=topk`,
  `lambdarank_num_pair_per_sample=8`, сиды 42/137/314/8848/2026);
- 1× LightGBM extended (LambdaRank, tuned-параметры, сид 42).

Каждая модель: ранняя остановка по B-NDCG@5 на групповом холдауте (находим число
итераций), затем рефит на всём B-train этим числом — так модель детерминирована и
inference из сохранённой модели совпадает с train байт-в-байт. Скоры → перцентильные
ранги внутри `request_id` → усреднение.

Простой multi-seed XGB-ансамбль (без NN-диверсити) оказался устойчивее на приватном
лидерборде, чем сложный public-рекорд (bBalanced + MLP), переобучившийся на public.

API: `fit_b_ball`/`save_b_ball`/`load_b_ball`/`predict_b_ball` (+ `build_b_ball`).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from alfa_cred.config import REQUEST_ID, TARGET

# 5 XGBoost: одни Optuna-параметры (Pipeline L), разные сиды.
XGB_SEEDS = (42, 137, 314, 8848, 2026)
XGB_PARAMS = dict(
    objective="rank:ndcg", eval_metric="ndcg@5", tree_method="hist",
    learning_rate=0.032959026707838505, max_depth=7, min_child_weight=19.997035733974325,
    subsample=0.882350284611091, colsample_bytree=0.5337430740506179,
    reg_alpha=0.16373711956722492, reg_lambda=0.005858948220212961,
    lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=8, n_jobs=-1,
)
XGB_MAX_N = 4000
XGB_ES = 150

# 1 LightGBM extended (LGBM_TUNED_PARAMS из Pipeline K/L).
LGB_PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[5], lambdarank_truncation_level=5,
    learning_rate=0.0138, num_leaves=374, min_data_in_leaf=388, feature_fraction=0.78,
    bagging_fraction=0.89, bagging_freq=5, lambda_l1=0.0005, lambda_l2=7.0e-05,
    min_gain_to_split=0.013, max_depth=0, verbose=-1, n_jobs=-1,
)
LGB_SEED = 42
LGB_MAX_N = 4000
LGB_ES = 150

HOLDOUT_FRAC = 0.15
HOLDOUT_SEED = 42


def _pct_rank(df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Перцентильные ранги скоров внутри каждого `request_id`."""
    return pd.Series(scores, index=df.index).groupby(df[REQUEST_ID].values).rank(pct=True).values


def _holdout(train_b: pd.DataFrame):
    """Групповой холдаут (по request_id) для ранней остановки."""
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRAC, random_state=HOLDOUT_SEED)
    tr_idx, va_idx = next(gss.split(train_b, groups=train_b[REQUEST_ID]))
    return train_b.iloc[tr_idx], train_b.iloc[va_idx]


def _qid(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df[REQUEST_ID])[0].astype("int32")


def _xgb_device(device: str) -> dict:
    return {"device": "cuda"} if device == "cuda" else {}


def _fit_xgb(train_b, feature_cols, seed, device="cpu"):
    """ES на холдауте -> число итераций -> рефит на всём B-train (детерминированно)."""
    tr, va = _holdout(train_b)
    tr, va = tr.sort_values(REQUEST_ID), va.sort_values(REQUEST_ID)
    es = xgb.XGBRanker(random_state=seed, n_estimators=XGB_MAX_N, early_stopping_rounds=XGB_ES,
                       **XGB_PARAMS, **_xgb_device(device))
    es.fit(tr[feature_cols], tr[TARGET].astype(int), qid=_qid(tr),
           eval_set=[(va[feature_cols], va[TARGET].astype(int))], eval_qid=[_qid(va)], verbose=False)
    best_n = int(es.best_iteration) + 1
    full = train_b.sort_values(REQUEST_ID)
    model = xgb.XGBRanker(random_state=seed, n_estimators=best_n, **XGB_PARAMS, **_xgb_device(device))
    model.fit(full[feature_cols], full[TARGET].astype(int), qid=_qid(full))
    return model


def _fit_lgb(train_b, feature_cols, cat_cols):
    """ES на холдауте -> число итераций -> рефит на всём B-train. LightGBM на CPU."""
    tr, va = _holdout(train_b)
    tr, va = tr.sort_values(REQUEST_ID), va.sort_values(REQUEST_ID)
    es = lgb.LGBMRanker(random_state=LGB_SEED, n_estimators=LGB_MAX_N, **LGB_PARAMS)
    es.fit(tr[feature_cols], tr[TARGET].astype(int), group=tr.groupby(REQUEST_ID, sort=False).size().values,
           eval_set=[(va[feature_cols], va[TARGET].astype(int))],
           eval_group=[va.groupby(REQUEST_ID, sort=False).size().values],
           categorical_feature=cat_cols, callbacks=[lgb.early_stopping(LGB_ES, verbose=False)])
    best_n = int(es.best_iteration_)
    full = train_b.sort_values(REQUEST_ID)
    model = lgb.LGBMRanker(random_state=LGB_SEED, n_estimators=best_n, **LGB_PARAMS)
    model.fit(full[feature_cols], full[TARGET].astype(int),
              group=full.groupby(REQUEST_ID, sort=False).size().values, categorical_feature=cat_cols)
    return model.booster_


def _predict_one(kind: str, model, feature_cols, predict_df) -> np.ndarray:
    return model.predict(predict_df[feature_cols])


def fit_b_ball(train_b, feature_cols, cat_cols, device: str = "cpu") -> list[tuple[str, object]]:
    """Обучает 6 B-моделей (5 XGB + 1 LGBM). `device="cuda"` — XGBoost на GPU, LightGBM на CPU."""
    models: list[tuple[str, object]] = [("xgb", _fit_xgb(train_b, feature_cols, s, device)) for s in XGB_SEEDS]
    models.append(("lgb", _fit_lgb(train_b, feature_cols, cat_cols)))
    return models


def predict_b_ball(models, feature_cols, cat_cols, predict_df) -> np.ndarray:
    """Rank-avg перцентильных рангов 6 B-моделей для строк `predict_df`."""
    ranks = [_pct_rank(predict_df, _predict_one(k, m, feature_cols, predict_df)) for k, m in models]
    return np.mean(ranks, axis=0)


def save_b_ball(models, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (kind, model) in enumerate(models):
        ext = {"xgb": "json", "lgb": "txt"}[kind]
        path = out_dir / f"b_{i:02d}_{kind}.{ext}"
        model.save_model(str(path))
        manifest.append({"kind": kind, "file": path.name})
    (out_dir / "b_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def load_b_ball(in_dir: Path) -> list[tuple[str, object]]:
    in_dir = Path(in_dir)
    manifest = json.loads((in_dir / "b_manifest.json").read_text(encoding="utf-8"))
    models: list[tuple[str, object]] = []
    for entry in manifest:
        path = in_dir / entry["file"]
        if entry["kind"] == "xgb":
            m = xgb.XGBRanker()
            m.load_model(str(path))
            models.append(("xgb", m))
        else:
            models.append(("lgb", lgb.Booster(model_file=str(path))))
    return models


def build_b_ball(train_b, feature_cols, cat_cols, device: str = "cpu") -> list[tuple[str, object]]:
    """Обучить (all-in-one)."""
    return fit_b_ball(train_b, feature_cols, cat_cols, device)
