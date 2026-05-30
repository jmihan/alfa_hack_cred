"""B-бленд для подзадачи B (заявки без pil1mtrx-оффера).

8 моделей: LightGBM LambdaRank ×3 + XGBoost rank:ndcg ×3 + CatBoost YetiRank ×2.
Каждая обучается один раз на всех train-B заявках, предсказывает test-B, скоры
переводятся в перцентильные ранги внутри `request_id` и усредняются (rank-avg
устойчив к разным шкалам моделей). Сиды и гиперпараметры зафиксированы.
"""

from __future__ import annotations

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


def _fit_lgb(train_b, feature_cols, cat_cols, seed, predict_df) -> np.ndarray:
    t = train_b.sort_values(REQUEST_ID)
    group = t.groupby(REQUEST_ID, sort=False).size().values
    model = lgb.LGBMRanker(random_state=seed, **LGB_B_PARAMS)
    model.fit(t[feature_cols], t[TARGET].astype(int), group=group, categorical_feature=cat_cols)
    return model.predict(predict_df[feature_cols])


def _fit_xgb(train_b, feature_cols, cat_cols, seed, predict_df) -> np.ndarray:
    t = train_b.sort_values(REQUEST_ID)
    qid = pd.factorize(t[REQUEST_ID])[0].astype("int32")  # подряд после сортировки
    model = xgb.XGBRanker(random_state=seed, **XGB_B_PARAMS)
    model.fit(_xgb_frame(t, feature_cols, cat_cols), t[TARGET].astype(int), qid=qid)
    return model.predict(_xgb_frame(predict_df, feature_cols, cat_cols))


def _fit_cb(train_b, feature_cols, cat_cols, seed, predict_df) -> np.ndarray:
    def prep(df):
        d = df.copy()
        for c in cat_cols:
            d[c] = d[c].astype("string").fillna("__NA__")
        return d

    t = prep(train_b).sort_values(REQUEST_ID)
    pr = prep(predict_df)
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    pool = Pool(t[feature_cols], label=t[TARGET].astype(int).values,
                group_id=t[REQUEST_ID].values, cat_features=cat_idx)
    model = CatBoostRanker(iterations=CB_N_ITER, random_seed=seed, **CB_B_PARAMS)
    model.fit(pool)
    return model.predict(pr[feature_cols])


def build_b_blend(train_b: pd.DataFrame, feature_cols, cat_cols, predict_df: pd.DataFrame) -> np.ndarray:
    """Собирает rank-avg B-бленд: LGB×3 + XGB×3 + CB×2 → среднее перцентильных рангов."""
    ranks = []
    for s in LGB_B_SEEDS:
        ranks.append(_pct_rank(predict_df, _fit_lgb(train_b, feature_cols, cat_cols, s, predict_df)))
    for s in XGB_B_SEEDS:
        ranks.append(_pct_rank(predict_df, _fit_xgb(train_b, feature_cols, cat_cols, s, predict_df)))
    for s in CB_B_SEEDS:
        ranks.append(_pct_rank(predict_df, _fit_cb(train_b, feature_cols, cat_cols, s, predict_df)))
    return np.mean(ranks, axis=0)
