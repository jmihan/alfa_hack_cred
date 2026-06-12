"""Интерпретация модели подзадачи B: SHAP + group-aware permutation на NDCG@5.

Зачем именно так (по итогам обзора методов для ranking-бустингов):

- **SHAP TreeExplainer** точен и быстр на деревьях (LightGBM/XGBoost/CatBoost) —
  считает аддитивные вклады в *скор оффера*, по которому и идёт сортировка внутри
  заявки. Глобальная важность = средний |SHAP|, локальная — разложение одного
  предсказания. Объясняем **одиночную** репрезентативную LightGBM-модель B-бленда
  (те же гиперпараметры), а не rank-avg бленд: TreeExplainer применим к одной
  модели, а сигнал у всех членов бленда один и тот же.
- **Group-aware permutation importance на NDCG@5** — модель-агностичная проверка:
  перемешиваем признак и смотрим падение NDCG@5 по B-заявкам. Устойчивее SHAP к
  коррелированным признакам (там SHAP может произвольно делить вклад между ними).

Память: SHAP считаем на сэмпле строк, не на всех train-B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alfa_cred.config import REQUEST_ID, TARGET
from alfa_cred.metrics import mean_ndcg_at_5
from alfa_cred.models.b_ball import LGB_PARAMS


def train_reference_model(train_b: pd.DataFrame, feature_cols: list[str], seed: int = 42):
    """Обучает одиночную LightGBM LambdaRank-модель на train-B (фич — как в бленде).

    Категориальные колонки ожидаются уже закодированными в целочисленные коды,
    чтобы SHAP TreeExplainer работал без проблем с pandas Categorical.
    """
    import lightgbm as lgb

    t = train_b.sort_values(REQUEST_ID)
    group = t.groupby(REQUEST_ID, sort=False).size().to_numpy()
    model = lgb.LGBMRanker(random_state=seed, n_estimators=500, **LGB_PARAMS)
    model.fit(t[feature_cols], t[TARGET].astype(int), group=group)
    return model


def compute_shap(model, x_sample: pd.DataFrame):
    """Возвращает `(explainer, shap_values)` для сэмпла строк (TreeExplainer)."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    return explainer, shap_values


def global_shap_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Глобальная важность: средний модуль SHAP по сэмплу, по убыванию."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _ndcg5_from_arrays(request_ids: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> float:
    return mean_ndcg_at_5(
        pd.DataFrame({REQUEST_ID: request_ids, "score": scores, TARGET: targets}),
        score_col="score",
        target_col=TARGET,
    )


def group_permutation_importance(
    model,
    valid_b: pd.DataFrame,
    feature_cols: list[str],
    features: list[str] | None = None,
    n_repeats: int = 3,
    seed: int = 42,
) -> tuple[float, pd.DataFrame]:
    """Падение NDCG@5 на B-заявках при перемешивании каждого признака.

    Возвращает `(base_ndcg5, df)`, где `df` отсортирован по среднему падению.
    Если `features=None`, перемешиваются все `feature_cols`.
    """
    rng = np.random.default_rng(seed)
    rid = valid_b[REQUEST_ID].to_numpy()
    y = valid_b[TARGET].astype(int).to_numpy()
    x = valid_b[feature_cols].reset_index(drop=True).copy()

    base = _ndcg5_from_arrays(rid, model.predict(x), y)

    rows = []
    for feat in (features or feature_cols):
        original = x[feat].to_numpy(copy=True)
        drops = []
        for _ in range(n_repeats):
            x[feat] = rng.permutation(original)
            score = _ndcg5_from_arrays(rid, model.predict(x), y)
            drops.append(base - score)
        x[feat] = original  # восстанавливаем колонку in-place
        rows.append({
            "feature": feat,
            "ndcg5_drop_mean": float(np.mean(drops)),
            "ndcg5_drop_std": float(np.std(drops)),
        })
    df = pd.DataFrame(rows).sort_values("ndcg5_drop_mean", ascending=False).reset_index(drop=True)
    return base, df


def pick_local_examples(valid_b: pd.DataFrame, n: int = 3, seed: int = 42) -> list[int]:
    """Позиционные индексы нескольких показательных B-офферов для локальных объяснений.

    Берём офферы, ставшие сделкой (`target=1`) — на них видно, как факторы тянут
    скор вверх.
    """
    v = valid_b.reset_index(drop=True)
    if TARGET in v.columns:
        idx = v.index[v[TARGET] == 1].to_numpy()
        if len(idx):
            rng = np.random.default_rng(seed)
            return list(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return list(range(min(n, len(v))))
