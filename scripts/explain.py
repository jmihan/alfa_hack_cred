"""Интерпретация модели подзадачи B: SHAP + group-aware permutation на NDCG@5.

Интерпретируется ИМЕННО модель B-стороны: одиночная LightGBM с теми же
гиперпараметрами (`LGB_PARAMS`), сидом (42) и расширенным набором (`build_feature_table`,
B-only), что и LightGBM-член bAllL (см. `models/b_ball.py`). Объясняем одну модель, а не
rank-avg 6 моделей (5 XGB + 1 LGBM), потому что SHAP TreeExplainer применим к одному
дереву; сигнал у всех членов бленда общий, поэтому одиночная модель репрезентативна.

Обучается на части B-заявок, объясняется на отложенных B-заявках:
- глобальная SHAP-важность (beeswarm/bar);
- локальные SHAP-объяснения нескольких офферов, ставших сделкой (waterfall);
- зависимость скора от топ-признака;
- permutation importance по падению NDCG@5 (group-aware).

Графики и полные таблицы пишутся в `reports/interpretation/` (gitignored),
сводка топ-признаков печатается в stdout.

Запуск:
    python scripts/explain.py
    python scripts/explain.py --shap-sample 20000 --perm-top 30 --perm-repeats 3
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from alfa_cred.config import PROJECT_ROOT, REQUEST_ID
from alfa_cred.interpret import (
    compute_shap,
    global_shap_importance,
    group_permutation_importance,
    pick_local_examples,
    train_reference_model,
)
from alfa_cred.two_stage import prepare_b_features
from alfa_cred.utils import get_logger

LOG = get_logger("explain")
REPORT_DIR = PROJECT_ROOT / "reports" / "interpretation"
# Признаки, которые всегда включаем в permutation (позиционные/ставочные приоры B).
ALWAYS_PERMUTE = ("rate", "rate_rank", "limit_rank", "term_rank", "ncl_rank")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP + permutation интерпретация B-модели")
    p.add_argument("--shap-sample", type=int, default=20000, help="Строк для SHAP")
    p.add_argument("--perm-top", type=int, default=30, help="Сколько топ-SHAP фич перемешивать")
    p.add_argument("--perm-repeats", type=int, default=3, help="Повторов перемешивания")
    p.add_argument("--valid-size", type=float, default=0.25, help="Доля B-заявок в valid")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _save_plots(explainer, shap_values, x_sample, local_idx, gimp, pimp) -> None:
    """Сохраняет графики интерпретации в REPORT_DIR.

    Сначала — гарантированные matplotlib-бары важностей (строятся прямо из
    посчитанных таблиц, не зависят от рисующих функций shap, поэтому всегда
    появляются). Затем — best-effort SHAP-графики (beeswarm/bar/dependence/
    waterfall): они зависят от версии shap и могут не нарисоваться, что не
    критично — ключевые важности уже сохранены барами выше.
    """
    # matplotlib в headless-режиме (Agg) + доступный кэш шрифтов (MPLCONFIGDIR),
    # иначе сохранение в контейнере падает.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def _fresh(filename: str) -> Path:
        # Удаляем старый файл перед записью: при bind-mount Docker Desktop перезапись
        # существующего бинарника иногда не синхронится на хост, а создание нового —
        # синхронится. Так PNG гарантированно обновляется (CSV-перезапись работает и так).
        path = REPORT_DIR / filename
        path.unlink(missing_ok=True)
        return path

    def _barh(df: pd.DataFrame, value_col: str, title: str, filename: str, color: str) -> None:
        try:
            top = df.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(top["feature"].astype(str), top[value_col].to_numpy(), color=color)
            ax.set_title(title)
            ax.set_xlabel(value_col)
            fig.tight_layout()
            fig.savefig(_fresh(filename), dpi=130, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("график %s не сохранён: %r", filename, exc)
        finally:
            plt.close("all")

    # --- Гарантированные графики важностей ---
    _barh(gimp, "mean_abs_shap", "Глобальная SHAP-важность (top-20)",
          "global_shap_importance.png", "#4C78A8")
    _barh(pimp, "ndcg5_drop_mean", "Permutation importance: падение NDCG@5 (top-20)",
          "permutation_importance_ndcg5.png", "#E45756")

    # --- Best-effort SHAP-графики ---
    def _save_shap(draw, filename: str) -> None:
        try:
            draw()
            plt.savefig(_fresh(filename), dpi=130, bbox_inches="tight")
            LOG.info("SHAP-график сохранён: %s", filename)
        except Exception as exc:  # noqa: BLE001 — зависят от версии shap, не критичны
            # Печатаем причину и в stdout (а не только в лог) — чтобы было видно,
            # почему конкретный shap-график не нарисовался (бары важностей уже сохранены).
            print(f"  [warn] SHAP-график {filename} не сохранён: {exc!r}", flush=True)
            LOG.warning("SHAP-график %s не сохранён: %r", filename, exc)
        finally:
            plt.close("all")

    import shap

    _save_shap(lambda: shap.summary_plot(shap_values, x_sample, show=False, max_display=20),
               "shap_beeswarm.png")
    _save_shap(lambda: shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False, max_display=20),
               "shap_bar.png")
    top_feat = str(gimp.iloc[0]["feature"])
    if top_feat in x_sample.columns:
        _save_shap(lambda: shap.dependence_plot(top_feat, shap_values, x_sample, show=False, interaction_index=None),
                   f"shap_dependence_{top_feat}.png")

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[0])
    for k, row in enumerate(local_idx):
        if row >= len(x_sample):
            continue
        _save_shap(lambda r=row: shap.plots._waterfall.waterfall_legacy(
            base_value, shap_values[r], x_sample.iloc[r], max_display=14, show=False),
            f"shap_waterfall_{k}.png")
    # Листинг того, что РЕАЛЬНО лежит в REPORT_DIR (взгляд из контейнера) — чтобы
    # отличить «не нарисовалось» от «не синхронилось на хост» по bind-mount.
    pngs = sorted(REPORT_DIR.glob("*.png"))
    LOG.info("Графиков в %s: %d", REPORT_DIR, len(pngs))
    for p in pngs:
        LOG.info("  %s — %d байт", p.name, p.stat().st_size)


def _print_table(title: str, df: pd.DataFrame, n: int = 20) -> None:
    print(f"\n=== {title} ===")
    print(df.head(n).to_string(index=False))


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_b, _ts, _is_b, _tb, feature_cols, cat_cols = prepare_b_features()
    LOG.info("train-B (расширенный набор, без ask-match): %d строк, %d заявок, %d фич",
             len(train_b), train_b[REQUEST_ID].nunique(), len(feature_cols))

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.valid_size, random_state=args.seed)
    tr_idx, va_idx = next(splitter.split(train_b, groups=train_b[REQUEST_ID]))
    fit_b, valid_b = train_b.iloc[tr_idx], train_b.iloc[va_idx].reset_index(drop=True)

    LOG.info("Интерпретирую B-модель: репрезентативная LightGBM (параметры bAllL, seed=%d) "
             "на %d B-заявках, %d фич", args.seed, fit_b[REQUEST_ID].nunique(), len(feature_cols))
    model = train_reference_model(fit_b, feature_cols, seed=args.seed)

    n_sample = min(args.shap_sample, len(valid_b))
    sample_rows = valid_b.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    x_sample = sample_rows[feature_cols]
    LOG.info("SHAP на %d строках", n_sample)
    explainer, shap_values = compute_shap(model, x_sample)

    gimp = global_shap_importance(shap_values, feature_cols)
    gimp.to_csv(REPORT_DIR / "shap_global_importance.csv", index=False)
    _print_table("Глобальная SHAP-важность (top-20)", gimp)

    perm_feats = list(dict.fromkeys(
        gimp.head(args.perm_top)["feature"].tolist() + [f for f in ALWAYS_PERMUTE if f in feature_cols]
    ))
    LOG.info("Permutation importance по NDCG@5 для %d признаков", len(perm_feats))
    base_ndcg, pimp = group_permutation_importance(
        model, valid_b, feature_cols, features=perm_feats,
        n_repeats=args.perm_repeats, seed=args.seed,
    )
    pimp.to_csv(REPORT_DIR / "permutation_importance_ndcg5.csv", index=False)
    print(f"\nBase B-NDCG@5 (valid, одиночная модель): {base_ndcg:.4f}")
    _print_table("Permutation importance по падению NDCG@5 (top-20)", pimp)

    # Локальные примеры берём из самого SHAP-сэмпла (`sample_rows`), чтобы
    # позиционные индексы совпадали с матрицей shap_values и x_sample.
    local_idx = pick_local_examples(sample_rows, n=3, seed=args.seed)
    try:
        _save_plots(explainer, shap_values, x_sample, local_idx, gimp, pimp)
    except Exception as exc:  # графики не критичны для выводов
        LOG.warning("Не удалось сохранить графики: %s", exc)

    print("\nГотово. Полные таблицы и графики — в", REPORT_DIR)


if __name__ == "__main__":
    main()
