"""Интерпретация модели подзадачи B: SHAP + group-aware permutation на NDCG@5.

Строит широкий offer-набор, обучает одиночную репрезентативную LightGBM-модель
B-бленда на части B-заявок, объясняет её на отложенных B-заявках:
- глобальная SHAP-важность (beeswarm/bar);
- локальные SHAP-объяснения нескольких `is_best_both`-офферов (waterfall);
- зависимость скора от `is_best_both`;
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

from alfa_cred.config import PROJECT_ROOT, REQUEST_ID, TARGET
from alfa_cred.features.pipeline import build_wide_feature_table
from alfa_cred.interpret import (
    compute_shap,
    global_shap_importance,
    group_permutation_importance,
    pick_local_examples,
    train_reference_model,
)
from alfa_cred.io_utils import load_raw
from alfa_cred.utils import get_logger

LOG = get_logger("explain")
PIL_COL = "pil1mtrx_offer"
REPORT_DIR = PROJECT_ROOT / "reports" / "interpretation"
# Признаки, которые всегда включаем в permutation (ядро гипотезы про подзадачу B).
ALWAYS_PERMUTE = ("is_best_both", "variant_no_norm", "variant_no_inv", "rate", "rate_rank")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP + permutation интерпретация B-модели")
    p.add_argument("--shap-sample", type=int, default=20000, help="Строк для SHAP")
    p.add_argument("--perm-top", type=int, default=30, help="Сколько топ-SHAP фич перемешивать")
    p.add_argument("--perm-repeats", type=int, default=3, help="Повторов перемешивания")
    p.add_argument("--valid-size", type=float, default=0.25, help="Доля B-заявок в valid")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _encode_categoricals_to_codes(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Categorical → целочисленные коды (нужно для SHAP TreeExplainer)."""
    for c in cat_cols:
        if str(df[c].dtype) == "category":
            df[c] = df[c].cat.codes.astype("int32")
        elif df[c].dtype == object:
            df[c] = df[c].astype("category").cat.codes.astype("int32")
    return df


def _save_plots(explainer, shap_values, x_sample, local_idx) -> None:
    # MPLCONFIGDIR в /tmp — чтобы matplotlib мог писать кэш шрифтов в контейнере.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def _save(draw, filename: str) -> None:
        """Рисует один график; сбой одного не должен ронять остальные."""
        try:
            draw()
            plt.tight_layout()
            plt.savefig(REPORT_DIR / filename, dpi=130, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001 — графики не критичны, логируем причину
            LOG.warning("график %s не сохранён: %r", filename, exc)
        finally:
            plt.close("all")

    _save(lambda: shap.summary_plot(shap_values, x_sample, show=False, max_display=20),
          "shap_beeswarm.png")
    _save(lambda: shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False, max_display=20),
          "shap_bar.png")
    if "is_best_both" in x_sample.columns:
        _save(lambda: shap.dependence_plot("is_best_both", shap_values, x_sample, show=False, interaction_index=None),
              "shap_dependence_is_best_both.png")

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[0])
    for k, row in enumerate(local_idx):
        if row >= len(x_sample):
            continue
        _save(lambda r=row: shap.plots._waterfall.waterfall_legacy(
            base_value, shap_values[r], x_sample.iloc[r], max_display=14, show=False),
            f"shap_waterfall_{k}.png")
    LOG.info("Графики сохранены в %s", REPORT_DIR)


def _print_table(title: str, df: pd.DataFrame, n: int = 20) -> None:
    print(f"\n=== {title} ===")
    print(df.head(n).to_string(index=False))


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train, _test, feature_cols, cat_cols = build_wide_feature_table(*load_raw())
    cat_cols = [c for c in cat_cols if c in feature_cols]
    train["req_has_pil1"] = train.groupby(REQUEST_ID, sort=False)[PIL_COL].transform("max").astype("int8")
    train_b = train[train["req_has_pil1"] == 0].reset_index(drop=True)
    train_b = _encode_categoricals_to_codes(train_b, cat_cols)
    LOG.info("train-B: %d строк, %d заявок, %d фич",
             len(train_b), train_b[REQUEST_ID].nunique(), len(feature_cols))

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.valid_size, random_state=args.seed)
    tr_idx, va_idx = next(splitter.split(train_b, groups=train_b[REQUEST_ID]))
    fit_b, valid_b = train_b.iloc[tr_idx], train_b.iloc[va_idx].reset_index(drop=True)

    LOG.info("Обучаю репрезентативную LightGBM-модель на %d B-заявках",
             fit_b[REQUEST_ID].nunique())
    model = train_reference_model(fit_b, feature_cols, seed=args.seed)

    n_sample = min(args.shap_sample, len(valid_b))
    sample_rows = valid_b.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    x_sample = sample_rows[feature_cols]
    LOG.info("SHAP на %d строках", n_sample)
    explainer, shap_values = compute_shap(model, x_sample)

    gimp = global_shap_importance(shap_values, feature_cols)
    gimp.to_csv(REPORT_DIR / "shap_global_importance.csv", index=False)
    _print_table("Глобальная SHAP-важность (top-20)", gimp)
    ibb = gimp.index[gimp.feature == "is_best_both"]
    if len(ibb):
        print(f"\nРанг `is_best_both` по SHAP: {int(ibb[0]) + 1} из {len(gimp)}")

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
        _save_plots(explainer, shap_values, x_sample, local_idx)
    except Exception as exc:  # графики не критичны для выводов
        LOG.warning("Не удалось сохранить графики: %s", exc)

    print("\nГотово. Полные таблицы и графики — в", REPORT_DIR)


if __name__ == "__main__":
    main()
