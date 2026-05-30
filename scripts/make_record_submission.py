"""Сборка рекордного two-stage сабмита (LB ≈ 92.18).

Архитектура:
- Подзадача A (есть `pil1mtrx_offer=1`): rank-avg blend record_11 + hard-rule
  (pil1-оффер на первое место).
- Подзадача B (нет pil1-оффера): выделенный 8-модельный B-бленд
  (LGB×3 + XGB×3 + CB×2) на широком offer-наборе (`features/offer_v3.py`),
  обученный ТОЛЬКО на B-заявках.

Главный сигнал B — `is_best_both` (ask-matching оффер с минимальным variant_no).

Запуск:
    python scripts/make_record_submission.py
    python scripts/make_record_submission.py --out submissions/record_submission.csv

Скрипт обучает B-бленд с нуля (~15-20 мин) и собирает сабмит. A-сторона берётся
из сохранённых `oof/<record_11>_test_scores.parquet` (rank-avg) + hard-rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alfa_cred.blends import RECORD_11_MODELS_LB_91_9668
from alfa_cred.config import (
    OOF_DIR,
    REQUEST_ID,
    SUBMISSIONS_DIR,
    VARIANT_ID,
)
from alfa_cred.features.offer_v3 import build_record_feature_table
from alfa_cred.inference import build_two_stage_submission
from alfa_cred.io_utils import load_raw, sort_by_request
from alfa_cred.models.record_b_blend import build_b_blend
from alfa_cred.utils import get_logger

LOG = get_logger("make_record")
PIL_COL = "pil1mtrx_offer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Рекордный two-stage сабмит (LB ≈ 92.18)")
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "record_submission.csv")
    return p.parse_args()


def _resolve_record_paths() -> list[Path]:
    paths = [OOF_DIR / f"{m}_test_scores.parquet" for m in RECORD_11_MODELS_LB_91_9668]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Отсутствуют test_scores record_11 (A-сторона):\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    return paths


def main() -> None:
    args = parse_args()

    train, test, feats = load_raw()
    train, test, feature_cols, cat_cols = build_record_feature_table(train, test, feats)
    cat_cols = [c for c in cat_cols if c in feature_cols]
    LOG.info("Фич: %d (категориальных %d)", len(feature_cols), len(cat_cols))

    for d in (train, test):
        d["req_has_pil1"] = d.groupby(REQUEST_ID, sort=False)[PIL_COL].transform("max").astype("int8")

    train_b = train[train["req_has_pil1"] == 0]
    test_sorted = sort_by_request(test)
    is_b = (test_sorted["req_has_pil1"] == 0).to_numpy()
    test_b = test_sorted[is_b]
    LOG.info("B-train: %d заявок -> B-test: %d заявок",
             train_b[REQUEST_ID].nunique(), test_b[REQUEST_ID].nunique())

    # sanity (не влияет на сабмит): deal-rate is_best_both на train-B ≈ 0.52
    dr = train_b.groupby("is_best_both")["is_deal"].mean().round(4).to_dict()
    LOG.info("deal-rate is_best_both (train-B): %s", dr)

    b_scores = build_b_blend(train_b, feature_cols, cat_cols, test_b)
    LOG.info("B-бленд готов (8 моделей, rank-avg)")

    # B-бленд для всех строк test: B-строки = бленд, A-строки = 0.5 (всё равно
    # перекрываются record-стороной в two-stage). Сохраняем как test_scores.
    b_full = pd.DataFrame({
        REQUEST_ID: test_sorted[REQUEST_ID].astype(str).values,
        VARIANT_ID: test_sorted[VARIANT_ID].astype("int32").values,
        "score_raw": np.full(len(test_sorted), 0.5, dtype=np.float64),
        PIL_COL: test_sorted[PIL_COL].astype("int8").values,
    })
    b_full.loc[is_b, "score_raw"] = b_scores
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    b_path = OOF_DIR / "record_b_blend_test_scores.parquet"
    b_full.to_parquet(b_path, index=False)

    build_two_stage_submission(
        b_test_paths=[b_path],
        record_test_paths=_resolve_record_paths(),
        test_sorted=test_sorted,
        out_path=args.out,
        pil_col=PIL_COL,
    )
    LOG.info("Рекордный сабмит сохранён: %s", args.out)


if __name__ == "__main__":
    main()
