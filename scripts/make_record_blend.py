"""Воспроизводит рекордный blend (LB = 91.9668) из сохранённых test_scores.

Запуск:
    python scripts/make_record_blend.py

Скрипт читает test_scores 11 моделей из `oof/` и собирает rank-averaging
blend с применением hard-rule для `pil1mtrx_offer=1`. На выходе CSV-сабмит
в `submissions/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alfa_cred.blends import RECORD_11_MODELS_LB_91_9668
from alfa_cred.config import (
    OOF_DIR,
    REQUEST_ID,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    VARIANT_ID,
)
from alfa_cred.inference import make_submission, verify_submission
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Воспроизведение рекордного blend")
    parser.add_argument(
        "--out", type=Path,
        default=SUBMISSIONS_DIR / f"record_blend_lb_91_9668_{datetime.now():%Y%m%d_%H%M}.csv",
        help="Куда сохранить сабмит",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_score_paths = [
        OOF_DIR / f"{run}_test_scores.parquet" for run in RECORD_11_MODELS_LB_91_9668
    ]
    missing = [p for p in test_score_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Отсутствуют test_scores для следующих моделей рекордного blend:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    base = pd.read_parquet(test_score_paths[0])[[REQUEST_ID, VARIANT_ID, "pil1mtrx_offer"]].copy()
    base[REQUEST_ID] = base[REQUEST_ID].astype(str)
    base[VARIANT_ID] = base[VARIANT_ID].astype("int32")
    rank_sum = np.zeros(len(base), dtype=np.float64)

    for path in test_score_paths:
        df = pd.read_parquet(path)
        df[REQUEST_ID] = df[REQUEST_ID].astype(str)
        df[VARIANT_ID] = df[VARIANT_ID].astype("int32")
        merged = base.merge(
            df[[REQUEST_ID, VARIANT_ID, "score_raw"]],
            on=[REQUEST_ID, VARIANT_ID], how="left",
        )
        if merged["score_raw"].isna().any():
            n_missing = int(merged["score_raw"].isna().sum())
            LOG.warning("Пропуски при merge с %s: %d строк", path.name, n_missing)
        ranks = merged.groupby(REQUEST_ID, sort=False)["score_raw"].rank(pct=True).fillna(0.5).to_numpy()
        rank_sum += ranks
        LOG.info("Учтена модель %s", path.stem.replace("_test_scores", ""))

    rank_avg = rank_sum / len(test_score_paths)
    # Hard-rule поверх перцентильных рангов: pil1mtrx=1 → score = rank_avg + 1.0
    final = rank_avg + 1.0 * base["pil1mtrx_offer"].astype(float).to_numpy()

    make_submission(base, final, args.out)
    verify_submission(args.out, SAMPLE_SUBMISSION_PATH)
    LOG.info("Рекордный blend (11 моделей) сохранён: %s", args.out)


if __name__ == "__main__":
    main()
