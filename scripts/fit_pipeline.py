"""TRAIN-режим: обучает весь рекордный пайплайн С НУЛЯ и сохраняет модели.

Обучает A-бленд (5 моделей), B-бленд (8 моделей) и pointwise-MLP (3 сида),
сохраняет их в `models/` (volume) и пишет финальный two-stage сабмит. После
этого `scripts/predict.py` может собирать сабмит без переобучения (inference).

Запуск:
    python scripts/fit_pipeline.py
    python scripts/fit_pipeline.py --out submissions/record_submission.csv --device cpu
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  — torch до lightgbm/catboost

import argparse  # noqa: E402
import gc  # noqa: E402
from pathlib import Path  # noqa: E402

from alfa_cred.config import MODELS_DIR, REQUEST_ID, SUBMISSIONS_DIR, VARIANT_ID  # noqa: E402
from alfa_cred.models.a_blend import fit_a_models, predict_a_blend, save_a_models  # noqa: E402
from alfa_cred.models.b_blend import fit_b_models, predict_b_blend, save_b_models  # noqa: E402
from alfa_cred.models.mlp_pointwise import fit_mlp, predict_mlp, save_mlp  # noqa: E402
from alfa_cred.two_stage import (  # noqa: E402
    B_BLEND_WEIGHT,
    PIL_COL,
    assemble_submission,
    pct_rank,
    prepare_a_features,
    prepare_b_features,
)
from alfa_cred.utils import get_logger  # noqa: E402

LOG = get_logger("fit_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучить рекордный пайплайн с нуля + сохранить модели (LB ≈ 92.19)")
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "record_submission.csv")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--device", type=str, default=None, help="cuda|cpu для MLP (по умолчанию авто)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # ---- A-сторона (расширенный набор) ----
    train_a, test_a, fc_a, cat_a = prepare_a_features()
    LOG.info("A: %d фич (%d кат.), train %d / test %d", len(fc_a), len(cat_a), len(train_a), len(test_a))
    a_models = fit_a_models(train_a, fc_a, cat_a)
    save_a_models(a_models, args.models_dir)
    a_pct = predict_a_blend(a_models, fc_a, test_a)
    a_keys = test_a[[REQUEST_ID, VARIANT_ID, PIL_COL]].reset_index(drop=True)
    LOG.info("A-бленд обучён и сохранён (5 моделей)")
    del train_a, test_a, a_models
    gc.collect()

    # ---- B-сторона (широкий набор) ----
    train_b, _test_sorted, _is_b, test_b, fc_b, cat_b = prepare_b_features()
    LOG.info("B: train-B %d заявок -> test-B %d заявок",
             train_b[REQUEST_ID].nunique(), test_b[REQUEST_ID].nunique())
    b_models = fit_b_models(train_b, fc_b, cat_b)
    save_b_models(b_models, args.models_dir)
    b_rank = pct_rank(test_b, predict_b_blend(b_models, fc_b, cat_b, test_b))
    LOG.info("b_blend обучён и сохранён (8 моделей)")
    mlp = fit_mlp(train_b, fc_b, cat_b, device=args.device)
    save_mlp(mlp, args.models_dir)
    mlp_rank = pct_rank(test_b, predict_mlp(mlp, test_b, device=args.device))
    LOG.info("pointwise-MLP обучён и сохранён (3 сида)")

    b_keys = test_b[[REQUEST_ID, VARIANT_ID]].copy()
    b_keys["b_score"] = B_BLEND_WEIGHT * b_rank + (1 - B_BLEND_WEIGHT) * mlp_rank

    assemble_submission(a_keys, a_pct, b_keys, args.out)
    LOG.info("Готово. Модели сохранены в %s, сабмит: %s", args.models_dir, args.out)


if __name__ == "__main__":
    main()
