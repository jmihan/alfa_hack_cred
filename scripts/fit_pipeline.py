"""TRAIN-режим: обучает пайплайн С НУЛЯ и сохраняет модели.

Обучает A-бленд (5 моделей) и B-сторону bAllL (5 XGBoost + 1 LightGBM extended),
сохраняет их в `models/` (volume) и пишет финальный two-stage сабмит. После этого
`scripts/predict.py` собирает сабмит без переобучения (inference).

A-бленд record_11 (агрегат 11 моделей) переобучить с нуля побайтно нельзя, поэтому
здесь обучается компактный 5-модельный A-бленд (близкий к record_11). Точное
воспроизведение лучшего сабмита — режим `reproduce` (`scripts/reproduce_record.py`).

Запуск:
    python scripts/fit_pipeline.py
    python scripts/fit_pipeline.py --out submissions/record_submission.csv --device cpu
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

from alfa_cred.config import MODELS_DIR, REQUEST_ID, SUBMISSIONS_DIR, VARIANT_ID
from alfa_cred.models.a_blend import fit_a_models, predict_a_blend, save_a_models
from alfa_cred.models.b_ball import fit_b_ball, predict_b_ball, save_b_ball
from alfa_cred.two_stage import (
    PIL_COL,
    assemble_submission,
    prepare_a_features,
    prepare_b_features,
)
from alfa_cred.utils import get_logger

LOG = get_logger("fit_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучить пайплайн с нуля + сохранить модели (близко к рекорду)")
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "record_submission.csv")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu для XGBoost/CatBoost; по умолчанию из ALFA_DEVICE, иначе cpu")
    return p.parse_args()


def _resolve_device(arg_device: str | None) -> str:
    # Приоритет: --device > ALFA_DEVICE > cpu. LightGBM всегда на CPU; cuda включает
    # GPU только для XGBoost (A- и B-сторона) и CatBoost (A-сторона).
    return arg_device or os.environ.get("ALFA_DEVICE") or "cpu"


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    LOG.info("Устройство обучения: %s (XGBoost/CatBoost; LightGBM на CPU)", device)

    # ---- A-сторона (расширенный набор) ----
    train_a, test_a, fc_a, cat_a = prepare_a_features()
    LOG.info("A: %d фич (%d кат.), train %d / test %d", len(fc_a), len(cat_a), len(train_a), len(test_a))
    a_models = fit_a_models(train_a, fc_a, cat_a, device=device)
    save_a_models(a_models, args.models_dir)
    a_pct = predict_a_blend(a_models, fc_a, test_a)
    a_keys = test_a[[REQUEST_ID, VARIANT_ID, PIL_COL]].reset_index(drop=True)
    LOG.info("A-бленд обучён и сохранён (5 моделей)")
    del train_a, test_a, a_models
    gc.collect()

    # ---- B-сторона bAllL (тот же расширенный набор, B-only) ----
    train_b, _test_sorted, _is_b, test_b, fc_b, cat_b = prepare_b_features()
    LOG.info("B: train-B %d заявок -> test-B %d заявок",
             train_b[REQUEST_ID].nunique(), test_b[REQUEST_ID].nunique())
    b_models = fit_b_ball(train_b, fc_b, cat_b, device=device)
    save_b_ball(b_models, args.models_dir)
    b_keys = test_b[[REQUEST_ID, VARIANT_ID]].copy()
    b_keys["b_score"] = predict_b_ball(b_models, fc_b, cat_b, test_b)  # rank-avg 5 XGB + 1 LGBM
    LOG.info("bAllL обучён и сохранён (5 XGBoost + 1 LightGBM)")

    assemble_submission(a_keys, a_pct, b_keys, args.out)
    LOG.info("Готово. Модели сохранены в %s, сабмит: %s", args.models_dir, args.out)


if __name__ == "__main__":
    main()
