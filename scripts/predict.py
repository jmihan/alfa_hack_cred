"""INFERENCE-режим: собирает финальный сабмит из УЖЕ ОБУЧЕННЫХ моделей.

Загружает модели из `models/` (их кладёт `scripts/fit_pipeline.py`), строит признаки
для test и собирает two-stage сабмит — без обучения (~5 мин).

Two-stage по `pil1mtrx_offer`:
- A (есть pil1, ~66%): rank-avg 5-модельного A-бленда + hard-rule.
- B (нет pil1, ~34%): bAllL — rank-avg 5 XGBoost + 1 LightGBM extended.

Если обученных моделей ещё нет — сначала запустите `scripts/fit_pipeline.py`.

Запуск:
    python scripts/predict.py
    python scripts/predict.py --out submissions/record_submission.csv
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from alfa_cred.config import MODELS_DIR, REQUEST_ID, SUBMISSIONS_DIR, VARIANT_ID
from alfa_cred.models.a_blend import load_a_models, predict_a_blend
from alfa_cred.models.b_ball import load_b_ball, predict_b_ball
from alfa_cred.two_stage import (
    PIL_COL,
    assemble_submission,
    prepare_a_features,
    prepare_b_features,
)
from alfa_cred.utils import get_logger

LOG = get_logger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: сабмит из обученных моделей (близко к рекорду)")
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "record_submission.csv")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.models_dir / "a_manifest.json").exists():
        raise FileNotFoundError(
            f"Модели не найдены в {args.models_dir}. Сначала обучите: python scripts/fit_pipeline.py"
        )

    # ---- A-сторона ----
    _train_a, test_a, fc_a, _cat_a = prepare_a_features()
    a_pct = predict_a_blend(load_a_models(args.models_dir), fc_a, test_a)
    a_keys = test_a[[REQUEST_ID, VARIANT_ID, PIL_COL]].reset_index(drop=True)
    LOG.info("A-бленд загружен и применён")
    del _train_a, test_a
    gc.collect()

    # ---- B-сторона bAllL ----
    _train_b, _test_sorted, _is_b, test_b, fc_b, cat_b = prepare_b_features()
    b_keys = test_b[[REQUEST_ID, VARIANT_ID]].copy()
    b_keys["b_score"] = predict_b_ball(load_b_ball(args.models_dir), fc_b, cat_b, test_b)
    LOG.info("bAllL загружен и применён (5 XGBoost + 1 LightGBM)")

    assemble_submission(a_keys, a_pct, b_keys, args.out)
    LOG.info("Сабмит собран из обученных моделей: %s", args.out)


if __name__ == "__main__":
    main()
