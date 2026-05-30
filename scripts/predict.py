"""INFERENCE-режим: собирает финальный сабмит из УЖЕ ОБУЧЕННЫХ моделей.

Загружает модели из `models/` (их кладёт `scripts/fit_pipeline.py`), строит
признаки для test и собирает two-stage сабмит — без обучения (~10 мин).

Two-stage по `pil1mtrx_offer`:
- A (есть pil1, ~65%): rank-avg 5-модельного A-бленда + hard-rule.
- B (нет pil1, ~35%): 0.70·b_blend + 0.30·pointwise-MLP.

Если обученных моделей ещё нет — сначала запустите `scripts/fit_pipeline.py`.

Запуск:
    python scripts/predict.py
    python scripts/predict.py --out submissions/record_submission.csv --device cpu
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  — torch до lightgbm/catboost

import argparse  # noqa: E402
import gc  # noqa: E402
from pathlib import Path  # noqa: E402

from alfa_cred.config import MODELS_DIR, REQUEST_ID, SUBMISSIONS_DIR, VARIANT_ID  # noqa: E402
from alfa_cred.models.a_blend import load_a_models, predict_a_blend  # noqa: E402
from alfa_cred.models.b_blend import load_b_models, predict_b_blend  # noqa: E402
from alfa_cred.models.mlp_pointwise import load_mlp, predict_mlp  # noqa: E402
from alfa_cred.two_stage import (  # noqa: E402
    B_BLEND_WEIGHT,
    PIL_COL,
    assemble_submission,
    pct_rank,
    prepare_a_features,
    prepare_b_features,
)
from alfa_cred.utils import get_logger  # noqa: E402

LOG = get_logger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: сабмит из обученных моделей (LB ≈ 92.19)")
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "record_submission.csv")
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--device", type=str, default=None, help="cuda|cpu для MLP (по умолчанию авто)")
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

    # ---- B-сторона ----
    _train_b, _test_sorted, _is_b, test_b, fc_b, cat_b = prepare_b_features()
    b_rank = pct_rank(test_b, predict_b_blend(load_b_models(args.models_dir), fc_b, cat_b, test_b))
    mlp_rank = pct_rank(test_b, predict_mlp(load_mlp(args.models_dir), test_b, device=args.device))
    LOG.info("b_blend и pointwise-MLP загружены и применены")

    b_keys = test_b[[REQUEST_ID, VARIANT_ID]].copy()
    b_keys["b_score"] = B_BLEND_WEIGHT * b_rank + (1 - B_BLEND_WEIGHT) * mlp_rank

    assemble_submission(a_keys, a_pct, b_keys, args.out)
    LOG.info("Сабмит собран из обученных моделей: %s", args.out)


if __name__ == "__main__":
    main()
