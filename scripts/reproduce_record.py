"""REPRODUCE-режим: байт-в-байт сборка рекордного сабмита (LB 92.1957).

Финальный сабмит — двухстадийный по `pil1mtrx_offer`:
- A-сторона (есть pil1-оффер): ранжирование A-бленда record_11 — зафиксированные
  предсказания на test (`artifacts/record/a_side_record11.parquet`).
- B-сторона (нет pil1): `0.70·rank(b_blend) + (1-0.70)·rank(MLP)` — зафиксированные
  перцентильные ранги на test (`artifacts/record/b_pct_*.npy`, ключи в
  `b_keys.parquet`).

Порядок строк берётся из самих данных (`load_raw` → `sort_by_request`), поэтому
сабмит маппится на актуальный test и сверяется со схемой `commit.csv`. Результат
детерминирован на любой машине (CPU, GPU не нужен) и проверяется по sha256 против
эталона из `manifest.json` — при расхождении скрипт падает с ошибкой.

Зачем отдельный режим: A-бленд record_11 (агрегат большого числа прогонов) и
NN-компонент B-стороны невозможно переобучить байт-в-байт на другом железе, тогда
как зафиксированные предсказания дают точное воспроизведение рекорда. Режим `train`
(`scripts/fit_pipeline.py`) обучает близкий пайплайн с нуля (LB ≈ 92.19).

Запуск:
    python scripts/reproduce_record.py
    python scripts/reproduce_record.py --out submissions/record_submission.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alfa_cred.config import (
    PROJECT_ROOT,
    REQUEST_ID,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    VARIANT_ID,
)
from alfa_cred.inference import make_submission, verify_submission
from alfa_cred.io_utils import load_raw, sort_by_request
from alfa_cred.utils import get_logger

LOG = get_logger("reproduce_record")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "record"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Байт-в-байт сборка рекордного сабмита (LB 92.1957)")
    p.add_argument("--out", type=Path, default=SUBMISSIONS_DIR / "record_submission.csv")
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.artifacts / "manifest.json").read_text(encoding="utf-8"))
    w = float(manifest["w_bblend"])

    # Порядок строк = sort_by_request(test): маппим зафиксированные предсказания
    # на актуальный test организаторов (а не на «вшитый» список ключей).
    _train, test, _features = load_raw()
    test_sorted = sort_by_request(test)
    base = test_sorted[[REQUEST_ID, VARIANT_ID]].copy()
    base[REQUEST_ID] = base[REQUEST_ID].astype(str)
    base[VARIANT_ID] = base[VARIANT_ID].astype("int32")

    # A-сторона: ранжирование record_11 (по ключам).
    a_side = pd.read_parquet(args.artifacts / "a_side_record11.parquet")
    a_side[REQUEST_ID] = a_side[REQUEST_ID].astype(str)
    a_side[VARIANT_ID] = a_side[VARIANT_ID].astype("int32")
    score = base.merge(a_side, on=[REQUEST_ID, VARIANT_ID], how="left")["score"].to_numpy()

    # B-сторона: 0.70·b_blend + (1-0.70)·MLP (перцентильные ранги), перекрывает A на B-строках.
    keys = pd.read_parquet(args.artifacts / "b_keys.parquet")
    keys[REQUEST_ID] = keys[REQUEST_ID].astype(str)
    keys[VARIANT_ID] = keys[VARIANT_ID].astype("int32")
    b_pct = np.load(args.artifacts / "b_pct_bblend.npy")
    mlp_pct = np.load(args.artifacts / "b_pct_mlp.npy")
    keys["b_comb"] = w * b_pct + (1 - w) * mlp_pct
    b_map = base.merge(keys[[REQUEST_ID, VARIANT_ID, "b_comb"]], on=[REQUEST_ID, VARIANT_ID], how="left")["b_comb"].to_numpy()
    is_b = ~np.isnan(b_map)
    score[is_b] = b_map[is_b]
    LOG.info("A-строк %d, B-строк %d (вес b_blend=%.2f)", (~is_b).sum(), int(is_b.sum()), w)

    if np.isnan(score).any():
        raise ValueError("После сборки остались NaN — артефакты не покрывают весь test")

    out = base[[REQUEST_ID, VARIANT_ID]].copy()
    make_submission(out, np.round(score, 6), args.out)
    verify_submission(args.out, SAMPLE_SUBMISSION_PATH)

    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    if digest == manifest["sha256"]:
        LOG.info("OK: sha256 совпал с эталоном record (LB %s) -> воспроизведено байт-в-байт", manifest["lb"])
    else:
        raise SystemExit(
            f"sha256 НЕ совпал: {digest} != {manifest['sha256']} (эталон record). "
            "Проверьте данные/артефакты — сабмит не байт-в-байт."
        )
    LOG.info("Рекордный сабмит собран: %s", args.out)


if __name__ == "__main__":
    main()
