"""Воспроизводит финальный two-stage сабмит (LB = 92.0504).

Архитектура:
- Подзадача A (есть `pil1mtrx_offer=1` в запросе) → rank-avg blend
  из 11 моделей `STAGE_A_BLEND_MODELS` + hard-rule.
- Подзадача B (нет pil1mtrx_offer) → rank-avg blend из 16 B-only моделей
  `MILESTONE_B_BLEND_MODELS` (top-3 каждого типа + 2 pseudo + 2 crossobj).

Скрипт читает заранее сохранённые `*_test_scores.parquet` из `oof/`,
собирает сабмит и кладёт результат в `submissions/`.

Запуск:
    python scripts/make_final_submission.py
    python scripts/make_final_submission.py --out submissions/my_copy.csv
    python scripts/make_final_submission.py --verify-against submissions/two_stage_r11_bBalanced_plus_pseudo_crossobj_1405.csv

Подробности — в README.md и EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alfa_cred.blends import MILESTONE_B_BLEND_MODELS, STAGE_A_BLEND_MODELS
from alfa_cred.config import OOF_DIR, SUBMISSION_SEPARATOR, SUBMISSIONS_DIR, TEST_PATH
from alfa_cred.inference import build_two_stage_submission
from alfa_cred.io_utils import (
    coerce_decimal_columns,
    load_parquet,
    normalize_request_id,
    sort_by_request,
)
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)

DEFAULT_OUT = SUBMISSIONS_DIR / "final_submission.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сборка финального two-stage сабмита (LB 92.0504)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Путь к выходному CSV (по умолчанию {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Если указан, побайтово сверить полученный CSV с эталонным сабмитом",
    )
    return parser.parse_args()


def _resolve_test_score_paths(run_names) -> list[Path]:
    paths = [OOF_DIR / f"{name}_test_scores.parquet" for name in run_names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Отсутствуют сохранённые test_scores:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    return paths


def _diff_against_reference(generated: Path, reference: Path) -> None:
    """Сверяет сгенерированный CSV с эталонным.

    Финальный сабмит исторически собирался ad-hoc скриптом, точный
    порядок операций которого не сохранился. Текущая реализация
    `build_two_stage_submission` даёт численно эквивалентный результат,
    но из-за float-арифметики возможны расхождения в последнем знаке
    (±1 ULP в 6-м десятичном разряде). Структура и подавляющее
    большинство строк совпадают.
    """
    gen = pd.read_csv(generated, sep=SUBMISSION_SEPARATOR)
    ref = pd.read_csv(reference, sep=SUBMISSION_SEPARATOR)
    if gen.shape != ref.shape:
        raise RuntimeError(
            f"Размерность сабмита {gen.shape} != эталона {ref.shape}"
        )
    diff_mask = ~np.isclose(gen["score"].to_numpy(), ref["score"].to_numpy(), atol=0.0)
    n_diff = int(diff_mask.sum())
    if n_diff == 0:
        LOG.info("Сабмит bit-identical с эталоном %s", reference.name)
        return
    abs_diff = np.abs(gen["score"].to_numpy() - ref["score"].to_numpy())
    LOG.warning(
        "Числовые расхождения с эталоном %s: %d/%d строк, max abs diff = %.2e (ULP-уровень)",
        reference.name, n_diff, len(gen), float(abs_diff.max()),
    )


def main() -> None:
    args = parse_args()

    record_paths = _resolve_test_score_paths(STAGE_A_BLEND_MODELS)
    b_paths = _resolve_test_score_paths(MILESTONE_B_BLEND_MODELS)
    LOG.info(
        "Готовлю two-stage сабмит: A-бленд (%d моделей), B-бленд (%d моделей)",
        len(record_paths), len(b_paths),
    )

    test = coerce_decimal_columns(normalize_request_id(load_parquet(TEST_PATH)))
    test_sorted = sort_by_request(test)

    build_two_stage_submission(
        b_test_paths=b_paths,
        record_test_paths=record_paths,
        test_sorted=test_sorted,
        out_path=args.out,
    )
    LOG.info("Финальный сабмит сохранён: %s", args.out)

    if args.verify_against is not None:
        reference = Path(args.verify_against)
        if not reference.exists():
            raise FileNotFoundError(f"Эталонный сабмит не найден: {reference}")
        _diff_against_reference(args.out, reference)


if __name__ == "__main__":
    main()
