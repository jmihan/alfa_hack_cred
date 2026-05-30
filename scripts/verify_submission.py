"""Сверка двух сабмитов на эквивалентность ранжирования.

Бустинговые бленды на CPU не дают побитово одинаковых скоров между прогонами
(многопоточность LightGBM), поэтому воспроизводимость проверяется не по равенству
чисел, а по совпадению **решения о ранжировании**: какой оффер встаёт первым в
каждой заявке и какой состав попадает в топ-5 (именно это влияет на NDCG@5).

Запуск:
    python scripts/predict.py --out submissions/check.csv
    python scripts/verify_submission.py submissions/check.csv submissions/record_submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alfa_cred.config import REQUEST_ID, SUBMISSION_SEPARATOR, VARIANT_ID

TOP1_THRESHOLD = 1.0  # ожидаем полное совпадение топ-1 при воспроизведении рекорда


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=SUBMISSION_SEPARATOR)
    df[REQUEST_ID] = df[REQUEST_ID].astype(str)
    df[VARIANT_ID] = df[VARIANT_ID].astype("int64")
    return df


def _top1(df: pd.DataFrame) -> pd.Series:
    """variant_no с максимальным score в каждой заявке (тай-брейк по variant_no asc)."""
    ordered = df.sort_values([REQUEST_ID, "score", VARIANT_ID], ascending=[True, False, True])
    return ordered.groupby(REQUEST_ID, sort=False)[VARIANT_ID].first()


def _top5(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values([REQUEST_ID, "score", VARIANT_ID], ascending=[True, False, True])
    return ordered.groupby(REQUEST_ID, sort=False)[VARIANT_ID].apply(lambda s: tuple(s.head(5)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Сверка двух сабмитов по ранжированию")
    parser.add_argument("generated", type=Path, help="Проверяемый сабмит")
    parser.add_argument("reference", type=Path, help="Эталонный сабмит")
    args = parser.parse_args()

    gen, ref = _load(args.generated), _load(args.reference)

    gen_keys = set(map(tuple, gen[[REQUEST_ID, VARIANT_ID]].itertuples(index=False)))
    ref_keys = set(map(tuple, ref[[REQUEST_ID, VARIANT_ID]].itertuples(index=False)))
    if gen_keys != ref_keys:
        raise SystemExit(
            f"Наборы (request_id, variant_no) различаются: {len(gen_keys ^ ref_keys)} строк"
        )

    g1, r1 = _top1(gen), _top1(ref)
    common = g1.index.intersection(r1.index)
    top1_match = float((g1.loc[common] == r1.loc[common]).mean())

    g5, r5 = _top5(gen), _top5(ref)
    top5_jaccard = float(
        np.mean([
            len(set(a) & set(b)) / len(set(a) | set(b))
            for a, b in zip(g5.loc[common], r5.loc[common])
        ])
    )

    merged = gen.merge(ref, on=[REQUEST_ID, VARIANT_ID], suffixes=("_gen", "_ref"))
    max_abs = float(np.abs(merged["score_gen"] - merged["score_ref"]).max())

    print(f"заявок сверено:         {len(common)}")
    print(f"top-1 совпадение:       {top1_match:.6f}")
    print(f"top-5 Jaccard (средн.): {top5_jaccard:.6f}")
    print(f"max abs score diff:     {max_abs:.2e}")

    if top1_match >= TOP1_THRESHOLD:
        print("OK: ранжирование воспроизведено (top-1 = 1.0)")
        return
    raise SystemExit(f"top-1 совпадение {top1_match:.6f} < {TOP1_THRESHOLD}")


if __name__ == "__main__":
    main()
