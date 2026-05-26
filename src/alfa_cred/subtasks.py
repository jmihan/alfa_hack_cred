"""Расщепление задачи на подзадачи A/B по бизнес-флагу `pil1mtrx_offer`.

В EDA нашлось, что задача распадается на две неравные части:

- **Подзадача A** (≈66% запросов): в группе есть оффер с `pil1mtrx_offer=1`,
  у него target rate 99.73%. Hard-rule «pil1mtrx первыми» сам по себе даёт
  NDCG@5 ≈ 1.0 на таких запросах.
- **Подзадача B** (≈34% запросов): pil1mtrx-флага нет ни у одного оффера,
  нужна полноценная LTR-модель.

Это даёт сильное упрощение: модель имеет смысл обучать только на подзадаче B,
а на подзадаче A работает hard-rule. Финальный two-stage сабмит так и устроен.
"""

from __future__ import annotations

import pandas as pd

from alfa_cred.config import REQUEST_ID


def filter_subtask_b(df: pd.DataFrame, pil_col: str = "pil1mtrx_offer") -> pd.DataFrame:
    """Возвращает строки запросов, где НЕТ оффера с `pil1mtrx_offer=1`.

    Это «трудные» запросы (~34% от всех), где нет гарантированного выбора
    через hard-rule.

    Параметры
    ----------
    df : pd.DataFrame
        Должен содержать колонки `REQUEST_ID` и `pil_col`.
    pil_col : str
        Имя бизнес-флага, по умолчанию `pil1mtrx_offer`.

    Возвращает
    ----------
    pd.DataFrame
        Подмножество `df` с `reset_index(drop=True)`.
    """
    has_pil1mtrx = df.groupby(REQUEST_ID, sort=False)[pil_col].transform("max")
    return df[has_pil1mtrx == 0].reset_index(drop=True)


def is_subtask_b_mask(df: pd.DataFrame, pil_col: str = "pil1mtrx_offer"):
    """Булева маска по строкам: True для строк из запросов подзадачи B.

    В отличие от `filter_subtask_b`, не отбрасывает строки, а возвращает
    маску длины `len(df)` — удобно для условной подстановки скоров в
    two-stage сабмите.
    """
    has_pil1mtrx = df.groupby(REQUEST_ID, sort=False)[pil_col].transform("max")
    return (has_pil1mtrx == 0).to_numpy()
