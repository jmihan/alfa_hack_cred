"""Общие настройки pytest.

`conftest.py` импортируется pytest ДО тестовых модулей, поэтому грузим здесь torch
первым (до numpy/lightgbm): иначе на Windows при импорте torch после numpy падает
c10.dll с WinError 1114 (конфликт OpenMP/MKL). torch может отсутствовать (лёгкий CG/
CI) — тогда просто пропускаем, а тесты, которым он нужен, сами делают importorskip.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:  # noqa: SIM105
    import torch  # noqa: F401  — грузим первым ради порядка OpenMP/MKL на Windows
except Exception:
    pass
