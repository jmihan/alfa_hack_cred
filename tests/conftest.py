"""Общие настройки pytest.

На Windows lightgbm и xgboost тянут свои OpenMP-рантаймы; снимаем конфликт дублей,
чтобы импорт моделей в тестах не падал.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
