"""Глобальные настройки проекта: пути, ключевые колонки, сиды."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Пути к каталогам
DATA_DIR: Path = PROJECT_ROOT / "data"
SUBMISSIONS_DIR: Path = PROJECT_ROOT / "submissions"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"
OOF_DIR: Path = PROJECT_ROOT / "oof"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# Файлы данных
TRAIN_PATH: Path = DATA_DIR / "train_dataset_small.pq"
TEST_PATH: Path = DATA_DIR / "test_dataset_small.pq"
FEATURES_PATH: Path = DATA_DIR / "features_small.pq"
FEATURE_DESCRIPTION_PATH: Path = DATA_DIR / "feature_description.csv"
SAMPLE_SUBMISSION_PATH: Path = DATA_DIR / "commit.csv"

# Ключевые колонки
TARGET: str = "is_deal"
REQUEST_ID: str = "request_id"
VARIANT_ID: str = "variant_no"
APP_ID: str = "app_id"
DATE_PART: str = "date_part"
REQUEST_RECEIVED: str = "request_received"

# Идентификаторы и нефичевые колонки (исключаем из feature set)
ID_COLUMNS: tuple[str, ...] = (
    "app_id",
    "request_id",
    "offer_id",
    "request_received",
    "date_part",
)

# Сиды
RANDOM_STATE: int = 42

# Параметры формата сабмита
SUBMISSION_SEPARATOR: str = ";"
SUBMISSION_COLUMNS: tuple[str, ...] = ("request_id", "variant_no", "score")
SUBMISSION_SCORE_DECIMALS: int = 6
