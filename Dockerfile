# Образ для воспроизведения пайплайна ранжирования кредитных офферов.
#
# CPU по умолчанию (всё обучается на CPU и воспроизводит LB-скор). Данные и
# артефакты (./data, ./models, ./submissions, ...) внутрь образа НЕ кладём —
# монтируются как volume (см. docker-compose.yml). torch ставится CPU-сборкой.
# GPU-вариант — опционально, через --gpus all + CUDA-образ (см. README).

FROM python:3.11-slim-bookworm

# libgomp1 — OpenMP-рантайм для LightGBM/XGBoost (иначе import падает).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    MPLCONFIGDIR=/tmp/mpl \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Зависимости — отдельным слоем (кэшируется при изменениях кода). torch — CPU-сборка
# (без CUDA-библиотек: легче образ и не нужна GPU на машине организаторов).
COPY requirements.txt ./
# torch — CPU-сборка (с индекса pytorch), остальное из requirements (pypi). Базовый
# образ уже содержит setuptools (>=68) и wheel, поэтому НЕ апгрейдим их (лишние
# обращения к pypi), а editable-install ниже идёт с --no-build-isolation.
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Пакет и точки входа.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
# Зафиксированные предсказания рекорда (~9 МБ) для режима reproduce — байт-в-байт 92.1957.
COPY artifacts ./artifacts
RUN pip install -e . --no-build-isolation

# По умолчанию — байт-в-байт сборка рекордного сабмита (LB 92.1957) из артефактов
# (нужен смонтированный ./data). Обучение с нуля — через docker compose run train.
CMD ["python", "scripts/reproduce_record.py", "--out", "submissions/record_submission.csv"]
