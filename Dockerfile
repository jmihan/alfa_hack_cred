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
RUN pip install --upgrade pip \
    && pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Пакет и точки входа.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install -e .

# По умолчанию — полное обучение пайплайна с нуля + сабмит (нужен смонтированный ./data).
CMD ["python", "scripts/fit_pipeline.py", "--out", "submissions/record_submission.csv"]
