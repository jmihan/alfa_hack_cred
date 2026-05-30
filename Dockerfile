# Образ для воспроизведения пайплайна ранжирования кредитных офферов.
#
# CPU по умолчанию (рекордный B-бленд считается на CPU и так воспроизводит LB).
# Данные и артефакты внутрь образа НЕ кладём — монтируются как volume (см.
# docker-compose.yml). GPU-вариант — опционально, через --gpus all и базовый
# CUDA-образ (см. секцию «Запуск в Docker» в README).

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
    PIP_RETRIES=10

WORKDIR /app

# Зависимости — отдельным слоем, чтобы кэшировались при изменениях кода.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Пакет и точки входа.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install -e .

# По умолчанию — сборка финального сабмита (нужны смонтированные data/ и oof/).
CMD ["python", "scripts/predict.py", "--out", "submissions/record_submission.csv"]
