#!/usr/bin/env bash
# Единая точка запуска пайплайна ранжирования кредитных офферов через Docker.
#
#   ./run.sh                # собрать образ + воспроизвести рекорд (LB 92.1957)
#   ./run.sh reproduce      # байт-в-байт рекорд из artifacts/record (~2 мин, GPU не нужен)
#   ./run.sh train          # обучить весь пайплайн с нуля (CPU, ~55 мин) -> ./models + сабмит
#   ./run.sh inference      # собрать сабмит из обученных моделей (без обучения)
#   ./run.sh interpret      # SHAP + permutation интерпретация -> ./reports/interpretation
#   ./run.sh cv             # CV одиночной модели по YAML-конфигу (эксперименты)
#   ./run.sh test           # юнит-тесты (локально, в активированном .venv)
#
# Данные должны лежать в ./data (train/test/features parquet + commit.csv).
set -euo pipefail

MODE="${1:-reproduce}"
IMAGE="alfa-cred:latest"

build_if_needed() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo ">> Образ $IMAGE не найден — собираю (docker build) ..."
    docker build -t "$IMAGE" .
  fi
}

require_data() {
  if [ ! -f "data/test_dataset_small.pq" ]; then
    echo "!! Нет data/test_dataset_small.pq — положите данные хакатона в ./data" >&2
    exit 1
  fi
}

case "$MODE" in
  reproduce|train|inference|interpret|cv)
    require_data
    build_if_needed
    echo ">> docker compose run --rm $MODE"
    docker compose run --rm "$MODE"
    ;;
  test)
    echo ">> pytest"
    python -m pytest -q
    ;;
  *)
    echo "Неизвестный режим: $MODE" >&2
    echo "Доступно: reproduce | train | inference | interpret | cv | test" >&2
    exit 1
    ;;
esac
