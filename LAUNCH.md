# Запуск за 3 шага

Краткая инструкция для проверки решения. Подробности — в [README](README.md).

## 0. Что нужно

- Docker (для GPU-обучения — дополнительно nvidia-container-toolkit).
- Данные хакатона в `./data/`:
  `train_dataset_small.pq`, `test_dataset_small.pq`, `features_small.pq`,
  `feature_description.csv`, `commit.csv`.

## 1. Собрать образ

```bash
docker build -t alfa-cred:latest .     # за корпоративным VPN: добавьте --network=host
```

## 2. Воспроизвести результат

**Точный рекорд (LB 92.1957), байт-в-байт, ~2 мин, GPU не нужен:**

```bash
docker compose run --rm reproduce
# сабмит -> ./submissions/record_submission.csv ; в логе "OK: sha256 совпал ... 92.1957"
```

**Обучение всего пайплайна с нуля (LB ≈ 92.19, ~55 мин CPU):**

```bash
docker compose run --rm train          # обучит модели в ./models и запишет сабмит
docker compose run --rm inference      # пересобрать сабмит из ./models (без обучения)
```

GPU-обучение (XGBoost/CatBoost/MLP на GPU, LightGBM на CPU):

```bash
docker build -f Dockerfile.gpu -t alfa-cred:gpu .
docker compose --profile gpu run --rm train-gpu
```

Всё одной командой: `./run.sh` (собрать + reproduce) или `./run.sh train|inference|interpret|test`.

## 3. Проверки (опционально)

```bash
pytest                                  # юнит-тесты: NDCG@5, формат сабмита, is_best_both
docker compose run --rm interpret       # SHAP + permutation -> ./reports/interpretation
docker compose --profile mlflow up      # MLflow UI -> http://localhost:5000
```

> Почему рекорд воспроизводится из артефактов, а не переобучением: A-сторона
> (бленд record_11) и нейросеть не воспроизводятся побайтно на другом железе,
> поэтому их предсказания зафиксированы в `artifacts/record/` (≈9 МБ, в репозитории),
> а `reproduce` детерминированно собирает из них ровно рекордный сабмит. Режим
> `train` обучает близкий пайплайн с нуля (≈ 92.19).
