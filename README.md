# Alfa Credit Offer Ranking

Решение задачи «Прогноз кредита по клиентам» хакатона Альфа-Банка
(Яндекс Контест). Цель — ранжировать кредитные предложения внутри
каждого запроса клиента так, чтобы максимизировать метрику **NDCG@5**.

## Постановка

Для каждого запроса `request_id` дано до 50 кредитных предложений
(`variant_no`). Нужно выдать `score` для каждой пары
`(request_id, variant_no)`, по которому предложения внутри запроса
ранжируются. Качество — средний NDCG@5 по запросам. На лидерборде
метрика умножается на 100.

Формат сабмита (`commit.csv`, разделитель `;`):

```
request_id;variant_no;score
0;10;0.016394
...
```

## Результаты

Стартовая планка организаторов — RandomForest c NDCG@5 ≈ 0.57 на train
(на LB не сабмитили). Простая эвристика «pil1mtrx_offer первыми +
variant_no asc» уже даёт NDCG@5 ≈ 0.80 на train.

| Этап | LB |
|------|----|
| LGBM LambdaRank baseline | 91.634 |
| LGBM Optuna tuned | 91.7512 |
| LGBM extended features | 91.8648 |
| Сильнейшая одиночка (lgbm_boot_v_s256) | 91.8774 |
| blend 11 моделей (record_11) | 91.9668 |
| two-stage прорыв (Pipeline K, B-only XGB) | 91.9939 |
| two-stage + bBalanced (12 B-моделей) | 92.0317 |
| two-stage + bBalanced + pseudo + crossobj | **92.0504** ← финал |

Сабмитов использовано: **75/100**. Место на момент финала: top-2/3.

## Архитектура решения

EDA показала, что задача распадается на две неравные части по флагу
`pil1mtrx_offer`. Финальное решение это эксплуатирует напрямую:

```
                          test (936 883 строк, 38 750 запросов)
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
       подзадача A (≈66% запросов)              подзадача B (≈34% запросов)
       в группе есть pil1mtrx_offer=1            pil1mtrx-флага нет
                  │                                         │
                  ▼                                         ▼
       rank-avg blend record_11                rank-avg blend 16 B-only моделей
       (11 моделей: LGBM, CatBoost, XGB)        (12 bBalanced + 2 pseudo + 2 crossobj)
                  │                                         │
                  └────────────────┬────────────────────────┘
                                   │
                          + hard-rule: pil1mtrx=1 → +1.0 к скору
                                   │
                                   ▼
                              финальный сабмит
                                LB 92.0504
```

**B-only blend** (16 моделей):
- 12 моделей bBalanced (top-3 каждого типа архитектуры): 3 CatBoost
  YetiRank разных seed, 3 LightGBM bootstrap, 3 XGBoost multi-seed,
  3 LightGBM Optuna.
- 2 pseudo-labeling XGB: дообучение на ~5% самых уверенных предсказаний
  test, добавленных как pseudo-train.
- 2 cross-objective модели: `xgb_pairwise` (pairwise objective) и
  `lgbm_xendcg` (cross-entropy NDCG). Дают диверсификацию через
  альтернативную целевую функцию.

Подробный лог экспериментов с цифрами CV/LB по каждому pipeline — в
[`EXPERIMENTS.md`](EXPERIMENTS.md). EDA-выводы — в
[`notebooks/EDA_FINDINGS.md`](notebooks/EDA_FINDINGS.md).

## Структура репозитория

```
.
├── data/                          # исходные данные (gitignored)
├── task/                          # условие задачи (gitignored)
├── src/alfa_cred/                 # переиспользуемый код
│   ├── config.py                  # пути и константы
│   ├── io_utils.py                # загрузка/мёрж parquet, кодирование cat-фич
│   ├── metrics.py                 # NDCG@5 (общий и для подзадачи B)
│   ├── validation.py              # CV-сплиттеры
│   ├── subtasks.py                # фильтр подзадачи B по pil1mtrx_offer
│   ├── blending.py                # rank-averaging blend нескольких моделей
│   ├── inference.py               # формирование сабмита, two-stage сборка
│   ├── blends.py                  # зафиксированные составы топ-blend-ов
│   ├── features/                  # инженерия признаков
│   ├── models/                    # обёртки бустингов и MLP
│   ├── training.py                # обучающий цикл с MLflow
│   ├── tuning.py                  # Optuna для гиперпараметров
│   ├── tracking.py                # обёртки MLflow
│   └── utils.py                   # логгер, seed_everything
├── notebooks/
│   ├── 01_eda.ipynb               # EDA, базовая визуализация
│   └── EDA_FINDINGS.md            # выжимка по EDA
├── scripts/
│   ├── train.py                   # CV + MLflow-логирование по YAML-конфигу
│   ├── tune.py                    # Optuna для LightGBM
│   └── make_final_submission.py   # воспроизведение финального сабмита (LB 92.0504)
├── configs/                       # YAML-конфиги экспериментов
├── submissions/                   # сабмиты (gitignored)
├── oof/                           # OOF и test_scores (gitignored)
├── mlruns/                        # MLflow runs (gitignored)
├── EXPERIMENTS.md                 # лог Pipeline H-P
├── README.md                      # этот файл
├── requirements.txt
└── pyproject.toml
```

## Окружение

- Python **3.11** (pyenv-win), изолированный `.venv` в корне.
- Зависимости пинятся в `requirements.txt`. Сам пакет ставится
  редактируемым: `pip install -e .`.
- GPU — RTX 3060 Ti 8 GB. CatBoost и XGBoost запускались на GPU,
  LightGBM по умолчанию на CPU (GPU-сборка не давала прироста на
  наших размерах).
- Все эксперименты помещаются в 32 GB RAM при разумном downcast-е
  (int32/float32).

## Установка

```powershell
# PowerShell (Windows)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

```bash
# bash
python3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Данные положите в `data/`:
- `train_dataset_small.pq`
- `test_dataset_small.pq`
- `features_small.pq`
- `feature_description.csv`
- `commit.csv` (пример сабмита)

## Воспроизведение финального сабмита

Финальный two-stage сабмит собирается из заранее сохранённых
`*_test_scores.parquet` в `oof/` (test-скоры одиночных моделей).
Все 11 + 16 = 27 моделей перечислены в
[`src/alfa_cred/blends.py`](src/alfa_cred/blends.py).

```powershell
python scripts/make_final_submission.py
# или с диагностикой против эталона:
python scripts/make_final_submission.py \
    --verify-against submissions/two_stage_r11_bBalanced_plus_pseudo_crossobj_1405.csv
```

Скрипт даёт численно эквивалентный результат с финальным сабмитом
LB 92.0504 (расхождения на 8 строках из 936 883 на уровне ±1e-06 —
артефакт float-арифметики, на NDCG@5 это не влияет).

## Запуск экспериментов

```powershell
# обучение и CV по YAML-конфигу
python scripts/train.py --config configs/baseline_lgbm.yaml

# тюнинг LightGBM через Optuna
python scripts/tune.py \
    --config configs/lgbm_optuna.yaml \
    --out configs/lgbm_optuna_tuned.yaml \
    --n-trials 30

# MLflow UI для просмотра прогонов
mlflow ui --backend-store-uri mlruns/
```

## Ключевые инсайты

1. **Двухсегментная природа задачи** (через `pil1mtrx_offer`) — главный
   рычаг. EDA-эвристика «pil1mtrx первыми» уже даёт NDCG@5 ≈ 0.80 на
   train. Финальный прирост этого подхода через two-stage блендинг —
   +0.027 LB от плоского blend record_11.
2. **B-only обучение** на 34% запросов выигрывает у моделей, обученных
   на всём train, для подзадачи B. Видимо, потому что общая модель
   тратит ёмкость на структуру подзадачи A.
3. **Multi-seed одного и того же XGB** даёт +0.024 от одиночной модели.
   Bootstrap-вариативность гораздо устойчивее, чем глубокий Optuna-тюнинг.
4. **bBalanced (top-3 каждого типа архитектуры)** — оптимальный размер
   B-blend в 12-16 моделей. Больше моделей размывают результат, меньше
   теряют диверсификацию.
5. **Cross-objective диверсификация** (`xgb_pairwise`, `lgbm_xendcg`)
   и **pseudo-labeling** — добавляют ~+0.004 каждый в небольшой blend.
6. **CV не предсказывает LB точно**: модели с CV 0.916 и CV 0.918 могут
   меняться местами на LB. Доверяем LB как ground truth, а CV — только
   для отсечки моделей ниже эмпирической границы.

## Что не сработало

Подробности — в [`EXPERIMENTS.md`](EXPERIMENTS.md). Кратко:

- Tabular DL (FT-Transformer, TabNet) — слабее GBM в одиночку и
  «размывают» blend.
- Customer history features — мало пересечений `app_id` train/test.
- Per-epoch модели — CV завышен на сабсете, LB катастрофический.
- Глубокий Optuna для XGBoost — переобучение, LB хуже дефолтных
  параметров.
- Pairwise binary classification (tournament) — не работает на нашей
  постановке.
- Subgroup specialization по `offer_type` — только как ДОПОЛНЕНИЕ,
  не как замена.
- Hybrid 50/50 (полусмешивание record и B-only) — чистый replacement
  лучше.
- LB-weighted blend — uniform даёт стабильнее.
- AutoEncoder embeddings и MI scan на 280 фичах — «скрытого сигнала»
  не нашёл; топ-MI ~0.06 в шуме.

## Метрика

Реализация NDCG@5 в [`src/alfa_cred/metrics.py`](src/alfa_cred/metrics.py)
сверена с baseline-оценкой организаторов. Для B-only моделей есть
отдельный хелпер `compute_b_ndcg5`, отфильтровывающий запросы
подзадачи A (там NDCG@5 ≈ 1.0 за счёт hard-rule и она «скрывает»
реальный прирост на подзадаче B).
