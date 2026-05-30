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
| two-stage + bBalanced + pseudo + crossobj | 92.0504 |
| + фича `is_best_both` (9-модельный B-бленд) | 92.1359 |
| широкий offer-набор + 8-модельный B-бленд | ≈ 92.18 |
| + pointwise-MLP диверсити в B-бленд (`0.70·b_blend + 0.30·MLP`) | **≈ 92.196** ← финал |

Два рычага финальной фазы:

1. Фича **`is_best_both`** — оффер, точно совпадающий с заявкой клиента
   (`limit == req_loan_amount` И `term == req_term`) и первый среди таких по
   `variant_no`. Deal-rate ≈ **52%** против ≈2.8% — «мягкий» hard-rule для подзадачи B.
2. **Pointwise-MLP** как ортогональная диверсити к GBDT-бленду. Бустинги в B-бленде
   архитектурно близки; нейросеть `P(is_deal)` (Spearman ≈ 0.84 с `b_blend`) даёт
   диверсити другой природы — её добавление (вес 0.30) дало **+0.018 LB**.

## Архитектура решения

EDA показала, что задача распадается на две неравные части по флагу
`pil1mtrx_offer`. Рекордное решение эксплуатирует это напрямую (two-stage):

```
                          test (936 883 строк, 38 742 запроса)
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
       подзадача A (≈66% запросов)              подзадача B (≈34% запросов)
       в группе есть pil1mtrx_offer=1            pil1mtrx-флага нет
                  │                                         │
                  ▼                                         ▼
       5-модельный A-бленд                      0.70·b_blend + 0.30·pointwise-MLP
       (3×LGBM + 2×CatBoost)                    b_blend: 8 GBDT (LGBM×3+XGB×3+CB×2),
       + hard-rule: pil1 → верх                 MLP: нейросеть P(is_deal) (диверсити),
                  │                              широкий offer-набор (361 фича)
                  │                                         │
                  └────────────────┬────────────────────────┘
                                   │
                                   ▼
                              финальный сабмит
                                LB ≈ 92.196
```

**Подзадача A** решается hard-rule (`pil1mtrx_offer=1` → первое место; такой
оффер становится сделкой в 99.7% случаев) поверх rank-avg 5-модельного A-бленда
([`models/a_blend.py`](src/alfa_cred/models/a_blend.py)): 3×LightGBM extended +
2×CatBoost YetiRank на расширенном наборе ([`build_feature_table`](src/alfa_cred/features/pipeline.py)).

**Подзадача B** (заявки без pil1-оффера) — реальная задача ранжирования. Её скор —
смесь `0.70·b_blend + 0.30·MLP` (перцентильные ранги):

- **`b_blend`** ([`models/b_blend.py`](src/alfa_cred/models/b_blend.py)) — 8 GBDT:
  LightGBM LambdaRank ×3 + XGBoost rank:ndcg ×3 + CatBoost YetiRank ×2 на широком
  offer-наборе ([`build_wide_feature_table`](src/alfa_cred/features/pipeline.py)):
  внутригрупповые ранги (см. [`features/group.py`](src/alfa_cred/features/group.py)),
  кросс-офферные сравнения, Парето, ask-match стек с **`is_best_both`**
  ([`features/match.py`](src/alfa_cred/features/match.py)).
- **pointwise-MLP** ([`models/mlp_pointwise.py`](src/alfa_cred/models/mlp_pointwise.py)) —
  детерминированная нейросеть `P(is_deal)` (эмбеддинги категориальных + числовые).
  Ортогональна деревьям (Spearman ≈ 0.84 с `b_blend`), поэтому добавляет диверсити
  другой природы.

Сборка обеих стадий — в [`two_stage.py`](src/alfa_cred/two_stage.py). Весь пайплайн
обучается с нуля; разделён на режимы train (обучить + сохранить модели) и inference
(собрать сабмит из сохранённых) — см. «Запуск в Docker».

> Прежняя стабильная веха — LB 92.0504 (B-бленд из 16 моделей: bBalanced +
> pseudo-labeling + cross-objective). Её сборка осталась в
> [`scripts/make_final_submission.py`](scripts/make_final_submission.py).

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
│   ├── two_stage.py               # подготовка признаков A/B + сборка сабмита (train+inference)
│   ├── blends.py                  # зафиксированные составы топ-blend-ов
│   ├── interpret.py               # SHAP + group-aware permutation важность
│   ├── features/                  # инженерия признаков
│   │   ├── basic.py / group.py    # базовые и внутригрупповые признаки оффера
│   │   ├── match.py               # ask-match признаки + is_best_both + Парето
│   │   └── pipeline.py            # сборка feature-таблиц (build_feature_table / build_wide_feature_table)
│   ├── models/                    # модели
│   │   ├── a_blend.py             # A-бленд (3×LGBM + 2×CatBoost)
│   │   ├── b_blend.py             # 8-модельный B-бленд (LGB×3+XGB×3+CB×2)
│   │   └── mlp_pointwise.py       # детерминированный pointwise-MLP (диверсити B)
│   ├── training.py                # обучающий цикл с MLflow (эксперименты)
│   ├── tuning.py                  # Optuna для гиперпараметров
│   ├── tracking.py                # обёртки MLflow
│   └── utils.py                   # логгер, seed_everything
├── notebooks/
│   ├── 01_eda.ipynb               # EDA, базовая визуализация
│   ├── EDA_FINDINGS.md            # выжимка по EDA
│   └── INTERPRETATION.md          # выводы по интерпретации модели
├── scripts/
│   ├── fit_pipeline.py            # TRAIN: обучить весь пайплайн с нуля + сохранить модели (LB ≈ 92.196)
│   ├── predict.py                 # INFERENCE: сабмит из сохранённых моделей
│   ├── train.py                   # CV одиночной модели по YAML-конфигу (эксперименты)
│   ├── tune.py                    # Optuna для LightGBM
│   ├── explain.py                 # SHAP + permutation интерпретация B-модели
│   ├── make_final_submission.py   # воспроизведение сабмита LB 92.0504
│   └── verify_submission.py       # сверка сабмита с эталоном (top-1)
├── configs/                       # YAML-конфиги экспериментов
├── models/                        # обученные модели (gitignored; train -> inference)
├── submissions/                   # сабмиты (gitignored)
├── oof/                           # OOF и test_scores (gitignored)
├── mlruns/                        # MLflow runs (gitignored)
├── EXPERIMENTS.md                 # лог экспериментов
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

## Запуск в Docker

Образ ([`Dockerfile`](Dockerfile)) собирает Python 3.11 + зависимости + пакет
(`pip install -e .`, torch — CPU-сборка). Данные и артефакты внутрь образа не
кладутся — монтируются как volume (`./data`, `./models`, `./submissions`,
`./oof`, ...). Режимы — через профили [`docker-compose.yml`](docker-compose.yml):

```powershell
docker compose build                     # собрать образ alfa-cred:latest

# полное воспроизведение рекорда с нуля (нужен только ./data):
docker compose run --rm train            # обучить весь пайплайн + сохранить ./models + сабмит (~50 мин)
docker compose run --rm inference        # быстрая пересборка сабмита из ./models (без обучения, ~10 мин)

docker compose run --rm cv               # CV одиночной модели по YAML-конфигу (эксперименты, логи в ./mlruns)
docker compose run --rm interpret        # SHAP + permutation (графики в ./reports/interpretation)
docker compose --profile mlflow up       # MLflow UI -> http://localhost:5000
docker compose --profile notebook up     # Jupyter -> http://localhost:8888
```

Организаторам для воспроизведения рекорда достаточно положить данные в `./data`,
собрать образ и запустить `docker compose run --rm train` — он обучит все модели
с нуля и запишет сабмит в `./submissions/record_submission.csv`.

**GPU (опционально).** Рекорд считается на CPU, поэтому образ по умолчанию
CPU-only. Для GPU соберите образ на CUDA-базе, раскомментируйте `deploy.resources`
в `docker-compose.yml` и запускайте с `--gpus all`; CatBoost/XGBoost тогда
переключаются на GPU параметрами (`task_type='GPU'`, `device='cuda'`), LightGBM
остаётся на CPU. Без GPU всё работает как есть (CPU-fallback).

**За корпоративным VPN/файрволом.** Если сборка падает с SSL-ошибкой при доступе
к pypi (`SSL: UNEXPECTED_EOF_WHILE_READING`), соберите образ через host-сеть и
дальше запускайте сервисы как обычно (compose возьмёт готовый `alfa-cred:latest`):

```powershell
docker build --network=host -t alfa-cred:latest .
docker compose run --rm inference
```

## Воспроизведение сабмитов

### Рекордный сабмит (LB ≈ 92.196), обучение с нуля

Пайплайн обучается полностью с нуля и разделён на два шага:

```powershell
# TRAIN: обучить все модели (A-бленд 5 + b_blend 8 + MLP 3) и сохранить в ./models (~50 мин)
python scripts/fit_pipeline.py --out submissions/record_submission.csv

# INFERENCE: быстрая пересборка сабмита из сохранённых моделей (без обучения)
python scripts/predict.py --out submissions/record_submission.csv
```

`fit_pipeline.py` обучает A-бленд (3×LGBM + 2×CatBoost), B-бленд (LGB×3+XGB×3+CB×2)
и pointwise-MLP, сохраняет их в `models/` и пишет сабмит. Дальше `predict.py`
собирает сабмит из этих моделей без переобучения. Главный сигнал подзадачи B —
фича `is_best_both` (deal-rate ≈ 52%) + ортогональная MLP-диверсити.

> NDCG@5 зависит только от порядка офферов, поэтому с нуля воспроизводится
> LB-скор; побитовое равенство файла между разными машинами не гарантируется
> (многопоточность GBDT/NN), но порядок (а значит и метрика) устойчив.

Сверка ранжирования двух сабмитов (например train- и inference-сборки):

```powershell
python scripts/predict.py --out submissions/check.csv
python scripts/verify_submission.py submissions/check.csv submissions/record_submission.csv
```

### Прежняя веха (LB 92.0504)

Сабмит на 16-модельном B-бленде собирается из заранее сохранённых
`*_test_scores.parquet` в `oof/` (test-скоры одиночных моделей). Все 11 + 16 = 27
моделей перечислены в [`src/alfa_cred/blends.py`](src/alfa_cred/blends.py).

```powershell
python scripts/make_final_submission.py
# или с диагностикой против эталона:
python scripts/make_final_submission.py \
    --verify-against submissions/two_stage_r11_bBalanced_plus_pseudo_crossobj_1405.csv
```

Даёт численно эквивалентный результат с сабмитом LB 92.0504 (расхождения на
8 строках из 936 883 на уровне ±1e-06 — артефакт float-арифметики).

## Интерпретация модели

Драйверы скоринга подзадачи B разбираются двумя взаимодополняющими методами —
**SHAP TreeExplainer** (аддитивные вклады в скор оффера) и **group-aware
permutation importance на NDCG@5** (модель-агностично, устойчиво к коррелированным
признакам). Реализация — [`src/alfa_cred/interpret.py`](src/alfa_cred/interpret.py),
выводы — [`notebooks/INTERPRETATION.md`](notebooks/INTERPRETATION.md).

```powershell
python scripts/explain.py   # таблицы + графики в reports/interpretation/
```

Главный вывод: `is_best_both` — **top-3 драйвер NDCG@5** по permutation (хотя по
среднему |SHAP| лишь 14-й — признак разреженный, ≈2.4% офферов, поэтому средняя
магнитуда размывается). Это подтверждает абляцию (+0.0063 B-NDCG@5, LB 92.05 →
92.18): фича несёт уникальный, неперекрываемый сигнал.

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
7. **`is_best_both` — главный сигнал подзадачи B.** Композит «точное
   совпадение суммы И срока И первый по `variant_no`» GBDT не строит дёшево
   из непрерывных разниц — его нужно подавать явной фичей. Контролируемая
   абляция в полном бленде: +0.0063 B-NDCG@5; на LB — рывок 92.05 → 92.18.
   Эффект значим только в бленде, не в одиночной модели.

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
