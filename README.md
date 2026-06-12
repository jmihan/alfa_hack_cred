# Alfa Credit Offer Ranking

[![tests](https://github.com/jmihan/alfa_hack_cred/actions/workflows/tests.yml/badge.svg)](https://github.com/jmihan/alfa_hack_cred/actions/workflows/tests.yml)

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
(на LB не сабмитил). Простая эвристика «pil1mtrx_offer первыми +
variant_no asc» уже даёт NDCG@5 ≈ 0.80 на train.

| Этап | LB |
|------|----|
| LGBM LambdaRank baseline | 91.634 |
| LGBM extended features | 91.8648 |
| blend 11 моделей (record_11) | 91.9668 |
| two-stage прорыв (B-only XGB) | 91.9939 |
| two-stage + bBalanced (12 B-моделей) | 92.0317 |
| + фича `is_best_both` | 92.1359 |
| широкий offer-набор + 8-модельный B-бленд + pointwise-MLP | 92.1957 (public-рекорд) |
| **A=record_11 + B=bAllL (5 XGB Optuna + 1 LGBM)** | **92.0006 (public) → лучший на private** |

> **Public ≠ private.** Сложный public-рекорд (8-модельный B-бленд + pointwise-MLP,
> 92.1957) переобучился под публичную часть лидерборда — на приватной он просел.
> Устойчивее всего на private оказался **`two_stage_record11_plus_bAllL`**: A-сторона
> record_11, B-сторона — простой multi-seed XGB-ансамбль (5 XGBoost с одними Optuna-
> параметрами на разных сидах + 1 LightGBM extended) без NN-диверсити. Репозиторий
> воспроизводит именно его.

> Точное воспроизведение — режим **`reproduce`**: детерминированно (на любой машине,
> GPU не нужен, ~2 мин) собирает **байт-в-байт** лучший сабмит из зафиксированных
> предсказаний (`artifacts/record/`) и сверяет sha256. Режим **`train`** обучает
> близкий пайплайн **с полного нуля**. A-бленд record_11 (агрегат 11 моделей) переобучить байт-в-байт нельзя, поэтому его предсказания
> зафиксированы как артефакт; B-сторона bAllL обучается с нуля. NDCG@5 зависит только
> от порядка офферов.

Ключевые рычаги:

1. **Two-stage по `pil1mtrx_offer`** — задача распадается на A (есть pil1, hard-rule
   почти решает: такой оффер становится сделкой в 99.7%) и B (реальный LTR). B-модели
   обучаются ТОЛЬКО на B-заявках, не тратя ёмкость на структуру A.
2. **Multi-seed XGB-ансамбль** (bAllL): 5 XGBoost с одними Optuna-параметрами на разных
   сидах + 1 LightGBM extended. Простой и устойчивый — без NN-диверсити, которая
   переобучила public-рекорд.

> Фича `is_best_both` (оффер, точно совпавший с заявкой по сумме И сроку, deal-rate
> ≈52%) дала заметный буст на public **позже** (Pipeline T, 29 мая; см.
> [EXPERIMENTS.md](EXPERIMENTS.md)), но в приватном рекорде bAllL (25 мая) её ещё не
> было — поэтому в каноническом решении она не используется.

## Архитектура решения

EDA показала, что задача распадается на две неравные части по флагу `pil1mtrx_offer`.
Решение эксплуатирует это напрямую (two-stage):

```
                          test (936 883 строк, 38 742 запроса)
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                          │
       подзадача A (≈66% запросов)              подзадача B (≈34% запросов)
       в группе есть pil1mtrx_offer=1            pil1mtrx-флага нет
                  │                                          │
                  ▼                                          ▼
       A-бленд record_11 (11 моделей)           bAllL: rank-avg 5 XGBoost (Optuna)
       + hard-rule: pil1 → верх                 + 1 LightGBM extended,
       (для train с нуля — компактный            расширенный набор (тот же, что у A)
        5-модельный бленд)                                  │
                  │                                          │
                  └────────────────┬─────────────────────────┘
                                   │
                                   ▼
                       финальный сабмит (лучший на private)
```

**Подзадача A** решается hard-rule (`pil1mtrx_offer=1` → первое место; такой оффер
становится сделкой в 99.7% случаев) поверх rank-avg A-бленда record_11 (11 моделей
LightGBM/CatBoost/XGBoost разных objective и сидов). Для обучения с нуля A-сторона —
компактный 5-модельный бленд ([`models/a_blend.py`](src/alfa_cred/models/a_blend.py):
3×LightGBM extended + 2×CatBoost YetiRank на [`build_feature_table`](src/alfa_cred/features/pipeline.py)),
близкий к record_11.

**Подзадача B** (заявки без pil1-оффера) — реальная задача ранжирования. Скор —
rank-avg перцентильных рангов 6 GBDT ([`models/b_ball.py`](src/alfa_cred/models/b_ball.py))
на том же расширенном наборе ([`build_feature_table`](src/alfa_cred/features/pipeline.py)),
что и A-сторона, отфильтрованном на B-заявки:

- **5× XGBoost** (`rank:ndcg`, Optuna-параметры, `lambdarank_pair_method=topk`,
  сиды 42/137/314/8848/2026);
- **1× LightGBM extended** (LambdaRank, tuned-параметры).

Multi-seed XGB-ансамбль однороден по архитектуре, но устойчив: разные сиды сглаживают
дисперсию, а отсутствие NN-компонента уберегло от переобучения под public LB.

Сборка обеих стадий — в [`two_stage.py`](src/alfa_cred/two_stage.py). Три режима (см.
«Запуск в Docker»): `reproduce` — байт-в-байт лучший сабмит из зафиксированных
предсказаний; `train` — обучить пайплайн с нуля; `inference` — собрать из сохранённых
моделей.

Полная хронология экспериментов (все pipeline-ы, CV/LB каждого сабмита,
гиперпараметры, что сработало и что нет) — в [`EXPERIMENTS.md`](EXPERIMENTS.md).
EDA-выводы — в [`notebooks/EDA_FINDINGS.md`](notebooks/EDA_FINDINGS.md).

## Что конкретно в финальном решении

### Признаки

Единый расширенный feature-набор ([`build_feature_table`](src/alfa_cred/features/pipeline.py),
~200 фич; детерминирован из pre-decision атрибутов, не лик) — используется и A-блендом,
и B-стороной (для B отфильтрован на заявки без pil1):

- внутригрупповые ранги / z-scores / агрегаты по `request_id`
  ([`features/group.py`](src/alfa_cred/features/group.py)), ранги внутри подгрупп
  `offer_type`/`risk_level_map`, кросс-фичи оффер×заявка, basket multi-hot
  ([`features/basket.py`](src/alfa_cred/features/basket.py)), временные из
  `request_received` ([`features/time.py`](src/alfa_cred/features/time.py)),
  клиентские из `features_small.pq` с фильтром по заполненности
  ([`features/client.py`](src/alfa_cred/features/client.py)), Парето-доминирование.

> Фича `is_best_both` и ask-match стек ([`features/match.py`](src/alfa_cred/features/match.py))
> в репозитории есть, но в каноническом (приватном) решении **не используются**:
> рекордный bAllL (25 мая) сформирован до их появления (Pipeline T, 29 мая). Это поздний
> public-эксперимент (deal-rate `is_best_both` ≈52%), описан в [EXPERIMENTS.md](EXPERIMENTS.md).

### Модели и гиперпараметры

**A-бленд** (5 моделей, rank-avg перцентильных рангов; [`models/a_blend.py`](src/alfa_cred/models/a_blend.py)):

- 3× LightGBM LambdaRank (сиды 42/123/777): `learning_rate=0.0138`, `num_leaves=374`,
  `min_data_in_leaf=388`, `feature_fraction=0.78`, `bagging_fraction=0.89`,
  `bagging_freq=5`, `lambda_l1=5e-4`, `lambda_l2=7e-5`, `min_gain_to_split=0.013`,
  `n_estimators=2500`.
- 2× CatBoost YetiRank (сиды 42/123): `learning_rate=0.05`, `depth=6`,
  `l2_leaf_reg=3.0`, `bootstrap_type=Bernoulli`, `subsample=0.85`, `iterations=1000`.

**B-сторона bAllL** (6 моделей, rank-avg перцентильных рангов; [`models/b_ball.py`](src/alfa_cred/models/b_ball.py)):

- 5× XGBoost `rank:ndcg` (сиды 42/137/314/8848/2026), Optuna-параметры:
  `learning_rate=0.0330`, `max_depth=7`, `min_child_weight=20.0`, `subsample=0.882`,
  `colsample_bytree=0.534`, `reg_alpha=0.164`, `reg_lambda=0.0059`,
  `lambdarank_pair_method=topk`, `lambdarank_num_pair_per_sample=8`.
- 1× LightGBM extended (LambdaRank, сид 42): те же tuned-параметры, что у A-extended
  (`learning_rate=0.0138`, `num_leaves=374`, `min_data_in_leaf=388`, …).

Число итераций каждой модели подбирается **ранней остановкой** по B-NDCG@5 на групповом
холдауте (`n_estimators` до 4000, early stopping 150), затем модель **рефитится на всём
B-train** этим числом итераций — так inference из сохранённой модели совпадает с train
байт-в-байт.

**Сборка:** B-сторона — rank-avg перцентильных рангов 6 моделей. A-сторона — rank-avg
A-бленда + hard-rule (`pil1mtrx=1` → +1.0, гарантирует первое место).

### Подходы, соображения, инструменты

- **Two-stage по `pil1mtrx_offer`** — главный рычаг: задача распадается на A (есть
  pil1, hard-rule почти решает) и B (реальный LTR). B-модели обучаются ТОЛЬКО на
  B-заявках — не тратят ёмкость на структуру A.
- **Rank-averaging** (перцентильные ранги внутри заявки) — устойчив к разным шкалам
  моделей, лучше усреднения сырых скоров.
- **Multi-seed XGB-ансамбль** (bAllL): один Optuna-набор параметров на 5 разных сидах
  + 1 LightGBM. Разброс по сидам сглаживает дисперсию, а простота уберегла от
  переобучения под public LB.
- **Public ≠ private** — ключевой урок: сложный public-рекорд (8-модельный B-бленд +
  pointwise-MLP, +0.018 на public от NN-диверсити) на приватном лидерборде **просел**,
  а простой bAllL оказался устойчивее. Публичный LB и CV не предсказывают private
  точно; простые устойчивые ансамбли надёжнее «подогнанных под public» сложных.
- **Детерминизм**: фиксированные сиды всех моделей.
- **Инструменты**: LightGBM 4.4, XGBoost 2.0, CatBoost 1.2, MLflow (трекинг
  CV-экспериментов), Optuna (тюнинг), SHAP (интерпретация), Docker (воспроизведение).

## Структура репозитория

```
.
├── data/                          # исходные данные (gitignored)
├── task/                          # условие задачи (gitignored)
├── src/alfa_cred/                 # переиспользуемый код
│   ├── config.py                  # пути и константы
│   ├── io_utils.py                # загрузка/мёрж parquet, кодирование cat-фич
│   ├── metrics.py                 # NDCG@5
│   ├── validation.py              # CV-сплиттеры (GroupKFold / time-split)
│   ├── inference.py               # формирование сабмита + hard-rule
│   ├── two_stage.py               # подготовка признаков A/B + сборка сабмита (train+inference)
│   ├── interpret.py               # SHAP + group-aware permutation важность
│   ├── features/                  # инженерия признаков
│   │   ├── basic.py / group.py    # базовые и внутригрупповые признаки оффера
│   │   ├── match.py               # ask-match признаки + is_best_both + Парето
│   │   └── pipeline.py            # сборка feature-таблиц (build_feature_table / build_wide_feature_table)
│   ├── models/                    # модели
│   │   ├── a_blend.py             # A-бленд (3×LGBM + 2×CatBoost)
│   │   └── b_ball.py              # B-сторона bAllL (5×XGBoost Optuna + 1×LGBM extended)
│   ├── training.py                # обучающий цикл с MLflow (эксперименты)
│   ├── tuning.py                  # Optuna для гиперпараметров
│   ├── tracking.py                # обёртки MLflow
│   └── utils.py                   # логгер, seed_everything
├── notebooks/
│   ├── 01_eda.ipynb               # EDA, базовая визуализация
│   ├── EDA_FINDINGS.md            # выжимка по EDA
│   └── INTERPRETATION.md          # выводы по интерпретации модели
├── scripts/
│   ├── reproduce_record.py        # REPRODUCE: байт-в-байт лучший сабмит из artifacts/record
│   ├── fit_pipeline.py            # TRAIN: обучить пайплайн с нуля + сохранить модели
│   ├── predict.py                 # INFERENCE: сабмит из сохранённых моделей
│   ├── train.py                   # CV одиночной модели по YAML-конфигу (эксперименты)
│   ├── tune.py                    # Optuna для LightGBM
│   ├── explain.py                 # SHAP + permutation интерпретация B-модели
│   └── verify_submission.py       # сверка двух сабмитов по ранжированию (top-1)
├── tests/                         # юнит-тесты: NDCG@5, формат сабмита, ask-match
├── artifacts/record/              # зафиксированные предсказания рекорда (для reproduce, ~9 МБ, в git)
├── docs/img/                      # графики интерпретации для README
├── configs/                       # YAML-конфиги экспериментов
├── models/                        # обученные модели (gitignored; train -> inference)
├── submissions/                   # сабмиты (gitignored)
├── oof/                           # OOF и test_scores (gitignored)
├── mlruns/                        # MLflow runs (gitignored)
├── Dockerfile                     # CPU-образ (reproduce/inference/train/cv/interpret)
├── Dockerfile.gpu                 # GPU-образ для train-gpu (XGBoost/CatBoost на GPU)
├── docker-compose.yml             # режимы запуска
├── run.sh                         # единая точка запуска (reproduce/train/.../test)
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
  таких размерах данных).
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

Быстрый старт одной командой (соберёт образ и воспроизведёт рекорд):

```bash
./run.sh                 # = собрать образ + reproduce (лучший сабмит); см. ./run.sh train|inference|interpret|test
```

Под капотом — CPU-образ ([`Dockerfile`](Dockerfile)): Python 3.11 + зависимости + пакет
(`pip install -e .`) и зафиксированные предсказания рекорда (`artifacts/record`) внутри.
Данные монтируются как volume (`./data`, `./models`, `./submissions`, ...). У каждого
режима свой профиль, образ собираем обычным `docker build`, режимы запускаем через
`run --rm` (см. [`docker-compose.yml`](docker-compose.yml)):

```powershell
docker build -t alfa-cred:latest .       # собрать образ (за VPN: добавьте --network=host)

docker compose run --rm reproduce        # БАЙТ-В-БАЙТ лучший сабмит из artifacts/record (~2 мин, GPU не нужен)
docker compose run --rm train            # обучить пайплайн с нуля + сохранить ./models + сабмит
docker compose run --rm inference        # быстрая пересборка сабмита из ./models (без обучения)

docker compose run --rm cv               # CV одиночной модели по YAML-конфигу (эксперименты, логи в ./mlruns)
docker compose run --rm interpret        # SHAP + permutation (графики/CSV в ./reports/interpretation)
docker compose --profile mlflow up       # MLflow UI -> открыть http://localhost:5000 (только UI; Ctrl+C — остановить)
```

Организаторам достаточно положить данные в `./data` и собрать образ. Для **точного
лучшего сабмита** — `docker compose run --rm reproduce` (детерминированно, на любой
машине, байт-в-байт, со сверкой sha256). Для **обучения с нуля** — `docker compose run
--rm train`. Оба пишут сабмит в `./submissions/record_submission.csv`. Сабмит пишется с
фиксированным переводом строки (CRLF), поэтому байты совпадают на Windows и Linux —
`reproduce` даёт одинаковый sha везде.

**Время по этапам** (Ryzen 5 7500F, CPU; ориентир):

| Режим | Время |
|-------|-------|
| `reproduce` (сборка из артефактов + sha256) | ≈ 2 мин |
| `train` (A-бленд 5 + bAllL 6, с нуля) | ≈ 55 мин (из них A-бленд ≈ 45 мин) |
| `inference` (загрузка моделей + сборка сабмита) | ≈ 5 мин |
| `cv` (одна LightGBM, 5-fold) | ≈ 25 мин |
| `interpret` (обучение reference-модели + SHAP + permutation) | ≈ 2 мин |

> MLflow UI открывать по `http://localhost:5000` (не `0.0.0.0` — внутри контейнера
> сервер слушает `0.0.0.0:5000`, а наружу проброшен на localhost хоста).

**GPU (опционально, для `train`).** Отдельный образ [`Dockerfile.gpu`](Dockerfile.gpu)
и сервис `train-gpu` обучают на GPU XGBoost (`device='cuda'`) и CatBoost
(`task_type='GPU'`); LightGBM остаётся на CPU (как весь хакатон). Нужен
nvidia-container-toolkit на хосте:

```powershell
docker compose --profile gpu build train-gpu
docker compose --profile gpu run --rm train-gpu
```

GPU-обучение деревьев даёт другие числа, чем CPU, поэтому `train-gpu` тоже не
байт-в-байт; точный сабмит — только `reproduce`.

**За корпоративным VPN/файрволом.** Если сборка падает с SSL-ошибкой при доступе
к pypi (`SSL: UNEXPECTED_EOF_WHILE_READING`), соберите образ через host-сеть и
дальше запускайте сервисы как обычно (compose возьмёт готовый `alfa-cred:latest`):

```powershell
docker build --network=host -t alfa-cred:latest .
docker compose run --rm reproduce
```

## Воспроизведение сабмитов

### Точный лучший сабмит — байт-в-байт

`reproduce_record.py` собирает лучший (на приватном лидерборде) сабмит
`two_stage_record11_plus_bAllL` из зафиксированных предсказаний в
[`artifacts/record/`](artifacts/record) и сверяет sha256 с эталоном:

```powershell
python scripts/reproduce_record.py --out submissions/record_submission.csv
```

Состав (двухстадийно по `pil1mtrx_offer`): A-сторона — ранжирование A-бленда `record_11`
(11 моделей, агрегат большого числа прогонов); B-сторона `bAllL` — rank-avg 5 XGBoost
(Optuna) + 1 LightGBM extended. Порядок строк берётся из самих данных (`load_raw` →
`sort_by_request`), поэтому сабмит маппится на актуальный test. Детерминированно на
любой машине (GPU не нужен). A-база `record_11` (агрегат разовых прогонов) и точные
числа multi-seed B-ансамбля невоспроизводимы побайтно переобучением на другом железе,
поэтому их предсказания на test зафиксированы как артефакты (OOF, ~4 МБ в git):

- `artifacts/record/a_side_record11.parquet` — A-сторона record_11;
- `artifacts/record/b_side_ball.parquet` — B-сторона bAllL;
- `artifacts/record/manifest.json` — эталонный sha256.

Эти OOF-файлы уже в репозитории — отдельно прикладывать их не нужно.

### Обучение с нуля

Близкий к рекорду пайплайн, обучаемый полностью с нуля; разделён на два шага:

```powershell
# TRAIN: обучить все модели (A-бленд 5 + bAllL 6) и сохранить в ./models (~55 мин)
python scripts/fit_pipeline.py --out submissions/record_submission.csv

# INFERENCE: быстрая пересборка сабмита из сохранённых моделей (без обучения)
python scripts/predict.py --out submissions/record_submission.csv
```

`fit_pipeline.py` обучает A-бленд (3×LGBM + 2×CatBoost) и B-сторону bAllL (5×XGBoost
Optuna + 1×LightGBM extended), сохраняет их в `models/` и пишет сабмит. Дальше
`predict.py` собирает сабмит из этих моделей без переобучения. Главный приём подзадачи
B — multi-seed XGB-ансамбль на расширенном наборе (rank-avg по сидам). На GPU
(XGBoost/CatBoost) — через `--device cuda` или образ `Dockerfile.gpu`.

> NDCG@5 зависит только от порядка офферов. Обучение с нуля воспроизводит ранжирование
> рекорда близко (на локальной машине: top-1 совпадение B-стороны ≈96%, Spearman ≈0.99), но
> НЕ байт-в-байт (multi-seed GBDT между прогонами/машинами не побитово идентичны). Для
> точного лучшего сабмита используйте `reproduce`.

### Готовые веса (опционально)

Чтобы пропустить 55-минутное обучение и сразу собрать сабмит через `inference`, можно
скачать предобученные модели (A-бленд 5 + bAllL 6) и распаковать их в `./models/`:

- **Скачать (Google Drive):** [архив с моделями](https://drive.google.com/file/d/1Jm1ByonwI1L4IWX6ARSgOrGSGwd7cCMB/view?usp=sharing)
- Распаковать архив в `./models/` (файлы `a_*`, `b_*` и манифесты `a_manifest.json`/`b_manifest.json`).
- Собрать сабмит: `docker compose run --rm inference` (или `python scripts/predict.py`).

Веса дают результат обучения с нуля (близко к рекорду). Для **точного лучшего сабмита**
готовые веса не нужны — он воспроизводится из `artifacts/record/` (уже в репозитории)
режимом `reproduce`.

Сверка ранжирования двух сабмитов (например train- и reproduce-сборки):

```powershell
python scripts/predict.py --out submissions/check.csv
python scripts/verify_submission.py submissions/check.csv submissions/record_submission.csv
```

## Интерпретация модели

Драйверы скоринга подзадачи B разбираются двумя взаимодополняющими методами —
**SHAP TreeExplainer** (аддитивные вклады в скор оффера) и **group-aware
permutation importance на NDCG@5** (модель-агностично, устойчиво к коррелированным
признакам). Объясняется **B-модель**: одиночная LightGBM с теми же гиперпараметрами
(`LGB_PARAMS`), сидом (42) и расширенным набором (B-only), что и LightGBM-член bAllL
(SHAP применим к одному дереву, а не к rank-avg 6 моделей; сигнал у всех членов общий).
Реализация — [`src/alfa_cred/interpret.py`](src/alfa_cred/interpret.py).

```powershell
python scripts/explain.py            # или: docker compose run --rm interpret
```

**Главные драйверы** (падение NDCG@5 при перемешивании признака, valid-сплит B):

| Признак | Падение NDCG@5 |
|---------|----------------|
| `basket_present` | 0.0669 |
| `req_to_limit_ratio` | 0.0375 |
| `req_minus_term` | 0.0343 |
| `limit_ratio_to_max` | 0.0301 |
| `variant_no` | 0.0244 |
| `req_minus_limit` | 0.0185 |
| `offer_type` | 0.0159 |

**Вывод:** ранжирование в подзадаче B определяют (1) **соответствие оффера заявке** —
`req_to_limit_ratio`, `req_minus_term`, `req_minus_limit` (насколько сумма и срок оффера
близки к запрошенным), (2) **наличие корзины товаров** (`basket_present`), (3) **позиция**
оффера (`variant_no`) и (4) **относительная величина лимита** (`limit_ratio_to_max`,
ранги внутри типа). Сигнал «оффер совпал с заявкой» модель извлекает из сырых разниц —
без явного композита `is_best_both` (в bAllL его не было; он дал буст на public позже,
см. [EXPERIMENTS.md](EXPERIMENTS.md)).

![Permutation importance по NDCG@5](docs/img/permutation_importance_ndcg5.png)

SHAP beeswarm (направление и величина вклада признаков в скор оффера):

![SHAP beeswarm](docs/img/shap_beeswarm.png)

Локальное объяснение (waterfall) одного оффера, ставшего сделкой — видно, как признаки
складываются в итоговый скор для конкретного оффера:

![SHAP waterfall](docs/img/shap_waterfall.png)

Полный набор графиков (beeswarm, bar, dependence по топ-признаку, локальные waterfall)
и CSV-таблицы пишутся в `reports/interpretation/` при запуске `explain.py`.

## Тесты

Юнит-тесты ([`tests/`](tests)) проверяют то, что должно быть верно при любых
изменениях: корректность метрики **NDCG@5** (по ней принимаются все решения), формат
и побайтную детерминированность сабмита (CRLF, округление до 6 знаков, сверка ключей)
и логику ask-match фичи **`is_best_both`** (есть в репозитории как public-эксперимент).
Данные хакатона не нужны — тесты на синтетике, запускаются где угодно.

```powershell
pytest            # или: ./run.sh test
```

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
7. **Public ≠ private — простота устойчивее.** Надстройки, тюненные под public
   (фича `is_best_both` +0.0063 B-NDCG@5 → LB 92.18; pointwise-MLP диверсити → 92.1957),
   на **приватном** лидерборде просели. Лучшим на private оказался простой bAllL
   (record_11 + multi-seed XGB) без них. Стоило сильнее доверять простоте и
   устойчивости, чем приросту на public.

## Что не сработало

Подробности — в [`EXPERIMENTS.md`](EXPERIMENTS.md). Кратко:

- Tabular DL (FT-Transformer, TabNet) — слабее GBM в одиночку и
  «размывают» blend.
- Customer history features — мало пересечений `app_id` train/test.
- Per-epoch модели — CV завышен на сабсете, LB катастрофический.
- Глубокий Optuna для XGBoost — переобучение, LB хуже дефолтных
  параметров.
- Pairwise binary classification (tournament) — не работает в этой
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
