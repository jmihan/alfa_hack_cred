# Alfa Credit Offer Ranking

Решение задачи **«Прогноз кредита по клиентам»** хакатона Альфа-Банка
(Яндекс Контест). Цель — ранжировать кредитные предложения внутри каждого
запроса клиента так, чтобы максимизировать метрику **NDCG@5**.

## Постановка задачи

Для каждого запроса `request_id` дано до 50 кредитных предложений
(`variant_no`). Требуется построить модель, выдающую `score` для каждой
пары `(request_id, variant_no)`, по которой предложения внутри запроса
ранжируются. Качество оценивается по среднему NDCG@5.

Формат сабмита (`commit.csv`, разделитель `;`):

```
request_id;variant_no;score
0;10;0.016394
...
```

## Структура репозитория

```
.
├── data/                       # исходные данные (gitignored)
├── task/                       # условие задачи (gitignored)
├── src/alfa_cred/              # переиспользуемый код
│   ├── config.py               # пути и константы
│   ├── io_utils.py             # загрузка/мёрж parquet
│   ├── metrics.py              # NDCG@5
│   ├── validation.py           # CV-сплиттеры
│   ├── features/               # инженерия признаков
│   ├── models/                 # бустинги и нейросети
│   ├── training.py             # обучающий цикл с MLflow
│   ├── inference.py            # формирование сабмита
│   ├── tracking.py             # обёртки MLflow
│   └── utils.py                # утилиты (логгер, seed)
├── notebooks/                  # ноутбуки EDA и экспериментов (.py с # %%)
├── scripts/                    # точки входа CLI
├── configs/                    # YAML-конфиги экспериментов
├── submissions/                # сабмиты (gitignored)
└── mlruns/                     # MLflow runs (gitignored)
```

## Запуск

### Подготовка окружения

Используется Python **3.11** и изолированное виртуальное окружение `.venv`.

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

Данные хакатона положите в каталог `data/`:
- `train_dataset_small.pq`
- `test_dataset_small.pq`
- `features_small.pq`
- `feature_description.csv`
- `commit.csv` (пример сабмита)

### EDA

```powershell
# Открыть ноутбук в Jupyter и выполнить все ячейки
jupyter notebook notebooks/01_eda.ipynb
```

Исходник EDA лежит в `notebooks/01_eda.py` (формат `# %%` ячеек) — при
необходимости конвертируется в `.ipynb`.

### Обучение и формирование сабмита

```powershell
python scripts/train.py --config configs/baseline_lgbm.yaml
python scripts/make_submission.py --config configs/baseline_lgbm.yaml
```

### Трекинг экспериментов

```powershell
mlflow ui --backend-store-uri mlruns/
```

## План работы

1. **EDA** — изучение распределений, временного разреза, position bias по
   `variant_no`, drift между train/test.
2. **Foundation** — реализация метрики NDCG@5, CV-сплиттера, LightGBM
   LambdaRank baseline.
3. **Эксперименты** — feature engineering (внутригрупповые ранги,
   агрегаты, target encoding), тюнинг гиперпараметров, ансамбли,
   нейросетевые ранкеры.
4. **Финал** — блендинг лучших моделей, оформление решения.

## Метрика

Базовый RandomForest даёт **NDCG@5 ≈ 0.57**. Целевая планка — заметно
выше за счёт LTR-моделей и инженерии признаков.
