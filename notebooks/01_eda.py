# %% [markdown]
# # 01 — Разведочный анализ данных (EDA)
#
# **Задача:** хакатон Альфа-Банка, ранжирование кредитных предложений
# по метрике NDCG@5. Для каждого `request_id` дан список вариантов
# `variant_no`, требуется отсортировать их так, чтобы реально купленное
# предложение (`is_deal=1`) оказалось как можно выше в топе.
#
# **Цели этого ноутбука:**
# 1. Понять схему всех трёх таблиц (`train`, `test`, `features`).
# 2. Изучить распределение таргета, число вариантов на запрос.
# 3. Проверить временной разрез — выбрать схему валидации (GroupKFold / time-based).
# 4. Найти потенциальный position bias по `variant_no`.
# 5. Оценить drift между train и test (adversarial validation).
# 6. Сформировать список наблюдений для построения пайплайна моделирования.
#
# **Как пользоваться:** каждая ячейка отделена `# %%`. Скопируйте код
# в Jupyter, выполните все ячейки. В конце ноутбук печатает текстовую
# сводку и сохраняет ключевые числа в `data/eda_summary.json`.

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("../data")
OUT_SUMMARY = DATA_DIR / "eda_summary.json"

TARGET = "is_deal"
REQUEST_ID = "request_id"
VARIANT_ID = "variant_no"

# Глобальные настройки отображения
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 100

SUMMARY: dict[str, object] = {}
print("Окружение настроено. Текущая директория:", Path.cwd())

# %% [markdown]
# ## 1. Загрузка датасетов и общая шейп-сводка

# %%
df_train = pd.read_parquet(DATA_DIR / "train_dataset_small.pq")
df_test = pd.read_parquet(DATA_DIR / "test_dataset_small.pq")
df_features = pd.read_parquet(DATA_DIR / "features_small.pq")
feature_desc = pd.read_csv(DATA_DIR / "feature_description.csv")

shapes = {
    "train": df_train.shape,
    "test": df_test.shape,
    "features": df_features.shape,
}
mem = {
    "train_MB": df_train.memory_usage(deep=True).sum() / 1024 ** 2,
    "test_MB": df_test.memory_usage(deep=True).sum() / 1024 ** 2,
    "features_MB": df_features.memory_usage(deep=True).sum() / 1024 ** 2,
}

print("Размеры таблиц:")
for k, v in shapes.items():
    print(f"  {k:10s} → rows={v[0]:>10,}  cols={v[1]}")
print("\nЗатраты памяти:")
for k, v in mem.items():
    print(f"  {k:14s} → {v:.1f} MB")

SUMMARY["shapes"] = {k: list(v) for k, v in shapes.items()}
SUMMARY["memory_MB"] = mem

# %% [markdown]
# ## 2. Колонки и сверка со словарём признаков
#
# Сравниваем фактические колонки таблиц с описанием в `feature_description.csv`,
# чтобы понять, какие признаки фактически присутствуют, а какие — только
# в словаре.

# %%
print("Описание признаков (feature_description.csv):")
print(feature_desc.to_string(index=False))

# %%
all_cols = set(df_train.columns) | set(df_test.columns) | set(df_features.columns)
described_cols = set(feature_desc["Обозначение"].dropna().astype(str))

print(f"Колонки в данных: {len(all_cols)}")
print(f"Колонки в словаре: {len(described_cols)}")

undocumented = sorted(all_cols - described_cols)
documented_but_missing = sorted(described_cols - all_cols)

print(f"\nКолонок без описания в словаре: {len(undocumented)}")
if undocumented:
    print("  ", undocumented[:20], "..." if len(undocumented) > 20 else "")

print(f"\nКолонок из словаря, отсутствующих в данных: {len(documented_but_missing)}")
print("  ", documented_but_missing)

SUMMARY["columns"] = {
    "train": sorted(df_train.columns.tolist()),
    "test": sorted(df_test.columns.tolist()),
    "features": sorted(df_features.columns.tolist()),
    "undocumented_count": len(undocumented),
}

# %% [markdown]
# ## 3. Сравнение схем train и test

# %%
train_only = sorted(set(df_train.columns) - set(df_test.columns))
test_only = sorted(set(df_test.columns) - set(df_train.columns))
common = sorted(set(df_train.columns) & set(df_test.columns))

print(f"Только в train ({len(train_only)}):", train_only)
print(f"Только в test  ({len(test_only)}):", test_only)
print(f"Общих колонок: {len(common)}")

# Сверка типов по общим колонкам
dtype_diff = []
for c in common:
    if df_train[c].dtype != df_test[c].dtype:
        dtype_diff.append((c, str(df_train[c].dtype), str(df_test[c].dtype)))
print("\nКолонки с разными типами в train/test:")
if dtype_diff:
    for row in dtype_diff:
        print(" ", row)
else:
    print("  отсутствуют")

# %% [markdown]
# ## 4. Базовая статистика и пропуски

# %%
print("=== TRAIN: dtypes ===")
print(df_train.dtypes.value_counts())

print("\n=== TRAIN: describe (числовые) ===")
print(df_train.describe().T)

print("\n=== TRAIN: describe (object) ===")
obj_cols = df_train.select_dtypes(include=["object", "category"]).columns
if len(obj_cols):
    print(df_train[obj_cols].describe(include=["object", "category"]).T)
else:
    print("Object-колонок нет")

# %%
def missing_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\n=== Пропуски в {name} (только > 0%) ===")
    if miss.empty:
        print("  отсутствуют")
        return pd.DataFrame()
    out = miss.to_frame("missing_fraction")
    out["missing_count"] = (df.isna().sum().loc[out.index]).astype(int)
    print(out.head(30).to_string())
    return out

miss_train = missing_report(df_train, "train")
miss_test = missing_report(df_test, "test")
miss_feat = missing_report(df_features, "features")

SUMMARY["missing_top"] = {
    "train": miss_train.head(15).to_dict(orient="index") if not miss_train.empty else {},
    "test": miss_test.head(15).to_dict(orient="index") if not miss_test.empty else {},
    "features": miss_feat.head(15).to_dict(orient="index") if not miss_feat.empty else {},
}

# %% [markdown]
# ## 5. Целевая переменная `is_deal`
#
# Проверяем общую частоту, число позитивов на `request_id`, число
# вариантов на запрос. Это ключ к выбору формулировки задачи:
# - если в каждом запросе ровно один позитив — задача типа «pick one
#   from list», LambdaRank подойдёт идеально;
# - если позитивов 0 или несколько — нужно учесть в метрике/обучении.

# %%
print("Доля позитивов (is_deal=1) в train:", df_train[TARGET].mean())
print("Абсолютное число позитивов:", int(df_train[TARGET].sum()))

pos_per_req = df_train.groupby(REQUEST_ID)[TARGET].sum()
variants_per_req = df_train.groupby(REQUEST_ID).size()

print("\nКоличество позитивов на запрос (train):")
print(pos_per_req.value_counts().sort_index().to_string())

print("\nЧисло вариантов на запрос (train):")
print(variants_per_req.describe().to_string())

print("\nЧисло вариантов на запрос (test):")
print(df_test.groupby(REQUEST_ID).size().describe().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(variants_per_req, bins=50, ax=axes[0])
axes[0].set_title("Train: количество вариантов на request_id")
axes[0].set_xlabel("число вариантов")
sns.histplot(df_test.groupby(REQUEST_ID).size(), bins=50, ax=axes[1], color="orange")
axes[1].set_title("Test: количество вариантов на request_id")
axes[1].set_xlabel("число вариантов")
plt.tight_layout()
plt.show()

SUMMARY["target"] = {
    "train_positive_rate": float(df_train[TARGET].mean()),
    "train_positives_total": int(df_train[TARGET].sum()),
    "positives_per_request": pos_per_req.value_counts().sort_index().to_dict(),
    "variants_per_request_train_describe": variants_per_req.describe().to_dict(),
    "variants_per_request_test_describe": df_test.groupby(REQUEST_ID).size().describe().to_dict(),
}

# %% [markdown]
# ## 6. Идентификаторы: уникальность app_id, request_id

# %%
ids_summary = {}
for name, df in [("train", df_train), ("test", df_test)]:
    ids_summary[name] = {
        "n_rows": len(df),
        "n_unique_request_id": df[REQUEST_ID].nunique(),
        "n_unique_app_id": df["app_id"].nunique() if "app_id" in df.columns else None,
        "request_per_app": (
            df.groupby("app_id")[REQUEST_ID].nunique().describe().to_dict()
            if "app_id" in df.columns else None
        ),
    }
    print(f"=== {name} ===")
    for k, v in ids_summary[name].items():
        print(f"  {k}: {v}")
    print()

overlap_apps = set(df_train["app_id"].unique()) & set(df_test["app_id"].unique())
overlap_requests = set(df_train[REQUEST_ID].unique()) & set(df_test[REQUEST_ID].unique())
print(f"Пересечение app_id между train/test: {len(overlap_apps)}")
print(f"Пересечение request_id между train/test: {len(overlap_requests)}")

SUMMARY["ids"] = ids_summary
SUMMARY["overlap"] = {
    "app_id_intersection": len(overlap_apps),
    "request_id_intersection": len(overlap_requests),
}

# %% [markdown]
# ## 7. Временной разрез (`date_part`, `request_received`)
#
# Сравниваем диапазоны дат в train и test. Если test строго в будущем
# относительно train — выбираем time-based валидацию. Если перемешано —
# можно безопасно использовать GroupKFold по `request_id`.

# %%
df_train["_date"] = pd.to_datetime(df_train["date_part"])
df_test["_date"] = pd.to_datetime(df_test["date_part"])

print("Train date range:", df_train["_date"].min(), "→", df_train["_date"].max())
print("Test  date range:", df_test["_date"].min(), "→", df_test["_date"].max())
print("Пересечение дат:", df_train["_date"].max() >= df_test["_date"].min()
      and df_test["_date"].max() >= df_train["_date"].min())

train_by_day = df_train.groupby("_date").agg(
    n_rows=(REQUEST_ID, "size"),
    n_requests=(REQUEST_ID, "nunique"),
    target_rate=(TARGET, "mean"),
)
test_by_day = df_test.groupby("_date").agg(
    n_rows=(REQUEST_ID, "size"),
    n_requests=(REQUEST_ID, "nunique"),
)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(train_by_day.index, train_by_day["n_requests"], label="train", color="steelblue")
axes[0].plot(test_by_day.index, test_by_day["n_requests"], label="test", color="orange")
axes[0].set_title("Число уникальных request_id по дням")
axes[0].legend()
axes[1].plot(train_by_day.index, train_by_day["target_rate"], color="green")
axes[1].set_title("Target rate (is_deal) по дням в train")
axes[1].set_xlabel("Дата")
plt.tight_layout()
plt.show()

SUMMARY["dates"] = {
    "train_min": str(df_train["_date"].min().date()),
    "train_max": str(df_train["_date"].max().date()),
    "test_min": str(df_test["_date"].min().date()),
    "test_max": str(df_test["_date"].max().date()),
    "n_unique_dates_train": int(df_train["_date"].nunique()),
    "n_unique_dates_test": int(df_test["_date"].nunique()),
}
print("\n=== ВЫВОД: временной разрез ===")
print(SUMMARY["dates"])

# %%
# Часовое распределение запросов (если request_received содержит время)
if "request_received" in df_train.columns:
    df_train["_ts"] = pd.to_datetime(df_train["request_received"], errors="coerce")
    if df_train["_ts"].notna().any():
        hour = df_train["_ts"].dt.hour
        dow = df_train["_ts"].dt.dayofweek
        print("Распределение по часам суток (train):")
        print(hour.value_counts().sort_index().to_string())
        print("\nРаспределение по дню недели (0=пн..6=вс):")
        print(dow.value_counts().sort_index().to_string())

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.countplot(x=hour.dropna(), ax=axes[0])
        axes[0].set_title("Часы (train)")
        sns.countplot(x=dow.dropna(), ax=axes[1])
        axes[1].set_title("Дни недели (train)")
        plt.tight_layout()
        plt.show()
    else:
        print("request_received не парсится в datetime — формат нужно посмотреть отдельно")

# %% [markdown]
# ## 8. Покрытие join'а с features_small.pq

# %%
key_cols_features = [c for c in ["app_id", "date_part"] if c in df_features.columns]
print("Ключи features:", key_cols_features)
print("Уникальность ключа в features:",
      df_features[key_cols_features].drop_duplicates().shape[0],
      "из", len(df_features))

if {"app_id", "date_part"}.issubset(df_features.columns):
    train_keys = df_train[["app_id", "date_part"]].drop_duplicates()
    test_keys = df_test[["app_id", "date_part"]].drop_duplicates()
    coverage_train = train_keys.merge(df_features[["app_id", "date_part"]], on=["app_id", "date_part"]).shape[0] / len(train_keys)
    coverage_test = test_keys.merge(df_features[["app_id", "date_part"]], on=["app_id", "date_part"]).shape[0] / len(test_keys)
    print(f"\nПокрытие join train↔features: {coverage_train:.4f}")
    print(f"Покрытие join test↔features:  {coverage_test:.4f}")
    SUMMARY["feature_join_coverage"] = {
        "train": float(coverage_train),
        "test": float(coverage_test),
    }

print("\nКолонки features_small.pq:")
print(df_features.dtypes.to_string())

# %% [markdown]
# ## 9. Числовые признаки заявки (`req_loan_amount`, `req_term`)

# %%
req_features = [c for c in ["req_loan_amount", "req_term"] if c in df_train.columns]
fig, axes = plt.subplots(len(req_features), 2, figsize=(14, 4 * len(req_features)))
if len(req_features) == 1:
    axes = axes.reshape(1, -1)
for i, c in enumerate(req_features):
    series_train = df_train[c].dropna()
    sns.histplot(series_train, bins=50, ax=axes[i, 0], kde=False)
    axes[i, 0].set_title(f"{c} (train) — гистограмма (log y)")
    axes[i, 0].set_yscale("log")
    sns.boxplot(data=df_train, x=TARGET, y=c, ax=axes[i, 1])
    axes[i, 1].set_title(f"{c} vs {TARGET}")
plt.tight_layout()
plt.show()

for c in req_features:
    print(f"{c}: train describe")
    print(df_train[c].describe().to_string())
    print()

# %% [markdown]
# ## 10. Числовые признаки оффера (`rate`, `term`, `limit`, `eva`, `eva_perc`, `ncl`)

# %%
offer_features = [c for c in ["rate", "term", "limit", "eva", "eva_perc", "ncl"] if c in df_train.columns]
print("Числовые признаки оффера:", offer_features)

for c in offer_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    sns.histplot(df_train[c].dropna(), bins=60, ax=axes[0])
    axes[0].set_title(f"{c} — гистограмма (train)")
    sns.boxplot(data=df_train, x=TARGET, y=c, ax=axes[1])
    axes[1].set_title(f"{c} vs {TARGET}")
    plt.tight_layout()
    plt.show()

# Spearman-корреляция с таргетом
if offer_features:
    spearman = df_train[offer_features + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
    print("Spearman-корреляция признаков оффера с is_deal:")
    print(spearman.sort_values(ascending=False).to_string())
    SUMMARY["spearman_with_target_offer"] = spearman.to_dict()

# %% [markdown]
# ## 11. Категориальные признаки оффера и заявки

# %%
cat_candidates = [
    "offer_type",
    "risk_level_map",
    "channel",
    "verif_compl",
    "verif_need",
    "need_2ndfl",
    "pil1mtrx_offer",
]
cat_features = [c for c in cat_candidates if c in df_train.columns]
print("Категориальные кандидаты:", cat_features)

cat_summary: dict[str, dict] = {}
for c in cat_features:
    vc = df_train[c].value_counts(dropna=False)
    target_by_cat = df_train.groupby(c, dropna=False)[TARGET].agg(["mean", "count"]).sort_values("count", ascending=False)
    print(f"\n=== {c} ===")
    print(target_by_cat.head(20).to_string())
    cat_summary[c] = {
        "n_unique": int(df_train[c].nunique(dropna=False)),
        "top": vc.head(10).to_dict(),
        "target_rate_top": target_by_cat.head(10)["mean"].to_dict(),
    }

    fig, ax = plt.subplots(figsize=(min(14, max(6, vc.shape[0] * 0.4)), 3.5))
    target_by_cat.head(20)["mean"].plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Target rate по {c} (top-20 по частоте)")
    ax.axhline(df_train[TARGET].mean(), color="red", linestyle="--", label="общий target rate")
    ax.legend()
    plt.tight_layout()
    plt.show()

SUMMARY["categoricals"] = cat_summary

# %% [markdown]
# ## 12. Анализ `basket_name`
#
# По описанию это список бизнес-правил. Может быть строкой или списком —
# нужно посмотреть.

# %%
if "basket_name" in df_train.columns:
    sample_values = df_train["basket_name"].dropna().head(10).tolist()
    print("Примеры значений basket_name:")
    for v in sample_values:
        print(f"  type={type(v).__name__}: {v}")
    print(f"\nКардинальность basket_name: {df_train['basket_name'].nunique()}")

# %% [markdown]
# ## 13. Position bias по `variant_no`
#
# Ключевой вопрос: есть ли смещение «нумерация → вероятность сделки»?
# Если позитивы концентрируются на маленьких variant_no — модель должна
# учитывать порядок (или мы выводим обратную пропенсити для устранения
# bias).

# %%
print("Описание variant_no в train:")
print(df_train[VARIANT_ID].describe().to_string())
print("\nЧастоты variant_no (top-30):")
print(df_train[VARIANT_ID].value_counts().sort_index().head(30).to_string())

variant_target = df_train.groupby(VARIANT_ID)[TARGET].agg(["mean", "count"]).reset_index()
variant_target = variant_target.sort_values(VARIANT_ID)

fig, axes = plt.subplots(2, 1, figsize=(14, 7))
sns.barplot(data=variant_target.head(50), x=VARIANT_ID, y="count", ax=axes[0], color="steelblue")
axes[0].set_title("Частота variant_no (first 50)")
sns.barplot(data=variant_target.head(50), x=VARIANT_ID, y="mean", ax=axes[1], color="darkred")
axes[1].axhline(df_train[TARGET].mean(), color="black", linestyle="--", label="overall")
axes[1].set_title("Target rate по variant_no (first 50)")
axes[1].legend()
plt.tight_layout()
plt.show()

SUMMARY["variant_no_target_top10"] = variant_target.head(10).to_dict(orient="records")

# %% [markdown]
# ## 14. Внутригрупповые паттерны: где находится позитив внутри запроса?

# %%
def positive_rank_stats(df: pd.DataFrame, sort_col: str, ascending: bool) -> dict:
    sub = df[[REQUEST_ID, sort_col, TARGET]].copy()
    sub["rank_within_req"] = sub.groupby(REQUEST_ID)[sort_col].rank(method="first", ascending=ascending)
    pos = sub.loc[sub[TARGET] == 1]
    if pos.empty:
        return {}
    return {
        "mean_rank_of_positive": float(pos["rank_within_req"].mean()),
        "median_rank_of_positive": float(pos["rank_within_req"].median()),
        "rank_distribution_top5": pos["rank_within_req"].clip(upper=5).value_counts(normalize=True).sort_index().to_dict(),
    }

for sort_col, ascending in [
    (VARIANT_ID, True),
    ("rate", True),
    ("eva", False),
    ("eva_perc", False),
    ("limit", False),
    ("ncl", True),
]:
    if sort_col in df_train.columns:
        stats = positive_rank_stats(df_train, sort_col, ascending)
        order = "asc" if ascending else "desc"
        print(f"=== {sort_col} ({order}) — где позитив? ===")
        print(stats)
        SUMMARY.setdefault("positive_rank_by", {})[f"{sort_col}_{order}"] = stats
        print()

# %% [markdown]
# ## 15. Корреляции и взаимная информация
#
# Корреляции считаем только на числовых, MI — на выборке для скорости.

# %%
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in {"app_id", REQUEST_ID}]
print("Числовые колонки для корреляций:", numeric_cols)

if numeric_cols:
    corr = df_train[numeric_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(min(14, len(numeric_cols) * 0.8), min(10, len(numeric_cols) * 0.6)))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman correlation между числовыми признаками")
    plt.tight_layout()
    plt.show()

    spearman_target = df_train[numeric_cols + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
    print("\nSpearman vs is_deal:")
    print(spearman_target.sort_values(key=lambda s: s.abs(), ascending=False).head(20).to_string())

# %% [markdown]
# ## 16. Leakage-проверка
#
# Ищем признаки, аномально сильно связанные с таргетом (R² > 0.5 на
# decision tree depth=1) — это потенциальный leakage. Также проверяем,
# нет ли строк в train, которые точно повторяются в test (по
# идентификаторам).

# %%
from sklearn.tree import DecisionTreeClassifier

leakage_report = []
for c in numeric_cols:
    series = df_train[c].fillna(-9999)
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(series.to_frame(), df_train[TARGET])
    score = clf.score(series.to_frame(), df_train[TARGET])
    if score > 0.95:
        leakage_report.append((c, score))

print("Подозрительные признаки (accuracy depth=1 > 0.95):")
for row in leakage_report:
    print(" ", row)

# Проверка дубликатов
dup_keys = ["app_id", REQUEST_ID, VARIANT_ID]
dup_keys = [k for k in dup_keys if k in df_train.columns and k in df_test.columns]
common_keys = df_train[dup_keys].drop_duplicates().merge(
    df_test[dup_keys].drop_duplicates(), on=dup_keys
)
print(f"\nОдинаковых (app_id, request_id, variant_no) в train и test: {len(common_keys)}")
SUMMARY["leakage_suspects"] = [c for c, _ in leakage_report]

# %% [markdown]
# ## 17. Adversarial validation: насколько train отличим от test
#
# Обучаем LightGBM (или, при его отсутствии, GradientBoosting) различать
# train (0) и test (1) на общих признаках. AUC ≈ 0.5 — распределения
# совпадают, AUC → 1 — сильный shift.

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

shared_numeric = [c for c in numeric_cols if c in df_test.columns]
shared_numeric = [c for c in shared_numeric if c not in {TARGET}]
print("Общие числовые признаки для adversarial validation:", shared_numeric)

sample_train = df_train[shared_numeric].sample(n=min(50_000, len(df_train)), random_state=42)
sample_test = df_test[shared_numeric].sample(n=min(50_000, len(df_test)), random_state=42)

X_adv = pd.concat([sample_train, sample_test], axis=0).fillna(-9999)
y_adv = np.array([0] * len(sample_train) + [1] * len(sample_test))

try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1)
except ImportError:
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

auc_scores = cross_val_score(model, X_adv, y_adv, cv=3, scoring="roc_auc")
print(f"Adversarial AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print("Чем ближе к 0.5 — тем больше train и test похожи.")

SUMMARY["adversarial_auc"] = {
    "mean": float(auc_scores.mean()),
    "std": float(auc_scores.std()),
}

# Если AUC > 0.7 — посмотреть feature importance
if auc_scores.mean() > 0.65 and hasattr(model, "fit"):
    model.fit(X_adv, y_adv)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=shared_numeric)
        print("\nTop признаки, отличающие train от test:")
        print(importances.sort_values(ascending=False).head(15).to_string())
        SUMMARY["adversarial_top_features"] = importances.sort_values(ascending=False).head(15).to_dict()

# %% [markdown]
# ## 18. Память и потенциал даункастинга

# %%
def downcast_estimate(df: pd.DataFrame, name: str) -> dict:
    before = df.memory_usage(deep=True).sum() / 1024 ** 2
    df2 = df.copy()
    for c in df2.select_dtypes(include=["int64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="integer")
    for c in df2.select_dtypes(include=["float64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="float")
    after = df2.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"{name}: {before:.1f} MB → {after:.1f} MB (экономия {(1 - after / before) * 100:.1f}%)")
    del df2
    return {"before_MB": float(before), "after_MB": float(after)}

SUMMARY["downcast"] = {
    "train": downcast_estimate(df_train, "train"),
    "test": downcast_estimate(df_test, "test"),
    "features": downcast_estimate(df_features, "features"),
}

# %% [markdown]
# ## 19. Сводка и сохранение `eda_summary.json`

# %%
SUMMARY["recommended_cv_scheme"] = (
    "time"
    if (
        SUMMARY["dates"]["train_max"] < SUMMARY["dates"]["test_min"]
        or SUMMARY.get("adversarial_auc", {}).get("mean", 0.5) > 0.7
    )
    else "group"
)

print("=" * 70)
print("СВОДКА EDA (то, что копируем в чат):")
print("=" * 70)
print(f"Размеры: {SUMMARY['shapes']}")
print(f"Память: {SUMMARY['memory_MB']}")
print(f"Target rate в train: {SUMMARY['target']['train_positive_rate']:.4f}")
print(f"Позитивов всего: {SUMMARY['target']['positives_total'] if 'positives_total' in SUMMARY['target'] else SUMMARY['target']['train_positives_total']}")
print(f"Позитивов на запрос: {SUMMARY['target']['positives_per_request']}")
print(f"Вариантов на запрос (train, describe): {SUMMARY['target']['variants_per_request_train_describe']}")
print(f"Даты train: {SUMMARY['dates']['train_min']} → {SUMMARY['dates']['train_max']}")
print(f"Даты test:  {SUMMARY['dates']['test_min']} → {SUMMARY['dates']['test_max']}")
print(f"Покрытие join с features: {SUMMARY.get('feature_join_coverage')}")
print(f"Колонок, общих в train/test: см. выше")
print(f"Adversarial AUC: {SUMMARY['adversarial_auc']}")
print(f"Рекомендация по схеме CV: {SUMMARY['recommended_cv_scheme']}")
print(f"Подозрительные на leakage признаки: {SUMMARY['leakage_suspects']}")

OUT_SUMMARY.write_text(json.dumps(SUMMARY, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"\nСохранено в {OUT_SUMMARY}")

# %% [markdown]
# ## 20. Финальные текстовые выводы
#
# Скопируйте этот блок в чат после прогона, чтобы передать ключевые
# наблюдения для построения пайплайна.

# %%
print("\n" + "=" * 70)
print("КЛЮЧЕВЫЕ ВЫВОДЫ (для копирования в чат):")
print("=" * 70)

bullets = [
    f"1. Размеры: train={shapes['train']}, test={shapes['test']}, features={shapes['features']}.",
    f"2. Target rate в train: {SUMMARY['target']['train_positive_rate']:.4f}; позитивов всего: {int(df_train[TARGET].sum())}.",
    f"3. Распределение позитивов на запрос: {SUMMARY['target']['positives_per_request']}.",
    f"4. Среднее число вариантов на request_id (train): {variants_per_req.mean():.2f}, max={int(variants_per_req.max())}.",
    f"5. Временной разрез train: {SUMMARY['dates']['train_min']} → {SUMMARY['dates']['train_max']}.",
    f"6. Временной разрез test : {SUMMARY['dates']['test_min']} → {SUMMARY['dates']['test_max']}.",
    f"7. Пересечение app_id между train и test: {SUMMARY['overlap']['app_id_intersection']}.",
    f"8. Покрытие join с features (train/test): {SUMMARY.get('feature_join_coverage')}.",
    f"9. Колонок, отсутствующих в test: {sorted(set(df_train.columns) - set(df_test.columns))}.",
    f"10. Adversarial AUC: {SUMMARY['adversarial_auc']['mean']:.4f} ± {SUMMARY['adversarial_auc']['std']:.4f}.",
    f"11. Рекомендованная схема CV: {SUMMARY['recommended_cv_scheme']}.",
    f"12. Подозрения на leakage: {SUMMARY['leakage_suspects']}.",
    f"13. Топ-3 признака с наибольшей |Spearman| с is_deal:",
]
print("\n".join(bullets))

if numeric_cols:
    top_spearman = (
        df_train[numeric_cols + [TARGET]].corr(method="spearman")[TARGET]
        .drop(TARGET).abs().sort_values(ascending=False).head(5)
    )
    for c, v in top_spearman.items():
        print(f"    • {c}: |ρ|={v:.4f}")

print("\nКонец EDA. Передайте этот вывод в чат для обновления плана фазы 2.")
