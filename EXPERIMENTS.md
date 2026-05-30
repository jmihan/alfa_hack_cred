# Лог экспериментов

Хронология ключевых направлений и pipeline-ов с цифрами CV/LB. Источник —
заметки из ночных прогонов; полная таблица LB всех сабмитов — в Приложении A ниже.
Финальный результат — **LB ≈ 92.18** (рекордный пайплайн с фичей `is_best_both`
и широким offer-набором; см. секции «Pipeline T» и «Рекордный пайплайн» ниже).
Предыдущая стабильная веха — LB 92.0504.

Цифры LB ниже — это NDCG@5 × 100 (как на лидерборде Яндекс Контеста).
CV считается на 5-фолд GroupKFold по `request_id` (см.
[`src/alfa_cred/validation.py`](src/alfa_cred/validation.py)).

## Шкала прироста

| LB | Что добавилось | Δ |
|----|----------------|---|
| 91.634 | LGBM LambdaRank baseline | — |
| 91.7512 | + Optuna для гиперпараметров | +0.117 |
| 91.8648 | + extended features | +0.114 |
| 91.9471 | + первый mega-blend (12 моделей) | +0.082 |
| 91.9668 | - cb_yetirank_extended (CV 0.9128, размывал) | +0.020 |
| 91.9939 | + two-stage (xgb_b_default для подзадачи B) | +0.027 |
| 92.0006 | + multi-seed XGB в B-blend | +0.007 |
| 92.0317 | + bBalanced (top-3 каждого типа, 12 B-моделей) | +0.031 |
| 92.0494 | + AE-recovered модели в B-blend (Pipeline O) | +0.018 |
| **92.0504** | + pseudo-labeling + cross-objective | +0.001 |

## Pipeline A-G: фундамент

Базовые модели и feature engineering до начала ночных прогонов.

- **Pipeline A (2026-05-22).** Baseline LGBM LambdaRank, CV 0.9143, LB 91.634.
- **Pipeline B (2026-05-22).** Optuna для LGBM (30 trials), CV 0.9158, LB 91.7512.
- **Pipeline C (2026-05-23).** Extended feature engineering — внутригрупповые
  ранги, z-scores, агрегаты, time features, target encoding. CV 0.9165, LB 91.8648.
  Adversarial pruning против drift пробовали — не помогло, оставил все фичи.
- **Pipeline D (2026-05-23).** Multi-seed LightGBM (extended) + CatBoost YetiRank.
  Сборка blend_mega_strong_only_d из 12 моделей → LB 91.9471.
- **Pipeline E-F (2026-05-23).** Hill-climbing weights, попытки взвешенного
  blend-а. Uniform работает не хуже, но проще. Финал — record_11 (11 моделей,
  убрана cb_yetirank_extended с CV 0.9128 как «размывающая»). LB 91.9668.
- **Pipeline G (2026-05-23 — 24).** Дополнительные сильные одиночки
  (lgbm_bootstrap_v_s256, lgbm_extended_features). Сильнейшая одиночка
  lgbm_boot_v_s256 = LB 91.8774 (CV 0.9165).

## Pipeline H (2026-05-24, 7.5ч). Tabular DL — ПРОВАЛ

**Цель.** Попробовать FT-Transformer и TabNet — теоретически могут найти
паттерны, недоступные GBM.

**Что обучалось.**
- FT-Transformer (rtdl-revisiting-models), 2 seed-варианта.
- TabNet (pytorch-tabnet).

**Результаты.**
- ft_trans_seed42: CV 0.9129, одиночка LB 91.4893.
- ft_trans_seed123: CV 0.9119.
- tabnet_seed42: CV 0.8886, одиночка LB 90.2362.
- Blend record_11 + 2 FT-T → LB 91.9601 (-0.0067).
- Blend record_11 + 2 FT-T + TabNet → LB 91.9247 (-0.042).

**Вывод.** Diversity без минимальной точности (CV ≥ 0.913) не помогает,
а только размывает. Корреляция TabNet с GBM 0.764 — звучит привлекательно,
но на практике добавление в blend ухудшает LB.

## Pipeline I (2026-05-24, 8ч). Customer history + CB Optuna — частично ПРОВАЛ

**Цель.** Использовать историю клиента через `app_id` join и более тщательный
Optuna для CatBoost.

**Что обучалось.**
- lgbm_history: с историческими фичами по `app_id`.
- lgbm_epoch_pre / lgbm_epoch_post: модели для разных временных эпох.
- cb_deep_optuna: 50 trials Optuna.

**Результаты.**
- lgbm_history: CV 0.9161 → не дотягивает.
- lgbm_epoch_post: CV 0.9521 (!) на сабсете, но одиночка на полном test
  LB 91.4377 — CV завышен из-за более простой задачи внутри эпохи.
- cb_deep_optuna: CV 0.9167, одиночка LB 91.8477. В blend record_11
  даёт LB 91.9497 (-0.017).

**Вывод.** Customer history бесполезен — мало пересечений app_id
train/test (76 на 845K строк). Per-epoch модели катастрофически
переобучаются на «лёгкой» подзадаче. CB Optuna одиночкой хороша,
но в blend ухудшает.

## Pipeline J (2026-05-25, 12ч). Bootstrap + XGB Optuna — переобучение

**Цель.** Bootstrap LGBM (7 seed-варианта), deep Optuna для XGBoost,
CatBoost multi-seed.

**Результаты.**
- xgb_deep_optuna: CV 0.9181 (рекорд CV!), но одиночка LB 91.8349 —
  переобучение Optuna.
- lgbm_boot multi-seed: добавление в record_11 даёт LB 91.9418 (-0.025).
- LB-weighted blend (попытка взвешивать по LB) — LB 91.9354 (-0.031).

**Вывод.** Парадокс CV vs LB подтвердился: модель с лучшим CV даёт
худший LB. Bootstrap-вариативность лучше Optuna-тюнинга. LB-weighted
хуже uniform.

## Pipeline K (2026-05-25, 0.34ч). TWO-STAGE — ПРОРЫВ

**Цель.** Расщепить задачу по `pil1mtrx_offer`: для 34% запросов без
этого флага обучить отдельные модели только на этой подзадаче (B-only).

**Что обучалось.**
- xgb_b_default: XGBoost LambdaRank на 12 650 B-запросах (по 25% от train).
- lgbm_b: то же на LGBM.
- cb_b: CatBoost YetiRank на B.

**Результаты.**
- xgb_b_default: B-NDCG@5 = 0.7532 (CV только по B!).
- two_stage_record11_plus_bTop1: A=record_11+hard-rule,
  B=xgb_b_default → **LB 91.9939 (+0.027 от 91.9668)**.

**Вывод.** Главный прорыв проекта. B-only модель, обученная **только
на 34% запросов**, даёт сильнее сигнал для подзадачи B, чем модели
обученные на всём train. Видимо потому что общая модель тратит
ёмкость на структуру подзадачи A.

## Pipeline L (2026-05-25, 1.38ч). Multi-seed B-only

**Цель.** Размножить B-only XGB на разные seed.

**Что обучалось.**
- xgb_b_seed42, s314, s8848, s137, s2026 (5 моделей с Optuna).
- lgbm_b_extended.

**Результаты.**
- two_stage_record11_plus_bAllL: 5 XGB Optuna + 1 LGBM ext → LB 92.0006.
- two_stage_record11_plus_bAllXGB (5 XGB) → LB 91.9765.

**Вывод.** Multi-seed XGB ensemble даёт +0.024 от одиночки. Пробил
символический рубеж 92.

## Pipeline M (2026-05-26, 2.22ч). Multi-arch B-only

**Цель.** Расширить B-only blend моделями разных типов (CatBoost, LGBM
bootstrap, LGBM Optuna).

**Что обучалось.**
- 8 CatBoost YetiRank с разными seed (s42, s123, s777, s1024, s314, s628, s99, s8848).
- 6 LGBM bootstrap (s41, s256, s99, s628, s1111, s2024).
- 5 XGB seed-варианта (s42, s137, s314, s8848, s2026).
- 3 LGBM Optuna seed-варианта (s42, s137, s314).

**Результаты.**
- two_stage_record11_plus_bMnew (только 22 M-моделей): LB 92.0232.
- two_stage_record11_plus_bFull (K+L+M, 34 моделей): LB 92.0263.
- two_stage_record11_plus_bMega (35 моделей): LB 92.0432.
- two_stage_record11_plus_bBalanced (top-3 каждого типа, 12 моделей): **LB 92.0317**.

**Вывод.** Размер blend имеет sweet spot. bBalanced (12 моделей)
обходит bMega (35) на 0.012 — слишком много моделей размывает.
Cross-architecture diversity (XGB + LGBM + CB) важна.

## Pipeline N (2026-05-26, 0.26ч). Pseudo-labeling + cross-objective

**Цель.** Pseudo-labeling (использовать confident test predictions как
pseudo-train), distillation, cross-objective диверсификация.

**Что обучалось.**
- xgb_b_pseudo_s42, xgb_b_pseudo_s137: дообучение на ~727 confident
  B-запросов из test (5.4%).
- lgbm_b_pseudo: то же на LGBM.
- lgbm_b_distill: дистилляция от blend (получилось плохо, CV 0.2007).
- xgb_b_pairwise: альтернативная objective.
- lgbm_b_xendcg: альтернативная objective (cross-entropy NDCG).

**Результаты.**
- two_stage_record11_plus_bKLM_plus_pseudo (32 моделей): LB 92.0288.
- two_stage_record11_plus_bKLM_plus_crossobj (31 моделей): LB 92.0324.
- two_stage_record11_plus_bN_0245 (только N, без других): LB 89.7603 (катастрофа).

**Вывод.** Pseudo-labeling и cross-objective дают +0.004 в blend каждый,
но сами по себе слабее одиночек. Дистилляция с CV 0.20 удивительным
образом тоже вкладывает в большой ансамбль (через ensemble dropout эффект).

## Pipeline O (2026-05-26, 0.35ч). Поиск «скрытого сигнала» — НЕ нашёл

**Цель.** Лидерборд показал 92.55, наш потолок ~92.05. Подозрение, что
есть какой-то сигнал в данных, который мы не используем. Поискать через:
- MI scan на 280 фичах,
- AutoEncoder embeddings (8/16 dim),
- паттерны в `request_received`,
- ext-фичи (производные от автоэнкодера).

**Результаты.**
- MI scan: топ-15 признаков с MI 0.06 в шуме, ничего значимого.
- AE embeddings (8/16): добавление в B-only XGB не помогло.
- xgb_b_pseudo_v2 (улучшенная версия pseudo с margin threshold 0.1):
  B-NDCG@5 = 0.8175 (рекорд!), но в blend +0.002 максимум.
- two_stage_record11_plus_bBalanced_plus_bO (18 моделей): LB 92.0494.

**Вывод.** «Скрытого сигнала» в данных нет (или мы не научились его
вынимать). Лидер 92.55 вероятно использует внешние данные или
специфический трюк, который мы повторить не можем.

## Pipeline P (2026-05-26, 1.25ч). Subgroup + stacking

**Цель.** Specialization по `offer_type` (subgroup модели), pointwise,
pairwise binary classification, stacking.

**Результаты.**
- xgb_b_subg_ot2: B-NDCG@5 = 0.7831 (хорошо!), но в blend ухудшает.
- xgb_b_pairwise_bin (pairwise classification): слабый.
- lgbm_b_stacking: CV 0.7497.
- two_stage_record11_plus_bAllKLMNOP (51 моделей): LB 92.0458.
- three_stage с subg_ot2 как заменой RA: LB 91.3545 (катастрофа).

**Вывод.** Subgroup специализация работает только как ДОПОЛНЕНИЕ к
основному B-blend, не как замена. Pairwise binary classification на
нашей постановке не работает (tournament не складывается в линейный
порядок).

## Pipeline T (2026-05-29). Фича `is_best_both` — рычаг по подзадаче B

**Идея.** Доразобрать подзадачу B через сильный детерминированный сигнал.
В разрезе B-запросов оффер, который точно повторяет заявку клиента
(`limit == req_loan_amount` И `term == req_term`) и стоит первым среди таких
(минимальный `variant_no`), почти в половине случаев становится сделкой:
deal-rate ≈ **52%** против ≈2.8% у остальных, покрытие ≈ 59% B-запросов. Это
«мягкий» аналог hard-rule `pil1mtrx_offer`, но для подзадачи B. Сигнал не лик
(детерминирован из pre-decision атрибутов) и не дрейфует (offer-rate train-B
0.0236 ↔ test-B 0.0248).

Реализовано фичей `is_best_both` (модуль [`features/match.py`](src/alfa_cred/features/match.py))
плюс вспомогательный match-стек (`lim_match`, `term_match`, `both_match`,
`n_both`, `is_uniq_both`, `min_vn_both`, `vn_rank_in_both`) и Парето-признаки
(`pareto_dominated_cnt`, `is_pareto_optimal`, относительные отношения).

**Ключевое наблюдение.** Ценность несёт именно композит `is_best_both`
(трёхуровневое «И» по точному равенству), а не сырые equality-флаги: GBDT не
строит такой композит дёшево из непрерывных разниц (`req_minus_limit`,
`req_minus_term`, внутригрупповые ранги), которые уже были в пайплайне.
Поэтому эффект почти не перекрывается имеющимися фичами.

**Контролируемая абляция (5-fold GroupKFold, B-only).** Один и тот же бленд из
9 B-моделей (3 CatBoost + 3 LGBM bootstrap + 3 XGBoost) обучен дважды — с
match-фичами и без, на одних данных/фолдах/сидах:

| Бленд (B-only, 9 моделей) | blend-OOF NDCG@5 |
|---------------------------|------------------|
| БЕЗ `is_best_both` | 0.7569 |
| С `is_best_both`   | **0.7632** |
| Δ | **+0.0063** |

Прирост значим (эмпирический порог +0.004) и стабилен по всем архитектурам
(per-model +0.006…+0.009). Эффект корректно мерить именно на бленде, а не на
одиночной модели. Финальный two-stage (A = record_11 + hard-rule, B = bBalanced
с `is_best_both`) — кандидат на новый рекорд (LB на проверке).

**Что НЕ сработало — дизамбигуация multi-both.** Когда в запросе несколько
ask-matching офферов (`n_both ≥ 2`), пробовали уточнить выбор «лучшим both по
eva / по rate» (`is_best_both_eva`, `is_best_both_rate` + ранги внутри
both-подмножества). На полном B-бленде (honest time-split): **NDCG@5 0.7430 vs
0.7443 (−0.0013)** — слегка хуже. Deal-rate внутри both-подмножества:
`is_best_both` (по min `variant_no`) = 52.0%, top-eva = 31.7%, min-rate = 30.2%.
Вывод: `variant_no` уже оптимальный дизамбигуатор (отражает коммерческий
приоритет банка лучше, чем финансовые параметры оффера). Эти фичи не добавляем.

## Рекордный пайплайн (2026-05-30). Широкий offer-набор + B-бленд — LB ≈ 92.18

После того как `is_best_both` подтвердился на LB, собрали отдельный рекордный
B-бленд на **широком offer-наборе** ([`build_wide_feature_table`](src/alfa_cred/features/pipeline.py),
361 признак; внутригрупповые функции — в [`features/group.py`](src/alfa_cred/features/group.py)).
Отличия от основного `build_feature_table`:

- клиентские признаки мерджатся **целиком** (минус служебные `*_date`), без
  фильтра по заполненности — больше контекста на подзадаче B;
- расширенные внутригрупповые ранги/min-max/dev, кросс-офферные сравнения внутри
  типа и уровня риска (`*_rank_in_type`, `*_rank_in_risk`, `*_share_lower/better`),
  Парето-доминирование, индикаторы `is_lowest_rate`/`is_highest_eva`/`is_max_limit`,
  и ask-match стек с `is_best_both`.

**B-бленд** ([`models/b_blend.py`](src/alfa_cred/models/b_blend.py)):
8 моделей — LightGBM LambdaRank ×3 + XGBoost rank:ndcg ×3 + CatBoost YetiRank ×2,
каждая обучается один раз на всех train-B заявках, скоры усредняются как
перцентильные ранги внутри `request_id`.

**Two-stage сборка** ([`scripts/predict.py`](scripts/predict.py)):
A = rank-avg blend record_11 + hard-rule (pil1 → верх), B = этот 8-модельный бленд.
Итог — **LB ≈ 92.18**.

**Что подтвердилось разложением по LB:**
- Рычаг рекорда — именно широкий offer-набор B + 8-модельный ranker-бленд; A-сторона
  на record_11 чуть сильнее альтернатив (свопы A-базы давали −0.002).
- Фича `is_best_both` остаётся ядром сигнала B и здесь.
- CV-OOF на подзадаче B слабо предсказывает прирост на test-B (тот же CV-LB разрыв,
  что и раньше): рост в OOF от широкого набора ≈ +0.0005, а на LB — заметно больше.

## Pointwise-MLP диверсити (2026-05-31) — LB ≈ 92.196

**Идея.** Все 8 моделей B-бленда — GBDT (LightGBM/XGBoost/CatBoost), архитектурно
близкие. Добавить ортогональную диверсити модели ДРУГОЙ природы — нейросеть.

**Модель** ([`models/mlp_pointwise.py`](src/alfa_cred/models/mlp_pointwise.py)):
детерминированный pointwise tabular-MLP, бинарный `P(is_deal)` (эмбеддинги
категориальных + стандартизованные числовые → MLP → сигмоида), усреднение по
3 сидам. Внутри заявки ровно один позитив, поэтому порядок по `P(is_deal)`
согласуется с ранжированием. Детерминизм: фиксированные сиды + `cudnn.deterministic`.

**Ключевое наблюдение.** MLP **ортогонален** GBDT-бленду: Spearman рангов с
`b_blend` ≈ **0.84** (для сравнения, GBDT-pointwise давал ≈0.99 — избыточен).
Бленд B-стороны `0.70·b_blend + 0.30·MLP`.

**CV-LB разрыв (снова в нашу пользу).** На honest B-сплите прирост в OOF едва
заметен (+0.0005), но на LB добавление MLP дало **+0.018** (92.1764 → 92.1957).
Подтверждает прежний вывод: OOF на подзадаче B систематически недооценивает
ценность диверсити; решение принимается по LB.

**A-сторона.** Заменили историческую 11-модельную A-базу (бралась из сохранённых
OOF) на компактный 5-модельный A-бленд, обучаемый с нуля
([`models/a_blend.py`](src/alfa_cred/models/a_blend.py): 3×LGBM extended +
2×CatBoost YetiRank). Top-1 на test совпадает 1:1 с прежней A-базой (hard-rule
доминирует на A-заявках), итог по LB не меняется.

**Воспроизводимость с нуля.** Пайплайн разнесён на train (обучить + сохранить
модели, [`scripts/fit_pipeline.py`](scripts/fit_pipeline.py)) и inference (собрать
сабмит из сохранённых, [`scripts/predict.py`](scripts/predict.py)); сборка обеих
стадий — в [`two_stage.py`](src/alfa_cred/two_stage.py). Всё обучается с нуля,
запускается через Docker (см. README).

## Финальный сабмит (2026-05-26, 14:05)

Ad-hoc сборка из лучших B-моделей:

```
bBalanced (12) + pseudo (2) + crossobj (2) = 16 моделей B-blend
+ record_11 (11 моделей) для A с hard-rule
→ LB 92.0504
```

Состав: 16 B-only моделей (top-3 каждого типа архитектуры + 2 pseudo-labeling +
2 cross-objective) — перечислены в Приложении B и в секциях Pipeline M/N выше.

## Топ-10 сабмитов по LB

| Файл | LB | Состав |
|------|----|--------|
| two_stage_r11_bBalanced_plus_pseudo_crossobj_1405 | **92.0504** | A=record_11, B=bBalanced(12)+pseudo(2)+crossobj(2) |
| two_stage_record11_plus_bBalanced_plus_bO_0622 | 92.0494 | A=record_11, B=bBalanced + Pipeline O (18) |
| three_stage_v2c_megaRA_with_subgOt2_1300 | 92.0486 | three-stage с subg_ot2 для RA |
| two_stage_record11_plus_bAllKLMNOP_0826 | 92.0458 | 51 B-моделей всех pipeline |
| two_stage_record11_plus_bMega_0245 | 92.0432 | bMega = K+L+M (35 моделей) |
| two_stage_r11_top1each_plus_psV2_crossobj_1405 | 92.0414 | ultra-clean 8 моделей |
| two_stage_r11_bBalanced_plus_crossobj_1405 | 92.0398 | bBalanced + crossobj без pseudo |
| two_stage_record11_plus_bAllKLMNO_0622 | 92.0386 | 38 моделей без P |
| two_stage_r11_bBalanced_plus_pseudo_1405 | 92.0384 | bBalanced + pseudo без crossobj |
| three_stage_r11_bBalancedO_plus_subgOt2_added_RA_1405 | 92.0347 | three-stage variant |

## Что не сработало (сводно)

- **Tabular DL** (Pipeline H): FT-Transformer, TabNet — слабее GBM.
- **Customer history** (Pipeline I): мало пересечений app_id.
- **Per-epoch модели** (Pipeline I): CV завышен на сабсете.
- **Deep Optuna XGB** (Pipeline J): переобучение, CV растёт — LB падает.
- **LB-weighted blend** (Pipeline J): uniform лучше.
- **Pairwise binary classification** (Pipeline P): не работает.
- **Subgroup как замена** (Pipeline P, three-stage v2a): катастрофа.
- **AutoEncoder embeddings** (Pipeline O): нет скрытого сигнала.
- **MI scan на 280 фичах** (Pipeline O): топ-MI 0.06 в шуме.
- **Single B-model без record_11** (например, two_stage_record11_plus_bN_0245):
  катастрофа — A-задача обязана идти через record_11+hard-rule.

## Что подтвердилось

- Расщепление по `pil1mtrx_offer` (two-stage) даёт **+0.027 LB**, ни одна
  отдельная техника близко не подошла.
- bBalanced (12-16 моделей разных архитектур) — оптимальный размер blend.
- Multi-seed XGB ensemble даёт **+0.024** от одиночки.
- Pseudo-labeling + cross-objective — **+0.004 каждое** в небольшом blend.
- Even с CV 0.20 модель вкладывается в большой ансамбль через
  ensemble-dropout эффект.

## CV vs LB парадокс

Несколько ярких случаев, где CV не предсказывал LB:

| Модель | CV | Одиночка LB |
|--------|----|-------------|
| lgbm_extended_tuned_seed123 | 0.9170 (топ) | 91.7962 |
| lgbm_boot_v_s256 | 0.9165 | **91.8774** (топ одиночка) |
| xgb_deep_optuna | **0.9181** (рекорд CV) | 91.8349 (слабее default 91.8215) |
| lgbm_epoch_post | 0.9521 (на сабсете) | 91.4377 |

Bootstrap-вариативность даёт лучшую генерализацию, чем точная
Optuna-настройка. CV полезен только для отсечки моделей ниже
эмпирической границы 0.913, дальше — доверяй LB.

## Приложение A. Полная таблица сабмитов (LB)

Зафиксированный LB-результат каждого сабмита за весь ход экспериментов
(NDCG@5 × 100). Сгруппировано по фазам.

| Сабмит / бленд | LB | Состав / заметка |
|----------------|----|------------------|
| candidate_selfcontained (pointwise-MLP) | **92.1957** | A=наш + B=0.70·b_blend+0.30·MLP — финальный рекорд |
| two_stage_drop5_plus_drop7_bNew | 92.0532 | mix 32 модели (drop-5 + drop-7 subsets) |
| two_stage_5subsets_bNew | 92.0532 | 80 моделей (5 subsets) — равно рекорду фазы |
| two_stage_r11_bBalanced_plus_pseudo_crossobj | 92.0504 | record_11 + bBalanced(12)+pseudo(2)+crossobj(2) |
| two_stage_record11_plus_bBalanced_plus_bO | 92.0494 | 18 моделей (bBalanced + Pipeline O) |
| three_stage_v2c_megaRA_with_subgOt2 | 92.0486 | subg_ot2 в mega-pool только для RA |
| two_stage_record11_plus_bAllKLMNOP | 92.0458 | 51 B-модель всех pipeline |
| two_stage_drop43_bNew_record_orig | 92.0428 | drop-43 (PSI scan) — не помог одиночно |
| two_stage_record11_plus_bMega | 92.0432 | bMega = K+L+M+N (35 моделей) |
| two_stage_r11_top1each_plus_psV2_crossobj | 92.0414 | ultra-clean 8 моделей |
| two_stage_r11_bBalanced_plus_crossobj | 92.0398 | bBalanced + crossobj без pseudo |
| two_stage_record11_plus_bAllKLMNO | 92.0386 | 38 моделей (без P) |
| two_stage_r11_bBalanced_plus_pseudo | 92.0384 | bBalanced + pseudo без crossobj |
| two_stage_3subsets_d5_d7_d2 | 92.0389 | 3 subsets — хуже mix на −0.014 |
| two_stage_drop2_9_time_bNew | 92.0378 | только новые 3 subsets — хуже |
| three_stage_r11_bBalancedO_plus_subgOt2 | 92.0347 | three-stage variant |
| two_stage_record11_plus_bKLM_plus_crossobj | 92.0324 | KLM + 2 crossobj (31) |
| two_stage_record11_plus_bBalanced (12) | 92.0317 | top-3 каждого типа (Pipeline M) |
| two_stage_drop7grp_bNew_record_orig | 92.0296 | одиночный drop-7 хуже mix |
| two_stage_record11_plus_bKLM_plus_pseudo | 92.0288 | KLM + 3 pseudo (32) |
| two_stage_record11_plus_bFull (K+L+M) | 92.0263 | 34 модели (без N) |
| two_stage_record11_plus_bMnew | 92.0232 | только M (22, без K/L/N) |
| two_stage_record11_plus_bAllL | 92.0006 | 5 XGB Optuna + 1 LGBM ext (Pipeline L) |
| two_stage_record11_plus_bTop1 | 91.9939 | xgb_b_default одиночка — прорыв two-stage (K) |
| two_stage_record11_plus_bAllXGB | 91.9765 | 5 multi-seed XGB Optuna |
| hybrid_record11_bw5 | 91.9679 | гибрид record + B-only (хуже чистой замены) |
| blend_11_no_cb_extended (record_11) | 91.9668 | прежний рекорд до two-stage |
| blend_record11_plus_h_ft | 91.9601 | record_11 + 2 FT-Transformer (Pipeline H) |
| blend_record11_plus_i_cb | 91.9497 | record_11 + cb_deep_optuna (Pipeline I) |
| blend_mega_strong_only_d | 91.9471 | mega-blend 12 моделей (Pipeline D) |
| blend_12_plus_e_top3 | 91.9427 | 12 + top-3 Pipeline E |
| blend_record11_plus_i_all | 91.9414 | record_11 + 4 модели Pipeline I |
| blend_record11_plus_j_xgb | 91.9418 | record_11 + XGB Optuna → −0.025 |
| blend_record11_plus_j_lbweighted | 91.9354 | LB-weighted blend → −0.031 |
| blend_record11_plus_h_all | 91.9247 | record_11 + 2 FT-T + TabNet → −0.042 |
| two_stage_record11_plus_bL_plus_Mcb | 91.9293 | ПРОВАЛ −0.07: CB не сочетается с bAllL |
| lgbm_boot_v_s256 (одиночка) | 91.8774 | CV 0.9165 — король одиночек |
| lgbm_extended_features (одиночка) | 91.8648 | CV 0.9165 |
| cb_deep_optuna (одиночка) | 91.8477 | CV 0.9167 |
| xgb_deep_optuna (одиночка) | 91.8349 | CV 0.9181 (рекорд CV) → LB слабый (переобучение) |
| xgb_rank_ndcg (одиночка) | 91.8215 | CV 0.9163 |
| cb_yetirank_tuned (одиночка) | 91.8049 | CV 0.9162 |
| lgbm_extended_tuned_seed123 (одиночка) | 91.7962 | CV 0.9170 (топ CV) → LB слабее |
| lgbm_optuna_30t (одиночка) | 91.7512 | Optuna 30 trials |
| baseline_lgbm_lambdarank | 91.634 | бейзлайн |
| ft_trans_seed42 (одиночка) | 91.4893 | CV 0.9129 → LB провал |
| lgbm_epoch_post (одиночка) | 91.4377 | CV 0.9521 на сабсете → LB провал |
| three_stage_v2a_subgBoth | 91.3545 | КАТАСТРОФА: subg_ot1 как замена RA |
| tabnet_seed42 (одиночка) | 90.2362 | CV 0.8886 → катастрофа |
| two_stage_record11_plus_bN | 89.7603 | МЕГА-ПРОВАЛ: только Pipeline N без других |

## Приложение B. CV моделей record_11 и исключённые модели

CV NDCG@5 (5-fold GroupKFold) каждой из 11 моделей A-бленда record_11
(прежняя A-база; в финале заменена компактным 5-модельным A-блендом):

| Модель | CV NDCG@5 |
|--------|-----------|
| lgbm_extended_tuned_seed42 | 0.9165 |
| lgbm_extended_tuned_seed123 | 0.9170 |
| lgbm_extended_tuned_seed777 | 0.9168 |
| cb_yetirank_tuned | 0.9162 |
| cb_yetirank_tuned_seed123 | 0.9158 |
| xgb_rank_ndcg | 0.9163 |
| cb_pairlogit | 0.9160 |
| lgbm_pseudo_label | 0.9161 |
| lgbm_oof_full | 0.9165 |
| lgbm_time_aware | 0.9159 |
| lgbm_bootstrap | 0.9155 |

**Эмпирическая граница CV ≈ 0.913**, ниже которой модель «размывает» blend
(выведена сравнением blend из 12 моделей с cb_yetirank_extended (CV 0.9128) →
91.9471 против 11 моделей без неё → 91.9668, разница +0.0197).

Исключённые как «размывающие» (CV ниже границы или доказанный вред на LB):
`cb_yetirank_extended` (0.9128), `lgbm_xendcg` (0.9081), `cb_queryrmse` (0.9102),
`mlp_listnet` (0.9088), `mlp_distill` (0.7506, ошибочная дистилляция),
`ft_trans_seed42/123` (0.9129/0.9119), `tabnet_seed42` (0.8886),
`lgbm_epoch_post/pre` (CV завышен на сабсете), `cb_deep_optuna` (0.9167, но −0.017
в blend — переобучение Optuna), `lgbm_adv_pruned`/`lgbm_stacking_v2` (stacking
переобучается на OOF).
