"""Зафиксированные составы лучших ансамблей по итогам экспериментов.

Документ описывает, какие модели входят в каждый из топовых сабмитов и
какой LB-результат они дают. Используется для воспроизводимости — чтобы
повторно собрать тот же blend по test_scores в `oof/`.

Метрика на лидерборде — NDCG@5 × 100.

История рекордов:
- baseline LGBM LambdaRank → 91.634
- LGBM Optuna → 91.7512
- LGBM extended features → 91.8648
- blend (mega_strong_only_d, 12 моделей) → 91.9471
- **blend (11 моделей без cb_extended) → 91.9668 ← текущий рекорд**
"""

from __future__ import annotations

# НОВЫЙ РЕКОРД LB = 91.9939 (Pipeline K, two-stage):
# для запросов подзадачи A (есть pil1mtrx_offer=1) → blend record_11 + hard-rule
# для запросов подзадачи B (нет pil1mtrx_offer) → xgb_b_default (B-only XGB)
# Файл: two_stage_record11_plus_bTop1_1251.csv
# Прирост +0.0271 от предыдущего рекорда 91.9668.
# Ключевая идея: B-only XGB обученный ТОЛЬКО на 34% запросов подзадачи B
# даёт лучший сигнал для этих запросов чем record_11, обученный на всём train.
RECORD_TWO_STAGE_LB_91_9939 = {
    "subtask_a_blend": "record_11",          # rank-avg blend record_11
    "subtask_b_model": "xgb_b_default_1244",  # B-only XGB на 12,650 B-train запросах
    "lb": 91.9939,
}

# Предыдущий рекорд LB = 91.9668. Состав:
# 11 моделей с CV NDCG@5 в диапазоне 0.9155 - 0.9170, выбранных по принципу
# «все модели сильнее эмпирической границы CV ≈ 0.913 — а cb_yetirank_extended
# с CV 0.9128 убран, потому что размывал blend (его исключение дало +0.02 на LB)».
RECORD_11_MODELS_LB_91_9668 = (
    "lgbm_extended_tuned_seed42_20260523_0226",
    "lgbm_extended_tuned_seed123_20260523_0245",
    "lgbm_extended_tuned_seed777_20260523_0252",
    "cb_yetirank_tuned_20260523_0422",
    "cb_yetirank_tuned_seed123_20260523_0434",
    "xgb_rank_ndcg_20260523_0621",
    "cb_pairlogit_20260523_0633",
    "lgbm_pseudo_label_20260523_0640",
    "lgbm_oof_full_20260523_0759",
    "lgbm_time_aware_20260523_1356",
    "lgbm_bootstrap_20260523_1406",
)

# Предыдущий рекорд (12 моделей, LB = 91.9471). Отличается от текущего
# добавленной моделью cb_yetirank_extended_20260523_0232 (CV 0.9128).
RECORD_12_MODELS_LB_91_9471 = RECORD_11_MODELS_LB_91_9668 + (
    "cb_yetirank_extended_20260523_0232",
)

# CV NDCG@5 каждой модели рекордного blend (мера «силы» одиночки).
MODEL_CV_NDCG5 = {
    "lgbm_extended_tuned_seed42_20260523_0226": 0.9165,
    "lgbm_extended_tuned_seed123_20260523_0245": 0.9170,
    "lgbm_extended_tuned_seed777_20260523_0252": 0.9168,
    "cb_yetirank_extended_20260523_0232": 0.9128,  # ↓ ниже эмпирической границы
    "cb_yetirank_tuned_20260523_0422": 0.9162,
    "cb_yetirank_tuned_seed123_20260523_0434": 0.9158,
    "xgb_rank_ndcg_20260523_0621": 0.9163,
    "cb_pairlogit_20260523_0633": 0.9160,
    "lgbm_pseudo_label_20260523_0640": 0.9161,
    "lgbm_oof_full_20260523_0759": 0.9165,
    "lgbm_time_aware_20260523_1356": 0.9159,
    "lgbm_bootstrap_20260523_1406": 0.9155,
}

# Эмпирически выведенная граница CV NDCG@5, ниже которой модель «размывает» blend.
# Получена сравнением blend_mega_strong_only_d (12, с cb_extended) → 91.9471
# и blend_11_no_cb_extended_f (11, без cb_extended) → 91.9668: разница +0.0197.
EMPIRICAL_BLEND_THRESHOLD_CV = 0.913

# Состав blend, который УХУДШАЕТ результат — для исключения из будущих сборок.
KNOWN_WEAK_MODELS = (
    # CV < 0.913
    "lgbm_xendcg_20260523_0626",        # CV 0.9081
    "cb_queryrmse_20260523_0636",       # CV 0.9102
    "mlp_listnet_20260523_0749",        # CV 0.9088 (MLP в одиночку слабее GBM)
    "mlp_distill_20260523_0814",        # CV 0.7506 (ошибочная дистилляция)
    "cb_yetirank_extended_20260523_0232",  # CV 0.9128 (на границе, размывает)
    # «не помогающие» эксперименты с CV около границы:
    "lgbm_adv_pruned_20260523_0807",    # CV 0.9158 (но drift pruning не дал выигрыша на LB)
    "lgbm_stacking_v2_20260523_0814",   # CV 0.9161 (stacking переобучается на OOF)
    # Pipeline H (tabular DL) — доказано на LB: добавление к рекорду ухудшает.
    # FT-T 2 модели → LB 91.9601 (-0.0067), + tabnet → LB 91.9247 (-0.042).
    # Корреляция рангов TabNet с GBM = 0.764 (очень diverse), но diversity
    # без минимальной точности (CV ≥ 0.913) не работает — наоборот размывает.
    "ft_trans_seed42_0845",             # CV 0.9129
    "ft_trans_seed123_1230",            # CV 0.9119
    "tabnet_seed42_1149",               # CV 0.8886
    # Pipeline I — per-epoch модели катастрофически слабые на полном test.
    # lgbm_epoch_post одиночкой LB 91.4377 (CV на post-сабсете 0.9521 был
    # обманчиво высоким — внутри-эпохи менее сложная задача).
    "lgbm_epoch_post_2025",             # CV 0.9521 на post-сабсете, LB 91.4377
    "lgbm_epoch_pre_2021",              # CV 0.9390 на pre-сабсете
    # cb_deep_optuna имел CV 0.9167 (выше границы 0.913), но в blend ухудшил
    # на -0.017. Гипотеза: переобучение Optuna на 50% train сэмпле.
    "cb_deep_optuna_2342",              # CV 0.9167, blend -0.017
)


def expected_lb_score(blend_name: str) -> float | None:
    """Возвращает зафиксированный LB-результат для известного blend'а."""
    return {
        # === Pipeline M+N: новый рекорд через большой/сбалансированный B-blend ===
        "two_stage_record11_plus_bBalanced_0229": 92.0317,    # НОВЫЙ РЕКОРД! top-3 каждого типа (12 моделей)
        "two_stage_record11_plus_bKLM_plus_pseudo_0245": 92.0288,  # KLM + 3 pseudo (32 модели)
        "two_stage_record11_plus_bL_plus_Mcb_0229": 91.9293,  # ПРОВАЛ -0.07: CB не сочетается с bAllL
        # === Pipeline L: ПРОБИТО 92+ через multi-seed B-only XGB + LGBM ===
        "two_stage_record11_plus_bAllL_1914": 92.0006,        # 5 XGB Optuna + 1 LGBM ext
        "two_stage_record11_plus_bAllXGB_1914": 91.9765,      # 5 multi-seed XGB Optuna
        "two_stage_record11_plus_bXGBseedBlend_1914": 91.9717,
        "two_stage_record11_plus_bXGBOptuna_1914": 91.9528,   # одиночный XGB Optuna (хуже default!)
        "two_stage_record11_plus_bXGB_LGBM_1914": 91.9211,    # маленький B-blend (2 модели) хуже
        # === Pipeline K: первый прорыв через two-stage ===
        "two_stage_record11_plus_bTop1_1251": 91.9939,        # xgb_b_default одиночка (КОРОЛЬ B-only одиночек)
        "two_stage_record11_plus_bAll_1251": 91.9358,
        "two_stage_record11_plus_bTop3_1251": 91.9328,
        "hybrid_record11_bw5_1251": 91.9679,
        # === Старый рекорд (до two-stage) ===
        "blend_11_no_cb_extended_f": 91.9668,
        "blend_record11_plus_h_ft_1708": 91.9601,   # record_11 + 2 FT-T (Pipeline H)
        "blend_record11_plus_h_all_1708": 91.9247,  # record_11 + 2 FT-T + TabNet
        "blend_record11_plus_i_cb_2355": 91.9497,   # record_11 + cb_deep_optuna (Pipeline I)
        "blend_record11_plus_i_all_2355": 91.9414,  # record_11 + все 4 модели Pipeline I
        # Одиночки (LB как singleton submission, без blend и без hard-rule
        # boost от других моделей — есть только pil1mtrx hard-rule):
        "lgbm_epoch_post_2025": 91.4377,            # CV 0.9521 (sub) → LB провал
        "lgbm_boot_v_s256_2202": 91.8774,           # CV 0.9165 → LB КОРОЛЬ одиночек
        "lgbm_extended_features": 91.8648,          # CV 0.9165 → LB сильный
        "xgb_deep_optuna_0532": 91.8349,            # CV 0.9181 (новый рекорд CV!) → LB слабый, переобучение Optuna
        "blend_record11_plus_j_xgb_0617": 91.9418,        # record_11 + XGB Optuna → -0.025
        "blend_record10_no_weak_plus_top3_0617": 91.9547, # без extended_seed123 + top-3 новых → -0.012
        "blend_record11_plus_j_lbweighted_0617": 91.9354, # LB-weighted blend → -0.031
        "cb_deep_optuna_2342": 91.8477,             # CV 0.9167 → LB второй
        "xgb_rank_ndcg_20260523_0621": 91.8215,
        "cb_yetirank_tuned_20260523_0422": 91.8049,
        "lgbm_extended_tuned_seed123_20260523_0245": 91.7962,  # CV 0.9170, LB слабее
        "lgbm_optuna_30t_20260522_2103_full": 91.7512,
        "baseline_lgbm_lambdarank": 91.634,
        "ft_trans_seed42_0845": 91.4893,            # CV 0.9129 → LB провал
        "tabnet_seed42_1149": 90.2362,              # CV 0.8886 → LB катастрофа
        "blend_mega_strong_only_d_20260523_1441": 91.9471,
        "blend_12_plus_e_top3_f": 91.9427,
        "blend_record12_cv2_weighted_f": 91.9315,
        "blend_mega_strong_v3_20260523_1751": 91.8840,
        "blend_hill_weighted_20260523_1754": 91.8825,
        "blend_mega_20260523_0651": 91.8679,
        "blend_top5_strict_d": 91.8289,
        "blend_3lgb_plus_3xgb_div_f": 91.8077,
        "blend_3lgb_only_d": 91.7681,
        "blend_weighted_v2_20260523_0822": 91.5291,
    }.get(blend_name)
