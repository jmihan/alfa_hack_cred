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

# Текущий рекорд LB = 91.9668. Состав:
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
)


def expected_lb_score(blend_name: str) -> float | None:
    """Возвращает зафиксированный LB-результат для известного blend'а."""
    return {
        "blend_11_no_cb_extended_f": 91.9668,
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
