"""Подбор гиперпараметров LightGBM через Optuna.

Сохраняет лучший набор гиперпараметров в новый YAML-конфиг, который
потом запускает `scripts/train.py` для полного CV и формирования
сабмита.

Пример:
    python scripts/tune.py --config configs/baseline_lgbm.yaml \\
        --out configs/lgbm_optuna.yaml --n-trials 30
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from alfa_cred.config import CONFIGS_DIR, MLRUNS_DIR
from alfa_cred.features.pipeline import build_feature_table, feature_columns
from alfa_cred.io_utils import load_raw
from alfa_cred.tracking import setup_mlflow, start_run
from alfa_cred.tuning import run_optuna
from alfa_cred.utils import get_logger, seed_everything, timer

LOG = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Тюнинг гиперпараметров LightGBM через Optuna")
    parser.add_argument("--config", type=Path, required=True,
                        help="Базовый YAML-конфиг (берутся пути features.min_fill_rate и т. п.)")
    parser.add_argument("--out", type=Path, default=CONFIGS_DIR / "lgbm_optuna.yaml",
                        help="Куда сохранить YAML с лучшими параметрами")
    parser.add_argument("--n-trials", type=int, default=30, help="Число итераций Optuna")
    parser.add_argument("--n-splits", type=int, default=3, help="Число фолдов внутри trial")
    parser.add_argument("--n-estimators", type=int, default=1500,
                        help="Верхняя граница итераций бустинга внутри trial (с early stopping)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)
    seed_everything(base_config.get("cv", {}).get("random_state", 42))

    setup_mlflow("alfa_cred_tuning", tracking_uri=MLRUNS_DIR)
    run_name = f"optuna_{args.n_trials}t_{datetime.now():%Y%m%d_%H%M}"

    with start_run(run_name, tags={"stage": "tuning"}) as _:
        import mlflow
        mlflow.log_params({
            "n_trials": args.n_trials,
            "n_splits": args.n_splits,
            "n_estimators": args.n_estimators,
        })

        with timer("load_raw"):
            df_train, _, df_features = load_raw()
        with timer("build features"):
            train_feats = build_feature_table(
                df_train, df_features,
                min_fill_rate=base_config.get("features", {}).get("min_fill_rate", 0.5),
                is_train=True,
            )
        feature_cols, categorical_cols = feature_columns(train_feats)
        train_feats = _encode_categoricals(train_feats, categorical_cols)

        with timer("optuna"):
            result = run_optuna(
                df=train_feats,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                n_trials=args.n_trials,
                n_splits=args.n_splits,
                n_estimators=args.n_estimators,
            )

        mlflow.log_metric("optuna_best_value", result.best_value)
        mlflow.log_dict(result.best_params, "best_params.json")

        # Сохраняем итоговый конфиг для последующего полного train
        out_config = _build_output_config(base_config, result.best_params, run_name)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(out_config, fh, allow_unicode=True, sort_keys=False)
        mlflow.log_artifact(str(args.out))

        LOG.info("Лучший CV NDCG@5 на подзадаче B: %.5f", result.best_value)
        LOG.info("Конфиг сохранён: %s", args.out)


def _build_output_config(base: dict, best_params: dict, run_name: str) -> dict:
    """Создаёт новый YAML с подобранными параметрами и финальными настройками CV."""
    new = {**base}
    new["experiment_name"] = "alfa_cred"
    new["run_name"] = run_name.replace("optuna_", "lgbm_optuna_") + "_full"
    # Возвращаем «производственные» n_estimators и early_stopping_rounds
    model = new.setdefault("model", {})
    model["n_estimators"] = 4000
    model["early_stopping_rounds"] = 200
    base_params = dict(model.get("params", {}))
    # Перезаписываем подобранные параметры поверх дефолтных
    base_params.update(best_params)
    # Гарантируем нужные обязательные опции LambdaRank
    base_params.setdefault("objective", "lambdarank")
    base_params.setdefault("metric", "ndcg")
    base_params.setdefault("eval_at", [5])
    base_params.setdefault("lambdarank_truncation_level", 5)
    base_params.setdefault("verbose", -1)
    base_params.setdefault("n_jobs", -1)
    base_params.setdefault("seed", 42)
    model["params"] = base_params
    new["cv"] = {"scheme": "group", "n_splits": 5, "random_state": 42}
    new.setdefault("inference", {})["apply_pil1mtrx_rule"] = True
    return new


def _encode_categoricals(df, categorical_cols):
    """Превращает категориальные колонки в int32 кодами для LightGBM."""
    for col in categorical_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category").cat.codes.astype("int32")
    return df


if __name__ == "__main__":
    main()
