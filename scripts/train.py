"""Обучение модели по YAML-конфигу с логированием в MLflow.

Пример:
    python scripts/train.py --config configs/baseline_lgbm.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

from alfa_cred.config import (
    MLRUNS_DIR,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
)
from alfa_cred.features.pipeline import build_feature_table, feature_columns
from alfa_cred.inference import (
    apply_pil1mtrx_hard_rule,
    make_submission,
    verify_submission,
)
from alfa_cred.io_utils import load_raw, sort_by_request
from alfa_cred.tracking import setup_mlflow, start_run
from alfa_cred.training import predict_full, train_cv
from alfa_cred.utils import get_logger, seed_everything, timer
from alfa_cred.validation import make_cv_splits

LOG = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение LTR-модели и формирование сабмита")
    parser.add_argument("--config", type=Path, required=True, help="Путь к YAML-конфигу")
    parser.add_argument("--no-submission", action="store_true",
                        help="Не формировать сабмит-файл")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("cv", {}).get("random_state", 42))

    experiment_name = config.get("experiment_name", "default")
    run_name = config.get("run_name", f"run_{datetime.now():%Y%m%d_%H%M%S}")
    setup_mlflow(experiment_name, tracking_uri=MLRUNS_DIR)

    with start_run(run_name, tags={"config": args.config.name}) as run:
        mlflow.log_artifact(str(args.config))
        mlflow.log_params(_flatten_params(config))

        with timer("load_raw"):
            df_train, df_test, df_features = load_raw()

        with timer("build features (train)"):
            train_feats = build_feature_table(
                df_train, df_features,
                min_fill_rate=config.get("features", {}).get("min_fill_rate", 0.5),
                is_train=True,
            )
        with timer("build features (test)"):
            test_feats = build_feature_table(
                df_test, df_features,
                min_fill_rate=config.get("features", {}).get("min_fill_rate", 0.5),
                is_train=False,
            )

        feature_cols, categorical_cols = feature_columns(train_feats)
        # Гарантируем, что в test есть те же колонки в том же порядке
        for col in feature_cols:
            if col not in test_feats.columns:
                test_feats[col] = 0
        LOG.info("Фичей всего: %d, категориальных: %d", len(feature_cols), len(categorical_cols))
        mlflow.log_metric("n_features", len(feature_cols))
        mlflow.log_metric("n_categorical", len(categorical_cols))
        mlflow.log_metric("train_rows", len(train_feats))
        mlflow.log_metric("test_rows", len(test_feats))

        # Категориальные колонки в LightGBM должны быть int или category
        train_feats, test_feats = _encode_categoricals(train_feats, test_feats, categorical_cols)

        cv_scheme = config.get("cv", {}).get("scheme", "group")
        n_splits = int(config.get("cv", {}).get("n_splits", 5))
        cv_splits = list(make_cv_splits(train_feats, scheme=cv_scheme, n_splits=n_splits))

        with timer("CV training"):
            result = train_cv(
                df_train=train_feats,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                cv_splits=cv_splits,
                params=config.get("model", {}).get("params", {}),
                n_estimators=int(config.get("model", {}).get("n_estimators", 2000)),
                early_stopping_rounds=int(config.get("model", {}).get("early_stopping_rounds", 100)),
                run_name=run_name,
                apply_pil1mtrx_rule=bool(config.get("inference", {}).get("apply_pil1mtrx_rule", True)),
            )

        LOG.info("Топ-15 признаков по gain:")
        LOG.info("\n%s", result.feature_importance.head(15).to_string())

        if args.no_submission:
            LOG.info("Флаг --no-submission установлен, сабмит не формируется")
            return

        with timer("predict test"):
            test_sorted = sort_by_request(test_feats)
            scores = predict_full(
                models=result.models,
                df_test=test_sorted,
                feature_cols=feature_cols,
                apply_pil1mtrx_rule=False,  # применяем ниже единообразно
            )
            scores = apply_pil1mtrx_hard_rule(test_sorted, scores)

        submission_path = SUBMISSIONS_DIR / f"{run_name}.csv"
        make_submission(test_sorted, scores, submission_path)
        verify_submission(submission_path, SAMPLE_SUBMISSION_PATH)
        mlflow.log_artifact(str(submission_path), artifact_path="submission")

        _append_submission_log(
            run_name=run_name,
            submission_path=submission_path,
            cv_ndcg=result.mean_ndcg,
            cv_std=result.std_ndcg,
            cv_a=result.mean_ndcg_subtask_a,
            cv_b=result.mean_ndcg_subtask_b,
            mlflow_run_id=run.info.run_id,
        )

        summary = {
            "run_name": run_name,
            "cv_ndcg5_mean": result.mean_ndcg,
            "cv_ndcg5_std": result.std_ndcg,
            "cv_ndcg5_subtask_a": result.mean_ndcg_subtask_a,
            "cv_ndcg5_subtask_b": result.mean_ndcg_subtask_b,
            "submission_path": str(submission_path),
            "n_features": len(feature_cols),
        }
        LOG.info("Сводка: %s", json.dumps(summary, ensure_ascii=False, indent=2))


def _flatten_params(config: dict, parent: str = "") -> dict[str, str]:
    """MLflow ожидает плоский словарь параметров (str: str)."""
    out: dict[str, str] = {}
    for k, v in config.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        else:
            out[key] = str(v)
    return out


def _encode_categoricals(train_feats, test_feats, categorical_cols):
    """Превращает object-колонки в pandas `category` с общими кодами.

    LightGBM поддерживает категориальные напрямую, если они int или
    pandas Categorical. Кодируем единообразно по train+test, чтобы
    модель видела одинаковые коды на этапе инференса.
    """
    import pandas as pd

    for col in categorical_cols:
        if col not in train_feats.columns:
            continue
        combined = pd.concat([train_feats[col], test_feats[col]], axis=0, ignore_index=True)
        cat = combined.astype("category")
        train_feats[col] = cat.iloc[: len(train_feats)].reset_index(drop=True).cat.codes.astype("int32")
        test_feats[col] = cat.iloc[len(train_feats):].reset_index(drop=True).cat.codes.astype("int32")
    return train_feats, test_feats


def _append_submission_log(
    run_name: str,
    submission_path: Path,
    cv_ndcg: float,
    cv_std: float,
    cv_a: float,
    cv_b: float,
    mlflow_run_id: str,
) -> None:
    log_path = SUBMISSIONS_DIR / "log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as fh:
        if is_new:
            fh.write(
                "# Журнал сабмитов\n\n"
                "| Дата | Эксперимент | CV NDCG@5 | CV A | CV B | LB | MLflow run | Файл |\n"
                "|------|-------------|-----------|------|------|----|------------|------|\n"
            )
        fh.write(
            f"| {datetime.now():%Y-%m-%d %H:%M} | {run_name} | {cv_ndcg:.4f} ± {cv_std:.4f} | "
            f"{cv_a:.4f} | {cv_b:.4f} | — | {mlflow_run_id[:8]} | {submission_path.name} |\n"
        )


if __name__ == "__main__":
    main()
