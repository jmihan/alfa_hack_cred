"""Обёртки MLflow для единообразного логирования экспериментов."""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import mlflow

from alfa_cred.config import MLRUNS_DIR
from alfa_cred.utils import get_logger

LOG = get_logger(__name__)


def setup_mlflow(experiment: str, tracking_uri: Path = MLRUNS_DIR) -> None:
    """Настраивает локальный MLflow tracking URI и эксперимент."""
    tracking_uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_uri.as_posix()}")
    mlflow.set_experiment(experiment)


@contextmanager
def start_run(
    name: str,
    tags: Mapping[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Запускает MLflow run и автоматически добавляет git-метаданные."""
    git_meta = _git_metadata()
    merged_tags = {**git_meta, **(tags or {})}
    with mlflow.start_run(run_name=name, tags=merged_tags) as run:
        LOG.info("MLflow run %s стартовал (id=%s)", name, run.info.run_id)
        yield run


def log_dict(payload: Mapping[str, object], filename: str) -> None:
    """Логирует словарь как JSON-артефакт."""
    out = Path(filename)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    mlflow.log_artifact(str(out))
    out.unlink(missing_ok=True)


def _git_metadata() -> dict[str, str]:
    """Возвращает короткий SHA и имя ветки текущего репозитория."""
    meta: dict[str, str] = {}
    try:
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        meta["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return meta
