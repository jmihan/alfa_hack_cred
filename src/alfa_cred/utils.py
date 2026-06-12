"""Сервисные утилиты: логирование, фиксация сидов, замер времени."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


def get_logger(name: str = "alfa_cred", level: int = logging.INFO) -> logging.Logger:
    """Возвращает настроенный логгер с компактным форматом."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def seed_everything(seed: int = 42) -> None:
    """Фиксирует сиды random/numpy и PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Контекстный менеджер для замера времени блока кода."""
    log = logger or get_logger()
    start = time.perf_counter()
    log.info("%s: старт", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s: завершено за %.2f сек", label, elapsed)
