"""CLI-обёртка для генерации сабмита из сохранённой модели.

Реализация наполняется в фазе 2 (после EDA).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Формирование сабмита из обученной модели")
    parser.add_argument("--config", type=Path, required=True, help="Путь к YAML-конфигу")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"Реализация будет добавлена в фазе 2 после EDA. Конфиг: {args.config}"
    )


if __name__ == "__main__":
    main()
