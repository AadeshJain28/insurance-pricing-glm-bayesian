"""Project logging."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "insurance_pricing", config_path: str | Path = "config/logging.yaml"):
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            logging.config.dictConfig(yaml.safe_load(fh))
    elif not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FMT)
    return logging.getLogger(name)
