from __future__ import annotations

from pathlib import Path

import pytest

from insurance_pricing.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_loads_and_matches_yaml():
    cfg = load_config(ROOT / "config" / "config.yaml")
    assert cfg.project.seed == 42
    assert cfg.risk.exposure_cap == 1.0
    assert cfg.demand.price_col == "Annual_Premium"
    assert cfg.demand.target_col == "Response"
    assert "Region" in cfg.bayesian.group_cols
    assert 0 < cfg.validation.test_size < 1


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "config" / "nope.yaml")
