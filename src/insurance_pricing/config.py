"""Typed configuration loaded from ``config/config.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectCfg(BaseModel):
    name: str = "insurance-pricing"
    seed: int = 42


class DataCfg(BaseModel):
    fremtpl_freq: str = "freMTPL2freq"
    fremtpl_sev: str = "freMTPL2sev"
    cache_dir: str = "data/raw"
    quotes_file: str = "data/raw/vehicle_insurance_cross_sell.csv"


class RiskCfg(BaseModel):
    exposure_col: str = "Exposure"
    claim_count_col: str = "ClaimNb"
    max_claim_amount: float = 200_000.0
    exposure_cap: float = 1.0
    categorical: list[str] = Field(default_factory=lambda: ["VehBrand", "VehGas", "Area", "Region"])
    numeric: list[str] = Field(
        default_factory=lambda: ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    )


class BayesianCfg(BaseModel):
    group_cols: list[str] = Field(default_factory=lambda: ["Region", "VehBrand"])
    coarse_group_cols: list[str] = Field(default_factory=lambda: ["Region"])
    draws: int = 3000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.95


class DemandCfg(BaseModel):
    price_col: str = "Annual_Premium"
    target_col: str = "Response"
    group_col: str = "Region_Code"


class PricingCfg(BaseModel):
    price_grid_lo: float = 0.5
    price_grid_hi: float = 3.0
    price_grid_n: int = 60


class ValidationCfg(BaseModel):
    test_size: float = 0.2


class PathsCfg(BaseModel):
    models_dir: str = "models"
    reports_dir: str = "reports"


class Config(BaseModel):
    """Root configuration object."""

    project: ProjectCfg = ProjectCfg()
    data: DataCfg = DataCfg()
    risk: RiskCfg = RiskCfg()
    bayesian: BayesianCfg = BayesianCfg()
    demand: DemandCfg = DemandCfg()
    pricing: PricingCfg = PricingCfg()
    validation: ValidationCfg = ValidationCfg()
    paths: PathsCfg = PathsCfg()


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Load and validate the YAML config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as fh:
        return Config(**yaml.safe_load(fh))
