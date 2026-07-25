"""Feature preparation for the risk (freMTPL2) and demand (cross-sell) models."""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
RISK_CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]

DEMAND_NUMERIC = [
    "Age", "log_premium", "Vintage", "vehicle_age_ord",
    "Vehicle_Damage_bin", "Gender_bin", "Previously_Insured", "is_placeholder_premium",
]


def prepare_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard actuarial capping and derive model-ready risk features.

    Caps follow common pricing practice: extreme ages/bonus-malus values are
    sparse and unstable, so they are truncated rather than allowed to drive
    the fit. Density is log-transformed (it spans several orders of magnitude).
    """
    out = df.copy()
    out["VehAge"] = out["VehAge"].clip(upper=20)
    out["DrivAge"] = out["DrivAge"].clip(lower=18, upper=90)
    out["BonusMalus"] = out["BonusMalus"].clip(upper=150)
    out["VehPower"] = out["VehPower"].clip(upper=15)
    out["LogDensity"] = np.log1p(out["Density"].astype(float))
    for c in RISK_CATEGORICAL:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def severity_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for a severity model: a claim actually occurred.

    Severity is *conditional* on a claim — fitting it on zero-claim policies is
    a classic error (and Gamma is undefined at zero).
    """
    sub = df[(df["ClaimNb"] > 0) & (df["ClaimAmount"] > 0)].copy()
    sub["AvgClaimAmount"] = sub["ClaimAmount"] / sub["ClaimNb"]
    return sub.reset_index(drop=True)


def prepare_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model-ready demand features (conversion/propensity)."""
    out = df.copy()
    out["Region_Code"] = out["Region_Code"].astype(int)
    out["Policy_Sales_Channel"] = out["Policy_Sales_Channel"].astype(int)
    return out


def train_test_split_frame(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, stratify_col: str | None = None):
    """Random split (freMTPL2 has no time dimension), optionally stratified."""
    from sklearn.model_selection import train_test_split

    strat = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
