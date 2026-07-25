"""Vehicle-insurance cross-sell loader (demand side).

Used for a **conversion/propensity** model, NOT for price elasticity: premium in
this data is set from risk, so the price-conversion association is confounded
(see :mod:`insurance_pricing.evaluation.endogeneity`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insurance_pricing.logger import get_logger

logger = get_logger("insurance_pricing.load_quotes")

PLACEHOLDER_PREMIUM = 2630.0


def load_quotes(path: str | Path = "data/raw/vehicle_insurance_cross_sell.csv") -> pd.DataFrame:
    """Load and clean the cross-sell data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Quote data not found at {path.resolve()}")
    df = pd.read_csv(path)

    # 17% of premiums equal a single placeholder value -> flag rather than drop
    df["is_placeholder_premium"] = (df["Annual_Premium"] == PLACEHOLDER_PREMIUM).astype("int8")
    df["log_premium"] = np.log(df["Annual_Premium"].clip(lower=1.0))
    df["Vehicle_Damage_bin"] = (df["Vehicle_Damage"] == "Yes").astype("int8")
    df["Gender_bin"] = (df["Gender"] == "Male").astype("int8")
    df["Region_Code"] = df["Region_Code"].astype(int)
    df["vehicle_age_ord"] = df["Vehicle_Age"].map(
        {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
    ).astype("int8")

    logger.info(
        "Quotes=%d | conversion=%.4f | regions=%d | placeholder premium=%.1f%%",
        len(df), float(df["Response"].mean()), int(df["Region_Code"].nunique()),
        100 * float(df["is_placeholder_premium"].mean()),
    )
    return df
