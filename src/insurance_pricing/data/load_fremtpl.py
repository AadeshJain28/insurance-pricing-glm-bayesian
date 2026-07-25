"""freMTPL2 loader (French motor third-party liability).

Fetched from OpenML via scikit-learn — no manual download. Joins the frequency
table (one row per policy, with Exposure) to the severity table (one row per
claim) to produce a policy-level frame with claim counts and total claim amount.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_pricing.logger import get_logger

logger = get_logger("insurance_pricing.load_fremtpl")


def fetch_fremtpl(cache_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (and disk-cache) the freMTPL2 frequency and severity tables."""
    from pathlib import Path

    from sklearn.datasets import fetch_openml

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    f_path, s_path = cache / "fremtpl2freq.parquet", cache / "fremtpl2sev.parquet"

    if f_path.exists() and s_path.exists():
        logger.info("Loading freMTPL2 from local cache")
        return pd.read_parquet(f_path), pd.read_parquet(s_path)

    logger.info("Fetching freMTPL2 from OpenML (first run only)…")
    freq = fetch_openml("freMTPL2freq", version=1, as_frame=True, parser="auto").frame
    sev = fetch_openml("freMTPL2sev", version=1, as_frame=True, parser="auto").frame
    freq.to_parquet(f_path, index=False)
    sev.to_parquet(s_path, index=False)
    return freq, sev


def build_policy_frame(
    freq: pd.DataFrame,
    sev: pd.DataFrame,
    exposure_cap: float = 1.0,
    max_claim_amount: float = 200_000.0,
) -> pd.DataFrame:
    """Join severity onto policies and apply standard actuarial cleaning.

    * Exposure > 1 year is a known data error in freMTPL2 -> capped.
    * Extreme claim amounts are capped (large-loss treatment).
    * Adds ``ClaimAmount`` (total per policy) and ``PurePremium`` (per year of exposure).
    """
    freq = freq.copy()
    freq.columns = [c.strip() for c in freq.columns]
    sev = sev.copy()
    sev.columns = [c.strip() for c in sev.columns]

    for col in ("IDpol",):
        if col in freq:
            freq[col] = freq[col].astype("int64")
        if col in sev:
            sev[col] = sev[col].astype("int64")

    sev["ClaimAmount"] = sev["ClaimAmount"].clip(upper=max_claim_amount)
    totals = sev.groupby("IDpol", as_index=False)["ClaimAmount"].sum()

    df = freq.merge(totals, on="IDpol", how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)
    df["Exposure"] = df["Exposure"].clip(upper=exposure_cap)
    df = df[df["Exposure"] > 0].reset_index(drop=True)

    # frequency target is a rate; severity is conditional on a claim occurring
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    df["AvgClaimAmount"] = np.where(df["ClaimNb"] > 0, df["ClaimAmount"] / df["ClaimNb"], 0.0)

    logger.info(
        "Policies=%d | claims=%d | claim rate=%.4f | mean exposure=%.3f",
        len(df), int(df["ClaimNb"].sum()), float(df["ClaimNb"].sum() / df["Exposure"].sum()),
        float(df["Exposure"].mean()),
    )
    return df
