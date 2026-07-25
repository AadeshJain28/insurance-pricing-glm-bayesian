"""Endogeneity diagnostic for observational price data.

A demand curve requires conversion to FALL as price rises. When price is set
from risk (as insurers do), the observed association is confounded and may even
be positive. This module makes that testable and reportable rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def price_response_table(
    df: pd.DataFrame, price_col: str, outcome_col: str, n_bins: int = 10
) -> pd.DataFrame:
    """Conversion rate by price decile."""
    work = df[[price_col, outcome_col]].copy()
    work["bin"] = pd.qcut(work[price_col].rank(method="first"), n_bins, labels=False) + 1
    out = work.groupby("bin").agg(
        mean_price=(price_col, "mean"), conversion=(outcome_col, "mean"), n=(outcome_col, "size")
    )
    return out.reset_index()


def price_response_corr(df: pd.DataFrame, price_col: str, outcome_col: str, n_bins: int = 10) -> float:
    """Correlation between binned price and conversion.

    Negative => consistent with a demand curve. Positive => confounded.
    """
    t = price_response_table(df, price_col, outcome_col, n_bins)
    if len(t) < 3:
        return float("nan")
    return float(np.corrcoef(t["mean_price"], t["conversion"])[0, 1])


def segmented_price_response(
    df: pd.DataFrame, price_col: str, outcome_col: str, by: list[str],
    n_bins: int = 5, min_rows: int = 5000,
) -> pd.DataFrame:
    """Repeat the diagnostic within segments to test whether the sign is stable."""
    rows = []
    for key, g in df.groupby(by):
        if len(g) < min_rows:
            continue
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {**dict(zip(by, key)), "n": len(g),
             "corr_price_conversion": round(price_response_corr(g, price_col, outcome_col, n_bins), 3)}
        )
    return pd.DataFrame(rows)


def verdict(overall_corr: float, segment_corrs: pd.Series) -> str:
    """Plain-language conclusion on whether elasticity is identifiable."""
    if overall_corr >= 0:
        base = f"NOT IDENTIFIABLE: conversion rises with price (corr={overall_corr:+.3f})."
    elif (segment_corrs > 0).any():
        base = f"UNSTABLE: overall corr={overall_corr:+.3f} but sign flips across segments."
    else:
        base = f"PLAUSIBLE: corr={overall_corr:+.3f} and negative in all segments."
    return base + " Price is set from risk, so this is an association, not a causal elasticity."
