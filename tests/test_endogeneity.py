from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_pricing.evaluation.endogeneity import (
    price_response_corr, price_response_table, segmented_price_response, verdict,
)


def _true_demand(n=4000, seed=0):
    """Synthetic data where conversion genuinely falls with price."""
    rng = np.random.default_rng(seed)
    price = rng.uniform(100, 1000, n)
    p = 1 / (1 + np.exp(-(3 - 0.005 * price)))
    return pd.DataFrame({"price": price, "conv": rng.binomial(1, p), "seg": rng.integers(0, 2, n)})


def _confounded(n=4000, seed=0):
    """Risk drives BOTH price and conversion -> positive association."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    price = 500 + 200 * risk + rng.normal(0, 20, n)
    p = 1 / (1 + np.exp(-(-1.5 + 1.2 * risk)))
    return pd.DataFrame({"price": price, "conv": rng.binomial(1, p), "seg": rng.integers(0, 2, n)})


def test_table_shape():
    t = price_response_table(_true_demand(), "price", "conv", n_bins=5)
    assert len(t) == 5
    assert t["mean_price"].is_monotonic_increasing


def test_detects_real_demand_curve():
    assert price_response_corr(_true_demand(), "price", "conv") < -0.5


def test_flags_confounded_data():
    assert price_response_corr(_confounded(), "price", "conv") > 0


def test_segmented_returns_rows():
    seg = segmented_price_response(_confounded(), "price", "conv", ["seg"], n_bins=4, min_rows=100)
    assert len(seg) == 2 and "corr_price_conversion" in seg.columns


def test_verdict_language():
    assert "NOT IDENTIFIABLE" in verdict(0.5, pd.Series([0.4, 0.6]))
    assert "UNSTABLE" in verdict(-0.6, pd.Series([-0.9, 0.8]))
    assert "PLAUSIBLE" in verdict(-0.8, pd.Series([-0.9, -0.7]))
