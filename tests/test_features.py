from __future__ import annotations

import pandas as pd

from insurance_pricing.features.build_features import (
    prepare_risk_features, severity_subset, train_test_split_frame,
)


def _risk_df():
    return pd.DataFrame({
        "VehPower": [4, 20], "VehAge": [2, 45], "DrivAge": [17, 99],
        "BonusMalus": [50, 300], "Density": [10.0, 30000.0],
        "VehBrand": ["B1", "B2"], "VehGas": ["Regular", "Diesel"],
        "Area": ["A", "B"], "Region": ["R11", "R24"],
        "ClaimNb": [0, 2], "ClaimAmount": [0.0, 5000.0], "Exposure": [0.5, 1.0],
    })


def test_caps_applied():
    out = prepare_risk_features(_risk_df())
    assert out["VehAge"].max() <= 20
    assert out["DrivAge"].min() >= 18 and out["DrivAge"].max() <= 90
    assert out["BonusMalus"].max() <= 150
    assert out["VehPower"].max() <= 15


def test_log_density_monotonic():
    out = prepare_risk_features(_risk_df())
    assert out["LogDensity"].iloc[1] > out["LogDensity"].iloc[0]


def test_categoricals_are_category_dtype():
    out = prepare_risk_features(_risk_df())
    assert str(out["VehBrand"].dtype) == "category"


def test_severity_subset_excludes_zero_claims():
    sub = severity_subset(_risk_df())
    assert len(sub) == 1
    assert (sub["ClaimNb"] > 0).all()
    assert sub["AvgClaimAmount"].iloc[0] == 2500.0   # 5000 / 2 claims


def test_split_sizes_and_disjoint():
    df = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})
    tr, te = train_test_split_frame(df, test_size=0.2, seed=1)
    assert len(tr) == 80 and len(te) == 20
    assert set(tr["x"]).isdisjoint(set(te["x"]))
