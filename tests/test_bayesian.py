from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_pricing.models.bayesian_hier import (
    build_frequency_cells, build_severity_cells, credibility_weight, shrinkage_summary,
)


def test_frequency_cells_aggregate():
    df = pd.DataFrame({"Region": ["A", "A", "B"], "VehBrand": ["X", "X", "Y"],
                       "ClaimNb": [1, 2, 0], "Exposure": [1.0, 1.0, 0.5]})
    cells = build_frequency_cells(df, ["Region", "VehBrand"])
    assert len(cells) == 2
    a = cells[cells.Region == "A"].iloc[0]
    assert a["claims"] == 3 and a["exposure"] == 2.0
    assert a["raw_frequency"] == pytest.approx(1.5)


def test_severity_cells_exclude_empty():
    df = pd.DataFrame({"Region": ["A", "B"], "VehBrand": ["X", "Y"],
                       "ClaimAmount": [4000.0, 0.0], "ClaimNb": [2, 0]})
    cells = build_severity_cells(df, ["Region", "VehBrand"])
    assert len(cells) == 1
    assert cells.iloc[0]["raw_severity"] == pytest.approx(2000.0)


def test_credibility_weight_properties():
    z = credibility_weight(np.array([0, 10, 1000]), k=100)
    assert z[0] == 0.0                 # no data -> no credibility
    assert 0 < z[1] < z[2] < 1.0       # monotone increasing in n, never reaches 1
    assert credibility_weight(np.array([100]), 100)[0] == pytest.approx(0.5)


def test_credibility_weight_validates():
    with pytest.raises(ValueError):
        credibility_weight(np.array([1.0]), k=-1)


def test_shrinkage_summary_direction():
    # raw=200 vs grand mean=100; shrunk=150 -> moved half way
    out = shrinkage_summary(np.array([200.0]), np.array([150.0]), np.array([10.0]), 100.0)
    assert out["moved_toward_mean_frac"].iloc[0] == pytest.approx(0.5)


def test_no_shrinkage_when_estimate_equals_raw():
    out = shrinkage_summary(np.array([200.0]), np.array([200.0]), np.array([999.0]), 100.0)
    assert out["moved_toward_mean_frac"].iloc[0] == pytest.approx(0.0)
