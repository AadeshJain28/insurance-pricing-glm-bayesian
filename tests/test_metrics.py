from __future__ import annotations

import numpy as np
import pytest

from insurance_pricing.evaluation.metrics import (
    gamma_deviance, gini_norm, lift_table, poisson_deviance, tweedie_deviance,
)


def test_poisson_deviance_zero_when_perfect():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)


def test_poisson_deviance_positive_when_wrong():
    assert poisson_deviance([1.0, 2.0], [2.0, 1.0]) > 0


def test_gamma_deviance_zero_when_perfect():
    y = np.array([100.0, 500.0, 2500.0])
    assert gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-9)


def test_gamma_deviance_rejects_zero_target():
    with pytest.raises(ValueError):
        gamma_deviance([0.0, 1.0], [1.0, 1.0])


def test_tweedie_power_validated():
    with pytest.raises(ValueError):
        tweedie_deviance([1.0], [1.0], power=2.5)


def test_tweedie_deviance_better_for_closer_prediction():
    y = [0.0, 0.0, 500.0, 0.0]
    near = tweedie_deviance(y, [10.0, 10.0, 400.0, 10.0], power=1.5)
    far = tweedie_deviance(y, [200.0, 200.0, 50.0, 200.0], power=1.5)
    assert near < far


def test_gini_perfect_ranking_is_one():
    y = np.array([0.0, 1.0, 2.0, 5.0])
    assert gini_norm(y, y) == pytest.approx(1.0, abs=1e-9)


def test_gini_reversed_ranking_is_negative():
    y = np.array([0.0, 1.0, 2.0, 5.0])
    assert gini_norm(y, -y) < 0


def test_lift_table_shape_and_ordering():
    rng = np.random.default_rng(0)
    n = 500
    pred = rng.uniform(0.01, 0.5, n)
    y = rng.poisson(pred)
    tbl = lift_table(y, pred, n_bins=5)
    assert len(tbl) == 5
    assert tbl["predicted"].is_monotonic_increasing
    assert {"decile", "exposure", "predicted", "actual", "a_over_e"} <= set(tbl.columns)


def test_weights_respected():
    y = [1.0, 1.0]
    p = [1.0, 2.0]
    heavy_on_good = poisson_deviance(y, p, weights=[10.0, 1.0])
    heavy_on_bad = poisson_deviance(y, p, weights=[1.0, 10.0])
    assert heavy_on_good < heavy_on_bad
