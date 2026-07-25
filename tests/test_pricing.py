from __future__ import annotations

import numpy as np
import pytest

from insurance_pricing.pricing.optimize import (
    logistic_demand, optimal_price, price_elasticity,
)


def test_demand_requires_negative_slope():
    with pytest.raises(ValueError):
        logistic_demand(5.0, 0.1)


def test_demand_decreases_with_price():
    d = logistic_demand(5.0, -0.01)
    q = d(np.array([100.0, 200.0, 400.0]))
    assert q[0] > q[1] > q[2]


def test_optimal_price_between_cost_and_max():
    d = logistic_demand(5.0, -0.01)
    res = optimal_price(technical_cost=150.0, demand_fn=d, price_lo=150.0, price_hi=800.0)
    assert 150.0 <= res.best_price <= 800.0
    assert res.margin == pytest.approx(res.best_price - 150.0)
    assert res.best_profit == pytest.approx(res.margin * res.best_conversion, rel=1e-6)


def test_optimal_price_rises_with_cost():
    d = logistic_demand(5.0, -0.01)
    cheap = optimal_price(100.0, d, 100.0, 900.0)
    dear = optimal_price(300.0, d, 300.0, 900.0)
    assert dear.best_price > cheap.best_price


def test_more_elastic_demand_lowers_optimal_price():
    inelastic = optimal_price(150.0, logistic_demand(5.0, -0.005), 150.0, 900.0)
    elastic = optimal_price(150.0, logistic_demand(5.0, -0.02), 150.0, 900.0)
    assert elastic.best_price < inelastic.best_price


def test_elasticity_is_negative():
    d = logistic_demand(5.0, -0.01)
    assert price_elasticity(d, 300.0) < 0


def test_price_range_validated():
    with pytest.raises(ValueError):
        optimal_price(100.0, logistic_demand(5.0, -0.01), 500.0, 100.0)
