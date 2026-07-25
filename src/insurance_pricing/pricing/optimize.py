"""Profit-optimal pricing.

Given a technical (risk) cost and a demand model P(convert | price), choose the
price maximising expected profit:

    profit(P) = (P - technical_cost) * P(convert | P)

This is where the actuarial model meets the commercial decision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PriceResult:
    """Outcome of a profit-maximising price search."""

    best_price: float
    best_profit: float
    best_conversion: float
    margin: float
    prices: NDArray[np.float64]
    profits: NDArray[np.float64]


def logistic_demand(beta0: float, beta_price: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return P(convert | price) = sigmoid(b0 + b_price * price). b_price < 0."""
    if beta_price >= 0:
        raise ValueError("beta_price must be negative (demand falls as price rises).")

    def demand(price: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(beta0 + beta_price * np.asarray(price, dtype=float))))

    return demand


def optimal_price(
    technical_cost: float,
    demand_fn: Callable[[np.ndarray], np.ndarray],
    price_lo: float,
    price_hi: float,
    n_grid: int = 200,
) -> PriceResult:
    """Grid-search the price maximising expected profit."""
    if price_hi <= price_lo:
        raise ValueError("price_hi must exceed price_lo.")
    prices = np.linspace(price_lo, price_hi, n_grid)
    conv = np.asarray(demand_fn(prices), dtype=float)
    profits = (prices - technical_cost) * conv
    i = int(np.argmax(profits))
    return PriceResult(
        best_price=float(prices[i]),
        best_profit=float(profits[i]),
        best_conversion=float(conv[i]),
        margin=float(prices[i] - technical_cost),
        prices=prices,
        profits=profits,
    )


def price_elasticity(demand_fn: Callable[[np.ndarray], np.ndarray], price: float, eps: float = 1e-4) -> float:
    """Point elasticity of demand: (dQ/dP)(P/Q). Negative for normal goods."""
    p = float(price)
    q = float(demand_fn(np.array([p]))[0])
    q_up = float(demand_fn(np.array([p + eps]))[0])
    if q == 0:
        return float("nan")
    return ((q_up - q) / eps) * (p / q)
