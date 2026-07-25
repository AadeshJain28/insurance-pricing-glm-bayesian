"""Actuarial evaluation metrics.

Deviance (Poisson/Gamma/Tweedie) is the correct loss for these GLM families.
Gini and lift measure *risk ranking*, which is what actually drives pricing
segmentation. R-squared and accuracy are deliberately absent: they are
meaningless for counts and for heavily right-skewed severities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _prep(y_true: ArrayLike, y_pred: ArrayLike, weights: ArrayLike | None):
    """Align inputs. Predictions are NOT clipped here — ranking metrics such as
    Gini accept negative scores. Deviances clip separately via :func:`_positive`.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(y_pred, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if not (y.shape == mu.shape == w.shape):
        raise ValueError(f"Shape mismatch: y{y.shape}, pred{mu.shape}, w{w.shape}")
    return y, mu, w


def _positive(mu: np.ndarray) -> np.ndarray:
    """Deviance requires a strictly positive mean."""
    return np.clip(mu, 1e-10, None)


def poisson_deviance(y_true, y_pred, weights=None) -> float:
    """Mean Poisson deviance (frequency models). Lower is better."""
    y, mu, w = _prep(y_true, y_pred, weights)
    mu = _positive(mu)
    # guard the log so zero-claim rows (the majority of a book) don't emit warnings
    ratio = np.where(y > 0, y / mu, 1.0)
    term = np.where(y > 0, y * np.log(ratio), 0.0)
    return float(2 * np.sum(w * (term - (y - mu))) / np.sum(w))


def gamma_deviance(y_true, y_pred, weights=None) -> float:
    """Mean Gamma deviance (severity models). Requires y > 0."""
    y, mu, w = _prep(y_true, y_pred, weights)
    if np.any(y <= 0):
        raise ValueError("Gamma deviance requires strictly positive targets.")
    mu = _positive(mu)
    return float(2 * np.sum(w * (-np.log(y / mu) + (y - mu) / mu)) / np.sum(w))


def tweedie_deviance(y_true, y_pred, power: float = 1.5, weights=None) -> float:
    """Mean Tweedie deviance for 1 < power < 2 (pure premium)."""
    if not 1 < power < 2:
        raise ValueError("power must lie strictly between 1 and 2.")
    y, mu, w = _prep(y_true, y_pred, weights)
    mu = _positive(mu)
    y = np.clip(y, 0, None)
    p = power
    dev = 2 * (
        np.power(y, 2 - p) / ((1 - p) * (2 - p))
        - y * np.power(mu, 1 - p) / (1 - p)
        + np.power(mu, 2 - p) / (2 - p)
    )
    return float(np.sum(w * dev) / np.sum(w))


def gini_norm(y_true, y_pred, weights=None) -> float:
    """Normalised (exposure-weighted) Gini: how well predictions *rank* risk.

    1.0 = perfect ranking, 0.0 = random. The standard commercial metric because
    pricing only needs correct ordering to segment.
    """
    y, mu, w = _prep(y_true, y_pred, weights)

    def _lorenz_gini(order: np.ndarray) -> float:
        yy, ww = y[order], w[order]
        cum_w = np.cumsum(ww) / np.sum(ww)
        cum_y = np.cumsum(yy * ww) / np.sum(yy * ww)
        return float(np.sum(cum_y[:-1] * cum_w[1:] - cum_y[1:] * cum_w[:-1]))

    model = _lorenz_gini(np.argsort(mu))
    perfect = _lorenz_gini(np.argsort(y))
    return model / perfect if perfect != 0 else 0.0


def lift_table(y_true, y_pred, weights=None, n_bins: int = 10):
    """Actual vs expected by predicted-risk decile (calibration + separation)."""
    import pandas as pd

    y, mu, w = _prep(y_true, y_pred, weights)
    order = np.argsort(mu)
    y, mu, w = y[order], mu[order], w[order]
    edges = np.floor(np.linspace(0, len(y), n_bins + 1)).astype(int)
    rows = []
    for b in range(n_bins):
        s, e = edges[b], edges[b + 1]
        if e <= s:
            continue
        ws = np.sum(w[s:e])
        rows.append(
            {
                "decile": b + 1,
                "exposure": round(float(ws), 2),
                "predicted": round(float(np.sum(mu[s:e] * w[s:e]) / ws), 5),
                "actual": round(float(np.sum(y[s:e] * w[s:e]) / ws), 5),
                "n": int(e - s),
            }
        )
    df = pd.DataFrame(rows)
    df["a_over_e"] = (df["actual"] / df["predicted"].replace(0, np.nan)).round(3)
    return df
