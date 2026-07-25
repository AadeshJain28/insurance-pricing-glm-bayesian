"""Phase 4 — from technical cost to profit-optimal commercial price.

The risk models give the **technical price** (expected claim cost). Converting
that into a **commercial price** needs a demand curve — and Phase 2 established
that this data *cannot* identify one (conversion rises with price, corr +0.51,
because premium is set from risk).

So the demand curve here is **stated, not estimated**: we calibrate a logistic
demand from an assumed reference conversion and elasticity, then sweep the
elasticity to show how the optimal price responds. Every pricing number is
therefore reported as conditional on that assumption.

    python -m insurance_pricing.pricing.run_pricing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insurance_pricing.logger import get_logger
from insurance_pricing.pricing.optimize import logistic_demand, optimal_price, price_elasticity

logger = get_logger("insurance_pricing.pricing")


def calibrate_logistic_demand(p_ref: float, q_ref: float, elasticity: float):
    """Build a logistic demand curve matching a reference point and elasticity.

    For ``q(P) = sigmoid(b0 + b1 P)`` the point elasticity is
    ``E = b1 * P * (1 - q)``, so::

        b1 = E / (P_ref * (1 - q_ref))
        b0 = logit(q_ref) - b1 * P_ref

    Args:
        p_ref: reference price (e.g. 1.5x technical cost).
        q_ref: assumed conversion at that price, in (0, 1).
        elasticity: assumed price elasticity at that point (negative).
    """
    if not 0 < q_ref < 1:
        raise ValueError("q_ref must lie strictly in (0, 1).")
    if elasticity >= 0:
        raise ValueError("elasticity must be negative.")
    if p_ref <= 0:
        raise ValueError("p_ref must be positive.")

    b1 = elasticity / (p_ref * (1 - q_ref))
    b0 = float(np.log(q_ref / (1 - q_ref)) - b1 * p_ref)
    return logistic_demand(b0, b1)


def sensitivity_table(technical_cost: float, elasticities: list[float], q_ref: float,
                      markup_ref: float, lo_mult: float, hi_mult: float, n_grid: int) -> pd.DataFrame:
    """Optimal price across a range of assumed elasticities."""
    p_ref = technical_cost * markup_ref
    rows = []
    for e in elasticities:
        demand = calibrate_logistic_demand(p_ref, q_ref, e)
        res = optimal_price(technical_cost, demand, technical_cost * lo_mult,
                            technical_cost * hi_mult, n_grid)
        rows.append({
            "assumed_elasticity": e,
            "optimal_price": round(res.best_price, 2),
            "markup_over_cost": round(res.best_price / technical_cost, 2),
            "conversion_at_optimum": round(res.best_conversion, 4),
            "expected_profit": round(res.best_profit, 2),
            "realised_elasticity": round(price_elasticity(demand, res.best_price), 3),
        })
    return pd.DataFrame(rows)


def run(config_path: str = "config/config.yaml") -> None:
    import joblib

    from insurance_pricing.config import load_config

    cfg = load_config(config_path)
    proc, models_dir = Path("data/processed"), Path(cfg.paths.models_dir)
    te = pd.read_parquet(proc / "risk_test.parquet")

    bundle = joblib.load(models_dir / "risk_glms.joblib")
    tweedie = bundle["tweedie"]                      # best-calibrated pure-premium model

    # ---- technical price per policy (expected annual claim cost) ----
    technical = np.clip(tweedie.predict(te), 1.0, None)
    te = te.assign(technical_price=technical)
    portfolio_cost = float(np.average(technical, weights=te["Exposure"]))
    logger.info("Technical price: mean=%.2f median=%.2f p90=%.2f (exposure-weighted mean=%.2f)",
                technical.mean(), np.median(technical), np.percentile(technical, 90), portfolio_cost)

    # ---- profit-optimal price for the average policy, across elasticity assumptions ----
    elasticities = [-0.5, -1.0, -1.5, -2.0, -3.0, -5.0]
    sens = sensitivity_table(portfolio_cost, elasticities, q_ref=0.20, markup_ref=1.5,
                             lo_mult=cfg.pricing.price_grid_lo, hi_mult=cfg.pricing.price_grid_hi,
                             n_grid=cfg.pricing.price_grid_n)

    # ---- segment view: cheapest vs dearest risk deciles ----
    te["risk_decile"] = pd.qcut(te["technical_price"].rank(method="first"), 10, labels=False) + 1
    seg_rows = []
    for d in (1, 5, 10):
        cost_d = float(np.average(te.loc[te.risk_decile == d, "technical_price"],
                                  weights=te.loc[te.risk_decile == d, "Exposure"]))
        demand = calibrate_logistic_demand(cost_d * 1.5, 0.20, -1.5)
        res = optimal_price(cost_d, demand, cost_d * cfg.pricing.price_grid_lo,
                            cost_d * cfg.pricing.price_grid_hi, cfg.pricing.price_grid_n)
        seg_rows.append({"risk_decile": d, "technical_cost": round(cost_d, 2),
                         "optimal_price": round(res.best_price, 2),
                         "markup": round(res.best_price / cost_d, 2),
                         "conversion": round(res.best_conversion, 4),
                         "expected_profit": round(res.best_profit, 2)})
    seg = pd.DataFrame(seg_rows)

    print(f"\n=== TECHNICAL PRICE (Tweedie GLM) ===\nportfolio mean = {portfolio_cost:.2f} "
          f"per exposure-year\n")
    print("=== OPTIMAL PRICE vs ASSUMED ELASTICITY (average policy) ===")
    print(sens.to_string(index=False))
    print("\n=== BY RISK DECILE (elasticity fixed at -1.5) ===")
    print(seg.to_string(index=False))

    reports = Path(cfg.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase4_pricing.md").write_text(
        "# Phase 4 — profit-optimal pricing\n\n"
        f"Technical price from the Tweedie GLM: portfolio mean **{portfolio_cost:.2f}** per exposure-year.\n\n"
        "> **The demand curve is assumed, not estimated.** Phase 2 showed this data cannot identify a "
        "price elasticity (conversion rises with price, corr +0.51, because premium is set from risk). "
        "Every figure below is therefore conditional on the stated elasticity, which is why the "
        "sensitivity sweep — not any single price — is the deliverable.\n\n"
        "## Optimal price vs assumed elasticity\n\n" + sens.to_markdown(index=False)
        + "\n\n## By risk decile (elasticity = -1.5)\n\n" + seg.to_markdown(index=False)
        + "\n\n*More elastic demand (more negative) => lower optimal markup. "
          "Higher-risk policies carry a higher absolute price but a similar markup, because the "
          "demand curve is calibrated relative to each segment's own technical cost.*\n",
        encoding="utf-8",
    )
    te[["technical_price", "risk_decile"]].to_parquet(proc / "technical_prices.parquet", index=False)
    logger.info("Saved reports/phase4_pricing.md")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
