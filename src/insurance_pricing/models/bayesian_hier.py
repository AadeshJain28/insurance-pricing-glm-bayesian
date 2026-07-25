"""Phase 3b — Bayesian hierarchical credibility (PyMC).

Phase 3a found the Gamma severity GLM performing *worse than a constant*: ~60
free parameters fitted to 19,843 very noisy claims. That is precisely the problem
credibility theory was built for.

A hierarchical prior pools information across cells: a cell with few claims is
shrunk hard toward the portfolio mean, while a cell with many claims is trusted.
The shrinkage weight that falls out of the posterior is the Bayesian counterpart
of the Buhlmann-Straub credibility factor ``Z = n / (n + k)``.

Models (both fitted on Region x VehBrand cells, not raw policies):

* **Frequency** ``N_j ~ Poisson(E_j * lambda_j)``, ``log lambda_j ~ N(mu, tau)``
* **Severity**  ``log(ybar_j) ~ N(theta_j, sigma / sqrt(n_j))``, ``theta_j ~ N(mu, tau)``

    python -m insurance_pricing.models.bayesian_hier
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insurance_pricing.logger import get_logger

logger = get_logger("insurance_pricing.bayesian")


# --------------------------------------------------------------------------- #
# Cell construction (pure pandas — unit tested)
# --------------------------------------------------------------------------- #
def build_frequency_cells(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate policies into cells for the hierarchical frequency model."""
    g = (
        df.groupby(group_cols, observed=True)
        .agg(claims=("ClaimNb", "sum"), exposure=("Exposure", "sum"), policies=("ClaimNb", "size"))
        .reset_index()
    )
    g = g[g["exposure"] > 0].reset_index(drop=True)
    g["raw_frequency"] = g["claims"] / g["exposure"]
    return g


def build_severity_cells(df_sev: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate claims into cells for the hierarchical severity model."""
    g = (
        df_sev.groupby(group_cols, observed=True)
        .agg(total_amount=("ClaimAmount", "sum"), n_claims=("ClaimNb", "sum"), rows=("ClaimNb", "size"))
        .reset_index()
    )
    g = g[(g["n_claims"] > 0) & (g["total_amount"] > 0)].reset_index(drop=True)
    g["raw_severity"] = g["total_amount"] / g["n_claims"]
    return g


def credibility_weight(n: np.ndarray, k: float) -> np.ndarray:
    """Buhlmann-Straub credibility factor Z = n / (n + k)."""
    n = np.asarray(n, dtype=float)
    if k < 0:
        raise ValueError("k must be non-negative.")
    return n / (n + k)


def shrinkage_summary(raw: np.ndarray, shrunk: np.ndarray, n: np.ndarray, grand_mean: float) -> pd.DataFrame:
    """How far each cell moved from its raw estimate toward the portfolio mean."""
    raw, shrunk, n = np.asarray(raw, float), np.asarray(shrunk, float), np.asarray(n, float)
    gap = raw - grand_mean
    moved = raw - shrunk
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(gap) > 1e-12, moved / gap, 0.0)
    return pd.DataFrame(
        {"n": n, "raw": raw, "shrunk": shrunk,
         "moved_toward_mean_frac": np.clip(frac, -1, 2).round(3)}
    )


# --------------------------------------------------------------------------- #
# PyMC models
# --------------------------------------------------------------------------- #
def fit_hierarchical_frequency(cells: pd.DataFrame, draws=1000, tune=1000, chains=2,
                               target_accept=0.9, seed=42):
    """Hierarchical Poisson: log-rate per cell drawn from a shared population."""
    import pymc as pm

    claims = cells["claims"].to_numpy(dtype=float)
    exposure = cells["exposure"].to_numpy(dtype=float)
    n_cells = len(cells)
    log_grand = float(np.log(claims.sum() / exposure.sum()))

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=log_grand, sigma=1.0)
        tau = pm.HalfNormal("tau", sigma=0.5)               # between-cell spread
        # NON-CENTRED parameterisation: sampling theta ~ N(mu, tau) directly creates
        # Neal's funnel (theta collapses as tau -> 0), which caused 1000 divergences
        # and R-hat ~1.5 with few cells. Decoupling location from scale fixes it.
        theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=1.0, shape=n_cells)
        theta = pm.Deterministic("theta", mu + tau * theta_raw)
        pm.Poisson("obs", mu=exposure * pm.math.exp(theta), observed=claims)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                          random_seed=seed, progressbar=False)
    return model, idata


def fit_hierarchical_severity(cells: pd.DataFrame, draws=1000, tune=1000, chains=2,
                              target_accept=0.9, seed=42):
    """Hierarchical log-normal on cell mean severity (Buhlmann-Straub structure).

    The observation sd shrinks as ``1/sqrt(n_claims)``, so cells backed by more
    claims pull the posterior harder — exactly the credibility weighting.
    """
    import pymc as pm

    y = np.log(cells["raw_severity"].to_numpy(dtype=float))
    n = cells["n_claims"].to_numpy(dtype=float)
    n_cells = len(cells)
    log_grand = float(np.log(cells["total_amount"].sum() / cells["n_claims"].sum()))

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=log_grand, sigma=1.0)
        tau = pm.HalfNormal("tau", sigma=0.5)
        # non-centred (see fit_hierarchical_frequency) — essential here, where the
        # coarse 22-cell model diverged badly under the centred form
        theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=1.0, shape=n_cells)
        theta = pm.Deterministic("theta", mu + tau * theta_raw)
        # weakly-informative rather than vague: with few cells, HalfNormal(2) let
        # sigma wander to 4+ and trade off against tau
        sigma = pm.HalfNormal("sigma", sigma=1.0)           # within-cell claim variability
        pm.Normal("obs", mu=theta, sigma=sigma / np.sqrt(n), observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                          random_seed=seed, progressbar=False)
    return model, idata


def posterior_cell_means(idata, transform: str = "exp") -> np.ndarray:
    """Posterior mean per cell on the natural scale."""
    theta = idata.posterior["theta"].values.reshape(-1, idata.posterior["theta"].shape[-1])
    return np.exp(theta).mean(axis=0) if transform == "exp" else theta.mean(axis=0)


def divergences(idata) -> int:
    """Number of divergent transitions — a geometry problem, not just noise."""
    try:
        return int(idata.sample_stats["diverging"].values.sum())
    except (KeyError, AttributeError):
        return -1


def convergence_report(idata) -> pd.DataFrame:
    """R-hat, effective sample size and divergences for the population parameters."""
    import arviz as az

    keep = [v for v in ("mu", "tau", "sigma") if v in idata.posterior]
    summ = az.summary(idata, var_names=keep, round_to=4)
    out = summ.reset_index().rename(columns={"index": "parameter"})
    out["divergences"] = divergences(idata)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _predict_from_cells(test: pd.DataFrame, cells: pd.DataFrame, values: np.ndarray,
                        group_cols: list[str], fallback: float) -> np.ndarray:
    lookup = cells[group_cols].copy()
    lookup["_pred"] = values
    merged = test[group_cols].merge(lookup, on=group_cols, how="left")
    return merged["_pred"].fillna(fallback).to_numpy()


def _fit_and_score(cfg, group_cols, tr, te, str_, ste, label: str):
    """Fit both hierarchical models at one cell granularity and score on test."""
    from insurance_pricing.evaluation.metrics import gamma_deviance, gini_norm, poisson_deviance

    b = cfg.bayesian
    fcells = build_frequency_cells(tr, group_cols)
    scells = build_severity_cells(str_, group_cols)
    logger.info("[%s] cells: frequency=%d severity=%d | median claims/cell=%d",
                label, len(fcells), len(scells), int(scells["n_claims"].median()))

    _, f_id = fit_hierarchical_frequency(fcells, b.draws, b.tune, b.chains, b.target_accept, cfg.project.seed)
    _, s_id = fit_hierarchical_severity(scells, b.draws, b.tune, b.chains, b.target_accept, cfg.project.seed)
    f_post, s_post = posterior_cell_means(f_id), posterior_cell_means(s_id)

    grand_freq = float(tr["ClaimNb"].sum() / tr["Exposure"].sum())
    grand_sev = float(str_["ClaimAmount"].sum() / str_["ClaimNb"].sum())

    yf = (te["ClaimNb"] / te["Exposure"]).to_numpy(); wf = te["Exposure"].to_numpy()
    f_pred = _predict_from_cells(te, fcells, f_post, group_cols, grand_freq)
    f_raw = _predict_from_cells(te, fcells, fcells["raw_frequency"].to_numpy(), group_cols, grand_freq)
    ys = ste["AvgClaimAmount"].to_numpy(); ws = ste["ClaimNb"].to_numpy()
    s_pred = _predict_from_cells(ste, scells, s_post, group_cols, grand_sev)
    s_raw = _predict_from_cells(ste, scells, scells["raw_severity"].to_numpy(), group_cols, grand_sev)

    freq = pd.DataFrame([
        {"cells": label, "model": "Baseline", "poisson_deviance": round(poisson_deviance(yf, np.full(len(te), grand_freq), wf), 6), "gini": round(gini_norm(yf, np.full(len(te), grand_freq), wf), 4)},
        {"cells": label, "model": "Raw cell means", "poisson_deviance": round(poisson_deviance(yf, f_raw, wf), 6), "gini": round(gini_norm(yf, f_raw, wf), 4)},
        {"cells": label, "model": "Hierarchical Bayes", "poisson_deviance": round(poisson_deviance(yf, f_pred, wf), 6), "gini": round(gini_norm(yf, f_pred, wf), 4)},
    ])
    sev = pd.DataFrame([
        {"cells": label, "model": "Baseline", "gamma_deviance": round(gamma_deviance(ys, np.full(len(ste), grand_sev), ws), 4), "gini": round(gini_norm(ys, np.full(len(ste), grand_sev), ws), 4)},
        {"cells": label, "model": "Raw cell means", "gamma_deviance": round(gamma_deviance(ys, s_raw, ws), 4), "gini": round(gini_norm(ys, s_raw, ws), 4)},
        {"cells": label, "model": "Hierarchical Bayes", "gamma_deviance": round(gamma_deviance(ys, s_pred, ws), 4), "gini": round(gini_norm(ys, s_pred, ws), 4)},
    ])
    shrink = shrinkage_summary(scells["raw_severity"].to_numpy(), s_post,
                               scells["n_claims"].to_numpy(), grand_sev)
    conv = pd.concat([convergence_report(f_id).assign(model="frequency"),
                      convergence_report(s_id).assign(model="severity")], ignore_index=True)
    conv.insert(0, "cells", label)
    return freq, sev, shrink, conv


def run(config_path: str = "config/config.yaml") -> None:
    from insurance_pricing.config import load_config
    from insurance_pricing.evaluation.metrics import gamma_deviance, gini_norm, poisson_deviance

    cfg = load_config(config_path)
    group_cols = cfg.bayesian.group_cols
    proc = Path("data/processed")
    tr = pd.read_parquet(proc / "risk_train.parquet")
    te = pd.read_parquet(proc / "risk_test.parquet")
    str_ = pd.read_parquet(proc / "severity_train.parquet")
    ste = pd.read_parquet(proc / "severity_test.parquet")

    # ---- granularity comparison: fine vs coarse cells ----
    fine_label = " x ".join(group_cols)
    coarse_cols = getattr(cfg.bayesian, "coarse_group_cols", ["Region"])
    coarse_label = " x ".join(coarse_cols)

    f_fine, s_fine, shrink_fine, conv_fine = _fit_and_score(cfg, group_cols, tr, te, str_, ste, fine_label)
    f_coarse, s_coarse, shrink_coarse, conv_coarse = _fit_and_score(cfg, coarse_cols, tr, te, str_, ste, coarse_label)

    freq_all = pd.concat([f_fine, f_coarse], ignore_index=True)
    sev_all = pd.concat([s_fine, s_coarse], ignore_index=True)
    conv_all = pd.concat([conv_fine, conv_coarse], ignore_index=True)
    max_rhat = float(conv_all["r_hat"].max())

    print("\n=== FREQUENCY (test) — fine vs coarse cells ===");  print(freq_all.to_string(index=False))
    print("\n=== SEVERITY (test) — fine vs coarse cells ===");   print(sev_all.to_string(index=False))
    print("\n=== CONVERGENCE (all population parameters) ===");  print(conv_all.to_string(index=False))
    print(f"\nmax R-hat across all parameters = {max_rhat:.4f} "
          f"({'OK (<=1.01)' if max_rhat <= 1.01 else 'STILL HIGH — chains not mixed'})")
    for lbl, sh in ((fine_label, shrink_fine), (coarse_label, shrink_coarse)):
        small = sh.loc[sh.n < 50, "moved_toward_mean_frac"].mean()
        large = sh.loc[sh.n > 500, "moved_toward_mean_frac"].mean()
        print(f"shrinkage [{lbl}]: small cells (n<50) = {small:.3f} | large cells (n>500) = {large:.3f}")

    reports = Path(cfg.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase3b_bayesian_credibility.md").write_text(
        "# Phase 3b — Bayesian hierarchical credibility\n\n"
        f"MCMC: {cfg.bayesian.chains} chains, {cfg.bayesian.draws} draws, {cfg.bayesian.tune} tune, "
        f"target_accept={cfg.bayesian.target_accept}. **Max R-hat = {max_rhat:.4f}**.\n\n"
        "## Frequency (test)\n\n" + freq_all.to_markdown(index=False)
        + "\n\n## Severity (test)\n\n" + sev_all.to_markdown(index=False)
        + "\n\n## Convergence\n\n" + conv_all.to_markdown(index=False)
        + "\n\n## Shrinkage — 5 smallest severity cells (fine)\n\n"
        + shrink_fine.nsmallest(5, "n").round(2).to_markdown(index=False)
        + "\n\n## Shrinkage — 5 largest severity cells (fine)\n\n"
        + shrink_fine.nlargest(5, "n").round(2).to_markdown(index=False)
        + "\n\n*`moved_toward_mean_frac` = 1.0 means fully shrunk to the portfolio mean, 0.0 means "
          "the raw cell estimate was trusted entirely — the Bayesian counterpart of the "
          "Buhlmann-Straub credibility factor Z = n/(n+k).*\n",
        encoding="utf-8",
    )
    logger.info("Saved reports/phase3b_bayesian_credibility.md (max R-hat=%.4f)", max_rhat)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
