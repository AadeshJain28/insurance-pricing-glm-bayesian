"""Phase 3a — the actuarial risk models.

Three routes to a technical price, plus a machine-learning challenger:

1. **Frequency** — Poisson GLM. Exposure enters as a weight on the *rate* target
   (`ClaimNb / Exposure`), which is mathematically identical to a `log(Exposure)`
   offset on the count target but works with scikit-learn's API.
2. **Severity** — Gamma GLM on average claim cost, weighted by claim count, and
   fitted **only on policies that actually had a claim** (Gamma is undefined at 0).
3. **Pure premium** — the product `frequency x severity`, and separately a direct
   **Tweedie** GLM (compound Poisson-Gamma) for comparison.
4. **GBM challenger** — LightGBM with Poisson/Tweedie objectives, to quantify what
   a GLM gives up for its interpretability.

    python -m insurance_pricing.models.risk_models
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insurance_pricing.evaluation.metrics import (
    gamma_deviance, gini_norm, lift_table, poisson_deviance, tweedie_deviance,
)
from insurance_pricing.features.build_features import RISK_CATEGORICAL, RISK_NUMERIC
from insurance_pricing.logger import get_logger

logger = get_logger("insurance_pricing.risk_models")


def build_preprocessor():
    """One-hot categoricals; bin driver age; scale the rest.

    Binning `DrivAge` follows standard pricing practice: risk by age is U-shaped,
    so a linear term would badly misfit young and elderly drivers.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

    binned = ["DrivAge", "VehAge"]
    scaled = [c for c in RISK_NUMERIC if c not in binned]

    # `quantile_method` is required in newer scikit-learn and absent in older
    # versions — detect rather than assume.
    import inspect

    kb_params = inspect.signature(KBinsDiscretizer.__init__).parameters
    kb_kwargs = dict(n_bins=10, encode="onehot-dense", strategy="quantile")
    if "quantile_method" in kb_params:
        kb_kwargs["quantile_method"] = "averaged_inverted_cdf"

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), RISK_CATEGORICAL),
            ("bin", KBinsDiscretizer(**kb_kwargs), binned),
            ("num", StandardScaler(), scaled),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def fit_frequency(train: pd.DataFrame, alpha: float = 1e-4):
    """Poisson GLM for claim *rate*, exposure-weighted."""
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("pre", build_preprocessor()),
                     ("glm", PoissonRegressor(alpha=alpha, max_iter=500))])
    y = train["ClaimNb"] / train["Exposure"]
    pipe.fit(train, y, glm__sample_weight=train["Exposure"])
    return pipe


def fit_severity(train_sev: pd.DataFrame, alpha: float = 1e-4):
    """Gamma GLM for average claim cost, weighted by number of claims."""
    from sklearn.linear_model import GammaRegressor
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("pre", build_preprocessor()),
                     ("glm", GammaRegressor(alpha=alpha, max_iter=500))])
    pipe.fit(train_sev, train_sev["AvgClaimAmount"], glm__sample_weight=train_sev["ClaimNb"])
    return pipe


def fit_tweedie(train: pd.DataFrame, power: float = 1.5, alpha: float = 1e-4):
    """Tweedie GLM straight onto pure premium, exposure-weighted."""
    from sklearn.linear_model import TweedieRegressor
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("pre", build_preprocessor()),
                     ("glm", TweedieRegressor(power=power, alpha=alpha, max_iter=500))])
    pipe.fit(train, train["PurePremium"], glm__sample_weight=train["Exposure"])
    return pipe


def fit_gbm_frequency(train: pd.DataFrame, seed: int = 42):
    """LightGBM challenger with a Poisson objective."""
    from lightgbm import LGBMRegressor
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("gbm", LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.05,
                              num_leaves=31, min_child_samples=100, random_state=seed,
                              n_jobs=-1, verbose=-1)),
    ])
    y = train["ClaimNb"] / train["Exposure"]
    pipe.fit(train, y, gbm__sample_weight=train["Exposure"])
    return pipe


def fit_gbm_tweedie(train: pd.DataFrame, seed: int = 42):
    """LightGBM challenger on pure premium with a Tweedie objective."""
    from lightgbm import LGBMRegressor
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("gbm", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5,
                              n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=100, random_state=seed, n_jobs=-1, verbose=-1)),
    ])
    pipe.fit(train, train["PurePremium"], gbm__sample_weight=train["Exposure"])
    return pipe


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_frequency(name: str, model, test: pd.DataFrame) -> dict:
    y = (test["ClaimNb"] / test["Exposure"]).to_numpy()
    w = test["Exposure"].to_numpy()
    pred = np.clip(model.predict(test), 1e-10, None)
    return {"model": name,
            "poisson_deviance": round(poisson_deviance(y, pred, w), 6),
            "gini": round(gini_norm(y, pred, w), 4)}


def eval_severity(name: str, model, test_sev: pd.DataFrame) -> dict:
    y = test_sev["AvgClaimAmount"].to_numpy()
    w = test_sev["ClaimNb"].to_numpy()
    pred = np.clip(model.predict(test_sev), 1e-10, None)
    return {"model": name,
            "gamma_deviance": round(gamma_deviance(y, pred, w), 4),
            "gini": round(gini_norm(y, pred, w), 4)}


def eval_pure_premium(name: str, pred: np.ndarray, test: pd.DataFrame) -> dict:
    y = test["PurePremium"].to_numpy()
    w = test["Exposure"].to_numpy()
    pred = np.clip(pred, 1e-10, None)
    return {"model": name,
            "tweedie_deviance": round(tweedie_deviance(y, pred, 1.5, w), 4),
            "gini": round(gini_norm(y, pred, w), 4),
            "mean_pred": round(float(np.average(pred, weights=w)), 2),
            "mean_actual": round(float(np.average(y, weights=w)), 2)}


def run(config_path: str = "config/config.yaml") -> None:
    from insurance_pricing.config import load_config

    cfg = load_config(config_path)
    proc = Path("data/processed")
    tr = pd.read_parquet(proc / "risk_train.parquet")
    te = pd.read_parquet(proc / "risk_test.parquet")
    str_, ste = pd.read_parquet(proc / "severity_train.parquet"), pd.read_parquet(proc / "severity_test.parquet")

    # honest note on the known freq/sev reconciliation gap
    with_claims = int((tr["ClaimNb"] > 0).sum())
    logger.info("Train policies with ClaimNb>0: %d | with a matched claim amount: %d (%.1f%% matched)",
                with_claims, len(str_), 100 * len(str_) / max(with_claims, 1))

    freq_rows, sev_rows, pp_rows = [], [], []

    # ---- baselines: charge everyone the portfolio average ----
    base_freq = float(np.average(tr["ClaimNb"] / tr["Exposure"], weights=tr["Exposure"]))
    freq_rows.append(eval_frequency("Baseline (portfolio mean)",
                                    _Const(base_freq), te))
    base_sev = float(np.average(str_["AvgClaimAmount"], weights=str_["ClaimNb"]))
    sev_rows.append(eval_severity("Baseline (mean severity)", _Const(base_sev), ste))
    pp_rows.append(eval_pure_premium("Baseline (mean pure premium)",
                                     np.full(len(te), base_freq * base_sev), te))

    # ---- frequency ----
    logger.info("Fitting Poisson frequency GLM…")
    f_glm = fit_frequency(tr)
    freq_rows.append(eval_frequency("Poisson GLM", f_glm, te))
    logger.info("Fitting LightGBM frequency (Poisson objective)…")
    f_gbm = fit_gbm_frequency(tr, cfg.project.seed)
    freq_rows.append(eval_frequency("LightGBM (Poisson)", f_gbm, te))

    # ---- severity ----
    logger.info("Fitting Gamma severity GLM…")
    s_glm = fit_severity(str_)
    sev_rows.append(eval_severity("Gamma GLM", s_glm, ste))

    # ---- pure premium ----
    logger.info("Fitting Tweedie GLM…")
    t_glm = fit_tweedie(tr)
    pp_rows.append(eval_pure_premium("Frequency x Severity (GLM)",
                                     f_glm.predict(te) * s_glm.predict(te), te))
    pp_rows.append(eval_pure_premium("Tweedie GLM", t_glm.predict(te), te))
    logger.info("Fitting LightGBM Tweedie…")
    t_gbm = fit_gbm_tweedie(tr, cfg.project.seed)
    pp_rows.append(eval_pure_premium("LightGBM (Tweedie)", t_gbm.predict(te), te))

    freq_df, sev_df, pp_df = pd.DataFrame(freq_rows), pd.DataFrame(sev_rows), pd.DataFrame(pp_rows)
    lift = lift_table(te["PurePremium"], t_glm.predict(te), te["Exposure"], n_bins=10)

    print("\n=== FREQUENCY (test) ===");     print(freq_df.to_string(index=False))
    print("\n=== SEVERITY (test) ===");      print(sev_df.to_string(index=False))
    print("\n=== PURE PREMIUM (test) ===");  print(pp_df.to_string(index=False))
    print("\n=== LIFT — Tweedie GLM, by predicted-risk decile ===")
    print(lift.to_string(index=False))

    reports = Path(cfg.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase3a_risk_models.md").write_text(
        "# Phase 3a — risk model leaderboard (test set)\n\n"
        f"Policies with a claim: {with_claims}; with a matched claim amount: {len(str_)} "
        f"({100 * len(str_) / max(with_claims, 1):.1f}% — a known freMTPL2 table-reconciliation gap).\n\n"
        "## Frequency\n\n" + freq_df.to_markdown(index=False)
        + "\n\n## Severity\n\n" + sev_df.to_markdown(index=False)
        + "\n\n## Pure premium\n\n" + pp_df.to_markdown(index=False)
        + "\n\n## Lift — Tweedie GLM\n\n" + lift.to_markdown(index=False)
        + "\n\n*Deviance: lower is better. Gini: higher is better (risk ranking).*\n",
        encoding="utf-8",
    )
    import joblib

    models_dir = Path(cfg.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"frequency": f_glm, "severity": s_glm, "tweedie": t_glm}, models_dir / "risk_glms.joblib")
    logger.info("Saved reports/phase3a_risk_models.md and models/risk_glms.joblib")


class _Const:
    """Constant predictor used for the intercept-only baseline."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x) -> np.ndarray:
        return np.full(len(x), self.value)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
