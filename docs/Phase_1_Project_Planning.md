# ML-3 · Phase 1 — Project Planning & Scoping
### Insurance Pricing: Frequency-Severity GLMs, Bayesian Credibility, and Profit-Optimal Price

> **Status:** Draft for your approval. No code yet.
> **Decisions locked:** Both datasets (Option A) · **Actuarial modelling is the headline** · Technical price (risk) + Commercial price (demand) → profit optimisation.

---

## 1. The framing (why this structure)

Real insurers price in two stages. This project builds both, then joins them:

| stage | question | model | data |
|---|---|---|---|
| **Technical price** | What will this policy *cost* us? | frequency × severity (Poisson × Gamma), Tweedie | **freMTPL2** (French motor claims) |
| **Commercial price** | Will the customer *buy* at price P? | conversion / demand model | **Vehicle-insurance cross-sell** (quote-side) |
| **Optimal price** | What price maximises expected profit? | `max_P (P − technical_cost) × P(convert \| P)` | both |

**Headline (per your choice):** the actuarial half — frequency-severity GLMs with **Bayesian hierarchical credibility** and a **GLM-vs-GBM** comparison. The pricing optimisation is the closing act that shows the OR bridge.

## 2. Objectives
1. Build a correct **frequency-severity** risk model with proper **exposure** handling.
2. Add a **Bayesian hierarchical** model — the modern form of actuarial **credibility theory**.
3. Compare **GLM vs GBM** on the regulator-friendliness/accuracy trade-off.
4. Build a **conversion (demand)** model, honestly treating **price endogeneity**.
5. Combine into a **profit-maximising price** with sensitivity analysis.
6. Ship a Streamlit quote-and-price demo.

## 3. Datasets

### 3.1 Risk side — freMTPL2 (headline)
- **Source:** OpenML via `sklearn.datasets.fetch_openml("freMTPL2freq")` and `"freMTPL2sev"` — **no manual download**.
- **Size:** ~678k policies; `ClaimNb` (claim count), `Exposure` (years at risk), plus `VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, Region`.
- **Severity:** `freMTPL2sev` gives individual claim amounts, joined on policy id.
- **Why:** the standard actuarial benchmark — real exposure, real claim counts and amounts, interpretable features, 22 regions for hierarchy.

### 3.2 Demand side — Vehicle-Insurance Cross-Sell (Kaggle)
- **Columns:** `Annual_Premium` (price), `Response` (0/1 interest), `Age, Gender, Region_Code` (53 regions), `Vehicle_Age, Vehicle_Damage, Previously_Insured, Vintage, Policy_Sales_Channel`.
- **Why:** it has both a **price** and a **binary outcome** — the minimum needed for a demand curve — and it's motor insurance, so it sits in the same domain as freMTPL2.
- **Honest limitation (must be stated in the README):** `Response` is *cross-sell interest*, not accept/reject of a specific quoted price, and **price is not randomised**. Premium is set from risk, so a naive `Response ~ Premium` regression measures **customer mix, not causal elasticity** (endogeneity). We will (a) say so plainly, (b) control for observed risk drivers, and (c) present the demand curve as a **conditional association with a sensitivity band**, never as a causal elasticity.

## 4. Metrics (and what we refuse to use)

| model | primary | secondary | banned |
|---|---|---|---|
| Frequency | **Poisson deviance** | Gini / lift by decile | R² (meaningless for counts) |
| Severity | **Gamma deviance** | actual-vs-expected | RMSE alone (skewed target) |
| Combined pure premium | **Tweedie deviance** | **Gini**, lift curve, calibration by decile | accuracy |
| Bayesian | posterior predictive checks, **R̂ / ESS** | shrinkage plots | point-estimate-only comparison |
| Conversion | **AUC**, log-loss | calibration curve | accuracy (imbalance) |
| Pricing | expected profit, optimal price | sensitivity to elasticity | — |

**Lift/Gini matters most commercially:** an insurer cares whether the model *ranks* risk correctly (so it can segment prices), not whether it nails each individual claim.

## 5. Theory (what · why · interview · mistakes)

### 5.1 Exposure as an offset
A policy held 3 months isn't comparable to one held 12. Poisson frequency GLM uses `log(Exposure)` as an **offset** (coefficient fixed at 1), modelling *rate* not count.
*Interview:* "Why an offset and not a feature?" *Mistake:* ignoring exposure — the single most common error in portfolio insurance projects.

### 5.2 Frequency × Severity vs Tweedie
Pure premium = E[N] × E[X]. Either model them separately (Poisson + Gamma) or jointly via **Tweedie** (compound Poisson-Gamma, `1 < p < 2`).
*Interview:* "Why Gamma for severity?" (positive, right-skewed, variance ∝ mean²). "When Tweedie over two models?" (simpler, one model; but you lose the ability to inspect frequency and severity drivers separately).

### 5.3 Bayesian hierarchical = credibility theory
Regions with few policies produce noisy estimates. A hierarchical prior **shrinks** them toward the portfolio mean, with shrinkage proportional to data volume — which is *exactly* Bühlmann credibility, derived in the 1960s. Framing it this way is the single strongest talking point in this project.
*Interview:* "How does Bayesian hierarchy relate to credibility?", "What does partial pooling do to a small region's estimate?", "R̂ and ESS — what are you checking?"
*Runtime plan:* fit on **aggregated region × vehicle-brand cells** (exposure-weighted), not 678k rows — faster and closer to actuarial practice.

### 5.4 GLM vs GBM in a regulated industry
GBMs usually predict better; GLMs are transparent, monotonic-constrainable and regulator-approved. We measure the gap and discuss it.
*Interview:* "Why do insurers still use GLMs?", "How would you deploy a GBM under regulatory constraints?" (monotonic constraints, SHAP, GLM surrogate).

### 5.5 Price optimisation
`max_P profit(P) = (P − technical_cost) × P(convert | P, X)`.
*Interview:* "How does optimal price shift with elasticity?", "What stops you charging the profit-maximising price?" (regulation, fairness, retention, competitive response).

## 6. Repository architecture (reuses the ML-1/ML-2 template)

```
insurance-pricing/
├── config/config.yaml
├── src/insurance_pricing/
│   ├── config.py  logger.py
│   ├── data/       load_fremtpl.py  load_quotes.py  validate.py
│   ├── features/   build_features.py
│   ├── models/     glm_frequency.py  glm_severity.py  tweedie.py
│   │               bayesian_hier.py  gbm.py  conversion.py
│   ├── pricing/    optimize.py            # profit-maximising price
│   └── evaluation/ metrics.py (deviance, Gini, lift, calibration)
├── app/streamlit_app.py                   # quote -> risk cost -> optimal price
├── tests/  reports/  docs/  .github/workflows/ci.yml
```

## 7. Environment
Python 3.11 · pandas, numpy, scikit-learn (has Poisson/Gamma/Tweedie GLMs built in) · **statsmodels** (GLM inference, standard errors) · **PyMC** or **NumPyro** (Bayesian) · lightgbm · matplotlib/seaborn · streamlit · pytest/ruff/black/mypy.

## 8. Roadmap
Phase 1 plan → scaffold → **Phase 2** (load both datasets, EDA, exposure handling, severity join, features, splits) → **Phase 3** (frequency GLM → severity GLM → Tweedie → Bayesian hierarchical → GBM comparison → lift/Gini) → **Phase 4** (conversion model + profit optimisation + Streamlit) → **Phase 5** (README, resume bullet, interview bank, push).

## 9. Deliverables
Frequency/severity/Tweedie GLMs with deviance + Gini + lift; Bayesian hierarchical posterior + shrinkage plot; GLM-vs-GBM table; conversion model with calibration; optimal-price curve + sensitivity; Streamlit demo; full docs.

## 10. Checklist (Phase 1)
- [ ] Two-stage framing agreed (technical + commercial)
- [ ] Datasets agreed (freMTPL2 via OpenML; cross-sell via Kaggle)
- [ ] Metrics agreed (deviance/Gini/lift; no R²)
- [ ] Exposure-as-offset agreed
- [ ] Bayesian on aggregated cells agreed
- [ ] Endogeneity caveat agreed (no causal elasticity claim)
- [ ] Repo architecture approved

## 11. Common mistakes (that we will avoid)
Ignoring exposure; using R²/accuracy; modelling severity on zero-claim policies; fitting Bayesian MCMC on 678k rows; claiming causal elasticity from observational prices; leaking claim info into the conversion model; forgetting that severity is conditional on a claim occurring.

## 12. Validation criteria (Phase 1 done)
You approve the framing, datasets, metrics, and architecture; the scaffold installs; freMTPL2 fetches successfully and the Kaggle quote CSV is in `data/raw/`.

## 13. References
Ohlsson & Johansson, *Non-Life Insurance Pricing with GLMs* · Bühlmann (1967), credibility · Wüthrich & Merz, *Statistical Foundations of Actuarial Learning* · Gelman et al., *Bayesian Data Analysis* (hierarchical models) · scikit-learn "Tweedie regression on insurance claims" example · Denuit et al. on Gini/lift for pricing.

## 14. Next phase requirements
1. You approve this plan.
2. I scaffold the repo.
3. You run the scaffold + `fetch_openml` smoke test, and download the cross-sell CSV to `data/raw/`.
4. Then Phase 2.
