# 🚗 Insurance Pricing — Frequency-Severity GLMs, Bayesian Credibility & Profit-Optimal Price

![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> Two-stage pricing the way insurers actually do it: a **technical price** from actuarial risk
> models, then a **commercial price** from demand — with an honest account of what the data can
> and cannot support.

## Results

### Risk models (freMTPL2, 678k policies, 542k train / 136k test)

| target | model | deviance ↓ | Gini ↑ |
|---|---|---|---|
| Frequency | Baseline | 0.6277 | 0.015 |
| Frequency | **Poisson GLM** | 0.5994 | **0.284** |
| Frequency | LightGBM (Poisson) | **0.5784** | **0.340** |
| Severity | **Baseline** | **1.4893** | **0.079** |
| Severity | Gamma GLM | 1.5291 | 0.045 |
| Pure premium | Baseline | 83.59 | 0.058 |
| Pure premium | Freq × Severity | 81.83 | 0.225 |
| Pure premium | **Tweedie GLM** | 79.69 | 0.296 |
| Pure premium | LightGBM (Tweedie) | **79.67** | **0.324** |

### Bayesian hierarchical credibility (PyMC, 4 chains, non-centred)

| cells | model | Gamma deviance | Gini |
|---|---|---|---|
| Region × VehBrand | Hierarchical Bayes | 1.5691 | −0.063 |
| Region (22) | Hierarchical Bayes | 1.4989 | 0.007 |
| — | **Baseline (flat severity)** | **1.4893** | **0.079** |

**Shrinkage** (the credibility signature): small cells (n<50) move **0.699** of the way to the
portfolio mean; large cells (n>500) move only **0.047**. One cell with a single €40,408 claim was
shrunk to €2,055 — the model correctly refuses to believe one observation.

**Convergence:** the primary (fine-cell) models are fully converged — max R̂ **1.0025**, **0
divergences**, ESS 1,382–3,808. The coarse 22-cell severity model remains marginal (R̂ 1.012,
136 divergences) because two variance components are weakly identified from 22 groups; it is
reported as a robustness check, not a headline.

## Five findings worth defending

1. **Exposure must be an offset, not a feature.** Fitting the *rate* with `sample_weight=Exposure`
   is provably identical to a `log(Exposure)` offset on counts — verified numerically
   (0.098203 both ways).
2. **The severity GLM is worse than a constant** (1.5291 vs 1.4893). ~60 parameters on 19,843 very
   noisy claims. Severity is genuinely hard to predict; frequency does the work.
3. **Frequency × Severity is biased +28% here** (predicts 197 vs actual 154) because freMTPL2's
   frequency and severity tables reconcile only 72.9%. **Tweedie sidesteps it** (145 vs 154) by
   modelling pure premium directly — a concrete, quantified reason to prefer it on this data.
4. **Ranking ≠ calibration.** LightGBM has the best Gini (0.324) but the worst calibration (−25%).
   You cannot charge premiums that rank well but sit 25% below cost.
5. **Bayesian pooling helps frequency but cannot rescue severity.** Credibility improved severity
   over unpooled cells (1.6268 → 1.5691) yet never beat the flat average, at either granularity —
   so the correct actuarial answer here is to price severity at the portfolio mean.

## Honesty: the elasticity that isn't there

The quote dataset **cannot identify a price elasticity**. Conversion *rises* with price:

| test | corr(price, conversion) |
|---|---|
| all 381k rows | **+0.513** |
| within `Previously_Insured=0`, placeholder premium removed | **+0.933** |
| by `Vehicle_Damage × Vehicle_Age` | −0.99, −0.84, +0.50, +0.89, +1.00 (**sign unstable**) |

Premium is set *from risk*, so the coefficient measures customer mix, not price response
(endogeneity). Rather than report a confidently wrong number, the pricing step uses a **stated
elasticity with a sensitivity sweep** — and the sweep, not any single price, is the deliverable.

| assumed elasticity | optimal markup | conversion |
|---|---|---|
| −0.5 | 3.00× | 11.8% |
| −1.5 | 1.90× | 13.2% |
| −5.0 | 1.35× | 32.1% |

## Quickstart
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .

python -m insurance_pricing.data.make_dataset       # freMTPL2 auto-downloads from OpenML
python -m insurance_pricing.models.risk_models      # GLMs + GBM challengers
python -m insurance_pricing.models.bayesian_hier    # PyMC credibility
python -m insurance_pricing.pricing.run_pricing     # profit-optimal price + sensitivity
python -m streamlit run app/streamlit_app.py        # quote demo
```

## Tech stack
Python 3.11 · scikit-learn (Poisson/Gamma/Tweedie GLMs) · **PyMC + ArviZ** · LightGBM · pandas ·
Streamlit · pytest · Ruff/Black/mypy.

## License & data
MIT. freMTPL2 is fetched from OpenML; the cross-sell CSV comes from Kaggle. Neither is
redistributed — `data/` and `models/` are git-ignored.
