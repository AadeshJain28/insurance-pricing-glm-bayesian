# ML-3 — Interview Q&A Bank + Resume Bullet

Every answer is backed by a number this project measured.

## Resume bullet

> **Insurance Pricing: Actuarial Risk Models & Bayesian Credibility** — Built a two-stage pricing
> system on **678k motor policies** (freMTPL2): Poisson-frequency / Gamma-severity / **Tweedie**
> GLMs with correct **exposure offsets**, a **Bayesian hierarchical credibility** model in **PyMC**
> (non-centred, 4 chains, R̂ ≤ 1.003, 0 divergences), and a **profit-optimal pricing** layer.
> Frequency Gini **0.284** (GLM) / **0.340** (LightGBM); diagnosed a **+28% bias** in the classic
> frequency×severity decomposition and showed Tweedie removes it; **proved the quote data could not
> identify a price elasticity** (conversion rises with price) and reported a sensitivity sweep
> instead of a spurious number. *(Python, scikit-learn, PyMC, LightGBM, Streamlit.)*

---

## Actuarial fundamentals

**Q. Why is exposure an offset rather than a feature?**
Frequency is a *rate*. A policy held 3 months isn't comparable to one held 12. `log(Exposure)` enters
with a coefficient fixed at 1. I fitted the rate with `sample_weight=Exposure`, which is provably
identical — I verified it numerically: both give `sum(N)/sum(E)` = 0.098203 exactly.

**Q. Why Gamma for severity?**
Claim amounts are positive, right-skewed, with variance growing with the mean (variance ∝ μ²) —
exactly Gamma's structure. And severity is modelled **only on policies that had a claim**; Gamma is
undefined at zero, and including zero-claim policies would be a category error.

**Q. Frequency × Severity or Tweedie?**
Normally the decomposition, because you can inspect what drives each part. **But on this data
Tweedie is better, and I can say why numerically:** freMTPL2's frequency and severity tables only
reconcile 72.9%, so frequency counts claims that severity never sees. The product over-predicts by
**28%** (197 vs actual 154); Tweedie models pure premium directly and lands at 145.

**Q. Why Gini and deviance, not R² or accuracy?**
Deviance is the GLM's own loss. Gini measures *ranking* — insurers segment prices, so ordering risk
correctly matters more than any single policy's error. R² is meaningless for counts; accuracy is
meaningless for a continuous skewed target.

## The severity result

**Q. Your severity model was worse than a constant. What happened?**
Overfitting: ~60 one-hot parameters on 19,843 extremely noisy claims (test deviance 1.5291 vs
baseline 1.4893). That's a textbook small-sample/high-variance failure — and precisely the problem
credibility theory exists to solve.

**Q. Did Bayesian pooling fix it?**
Partially, and I report that honestly. Pooling improved on unpooled cell means (1.6268 → 1.5691),
but never beat the flat average — at *either* granularity (coarse cells: 1.4989). The Ginis went
negative, meaning cell-level severity predictions rank worse than random out-of-sample. The
conclusion is that **`Region × VehBrand` carries no severity signal**, so the correct price for
severity is the portfolio mean. Shrinkage can't extract signal that isn't there.

## Bayesian / credibility

**Q. How does hierarchical Bayes relate to credibility theory?**
They're the same idea. A hierarchical prior shrinks a cell's estimate toward the portfolio mean in
proportion to how much data backs it — which is Bühlmann's `Z = n/(n+k)`, derived in the 1960s.
Modern MCMC just lets you fit it without closed-form assumptions.

**Q. Show me the shrinkage actually worked.**
Small cells (n<50) moved **0.699** of the way to the mean; large cells (n>500) moved **0.047** —
a 13× difference. Concretely: a cell with one €40,408 claim was shrunk to €2,055 (fully pooled),
while a cell with 1,695 claims moved 3%.

**Q. You had divergences. What did you do?**
I hit **Neal's funnel** — the centred parameterisation `theta ~ N(mu, tau)` makes the geometry
pathological as `tau → 0`, giving 1000 divergences and R̂ 1.53. I switched to the **non-centred**
form (`theta = mu + tau·z`, `z ~ N(0,1)`), which samples the identical distribution but with benign
geometry — divergences fell to 31, then to 0 for the primary models. Raising `target_accept` alone
wouldn't have fixed it, because the problem is geometry, not step size.

**Q. Why do you report R̂ and ESS?**
Because MCMC that hasn't converged gives numbers that look fine and mean nothing. My primary models
reach R̂ ≤ 1.0025 with 0 divergences. The coarse 22-cell model stayed at R̂ 1.012 — two variance
components are weakly identified from 22 groups — so I flagged it as a robustness check rather than
publishing it as a result.

## Pricing & causality

**Q. How do you get from risk cost to the price you charge?**
`max_P (P − technical_cost) × P(convert | P)`. Margin rises with price, conversion falls; the peak
is the profit-maximising quote.

**Q. What's your price elasticity?**
**I don't have one, and I won't invent one.** I tested whether the data could identify it: conversion
*rises* with price (corr +0.513 overall, +0.933 within the converting segment), and the sign flips
across segments when I control for risk. That's endogeneity — premium is set from risk, so the
coefficient measures customer mix. I report a sensitivity sweep instead: at elasticity −0.5 the
optimal markup is 3.0×, at −5.0 it's 1.35×.

**Q. What would you need to measure elasticity properly?**
Randomised or quasi-randomised price variation — a price test, a discontinuity in a rating rule, or
an instrument that shifts price without touching risk.

## Regulated ML

**Q. LightGBM beat your GLMs. Why not just ship it?**
Two reasons. Regulators require explainable, auditable rating factors, and GLMs give coefficients you
can defend. And on this data the GBM had the best Gini (0.324) but the **worst calibration** (−25%
vs actual) — great ranking, unusable absolute prices. In practice: GLM for the filed rating plan,
GBM to discover structure you then encode as GLM factors.
