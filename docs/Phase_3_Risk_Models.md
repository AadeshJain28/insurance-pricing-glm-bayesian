# ML-3 · Phase 3 — Risk Models (the technical price)

> **3a (now):** frequency GLM → severity GLM → Tweedie → GBM challenger, on deviance/Gini/lift.
> **3b (next):** Bayesian hierarchical credibility.

## 1. The three routes to a technical price

| route | target | family | weight | note |
|---|---|---|---|---|
| Frequency | `ClaimNb / Exposure` | Poisson | `Exposure` | claim **rate**, not count |
| Severity | `ClaimAmount / ClaimNb` | Gamma | `ClaimNb` | **only** where a claim occurred |
| Pure premium | `frequency × severity` | — | — | the classic decomposition |
| Pure premium (direct) | `ClaimAmount / Exposure` | Tweedie (p=1.5) | `Exposure` | compound Poisson-Gamma in one model |

**Exposure handling — the detail that signals actuarial literacy.** Fitting the *rate* with
`sample_weight = Exposure` is mathematically identical to fitting the *count* with a
`log(Exposure)` offset. Verified numerically: for a constant model both give
`sum(N)/sum(E)` exactly (0.098203 vs 0.098203 on a simulated book).

## 2. Why these metrics
- **Deviance** is the GLM's own loss — the correct in-family measure.
- **Gini** measures *ranking*: an insurer segments prices, so ordering risk correctly matters more
  than nailing any single policy.
- **Lift by decile** exposes calibration (actual vs expected) across the risk spectrum.
- **No R², no accuracy** — meaningless for counts and heavily skewed severities.

## 3. Baselines
An intercept-only model ("charge everyone the portfolio average") is included for every target.
A risk model that cannot beat it has no commercial value. Its Gini is ≈ 0 by construction.

## 4. Modelling choices
- `DrivAge` and `VehAge` are **binned** (10 quantile bins): risk by age is U-shaped, so a linear
  term would misfit young and elderly drivers badly.
- Categoricals are one-hot with `min_frequency=50` to avoid unstable rare levels.
- Light L2 (`alpha=1e-4`) for numerical stability.
- **GBM challenger** (LightGBM, Poisson/Tweedie objectives) quantifies what the GLM gives up for
  interpretability — the live GLM-vs-ML debate in regulated pricing.

## 5. Known data quirk (reported, not hidden)
freMTPL2's frequency and severity tables do not fully reconcile: ~25% of policies with
`ClaimNb > 0` have no matching claim amount. Frequency uses all policies; severity uses only
matched claims. The script prints the matched percentage every run.

## 6. Interview questions this phase answers
"Why an offset instead of a feature?" · "Why Gamma for severity?" · "When would you use Tweedie
instead of two models?" · "Why Gini rather than R²?" · "Why do insurers still prefer GLMs?" ·
"What does a lift table tell you that deviance doesn't?"

## 7. Checklist
- [ ] Frequency GLM beats the baseline on Poisson deviance and Gini
- [ ] Severity GLM fitted on claims only
- [ ] Freq × Sev and Tweedie compared on the same pure-premium scale
- [ ] GBM challenger measured
- [ ] Lift table produced and inspected for calibration
- [ ] Reconciliation gap reported

## 8. Next
Phase 3b — Bayesian hierarchical credibility on region/brand, showing shrinkage of small,
noisy segments toward the portfolio mean.
