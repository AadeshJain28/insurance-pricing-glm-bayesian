# ML-3 · Phase 2 — Data Preparation

> Both halves prepared: **risk** (freMTPL2) and **demand** (cross-sell). Includes a formal
> endogeneity investigation that changed the project's scope — documented, not hidden.

## 1. Objectives
1. Build a policy-level risk frame with correct **exposure** handling and severity joined.
2. Apply standard actuarial **capping** and derive model-ready features.
3. Isolate the **severity subset** (claims only — Gamma is undefined at zero).
4. Prepare the demand frame and flag its data-quality issues.
5. **Test whether price elasticity is identifiable** — and act on the answer.

## 2. Findings from the real data

### Risk side (freMTPL2) — validated against benchmark
| quantity | value | expected |
|---|---|---|
| policies | 678,013 | ~678k ✓ |
| claims | 36,102 | ~36k ✓ |
| portfolio frequency | 0.1007 claims/exposure-year | ~0.10 ✓ |
| mean exposure | 0.529 years | ~0.53 ✓ |

**Note on `PurePremium` extremes (max ≈ 18.3M):** this is an artefact of dividing a claim by a
very small exposure (min ≈ 1 day), *not* a real risk signal. It is handled by using
**Exposure as the sample weight** when modelling pure premium, so those rows carry
negligible influence. Claim amounts are capped at 200k *per claim* (a policy with several
claims may therefore exceed the cap in total — intended).

### Demand side (cross-sell) — 381,109 rows, 12.26% conversion
- **53 regions**, ranging from 183 to 106,415 policies — an ideal setting to demonstrate
  hierarchical shrinkage (credibility).
- **17% of premiums equal a single placeholder value (2630)** → flagged, not dropped.
- `Previously_Insured` is near-deterministic: 22.6% vs 0.09% conversion.

### The endogeneity investigation (scope-changing)
| test | corr(price, conversion) |
|---|---|
| all rows | **+0.513** |
| within `Previously_Insured=0`, placeholder removed | **+0.933** |
| by `Vehicle_Damage × Vehicle_Age` | −0.99, −0.84, +0.50, +0.89, +1.00 (**sign unstable**) |

A demand curve requires conversion to **fall** with price. It rises, and controlling for
observables makes the sign unstable rather than fixing it — because premium is set *from risk*.

**Decision:** the cross-sell data is used for a **conversion/propensity model** only. The pricing
step uses a **stated elasticity with sensitivity bands**, and **no causal elasticity is claimed**.
The diagnostic lives in `evaluation/endogeneity.py` and its output in `reports/endogeneity_check.md`.

## 3. Outputs
`data/processed/{risk,severity,demand}_{train,test}.parquet` · `reports/endogeneity_check.md`.

## 4. Theory notes
- **Exposure offset:** frequency is a *rate*; `log(Exposure)` enters as an offset, not a feature.
- **Severity is conditional:** modelled only where `ClaimNb > 0`.
- **Capping:** sparse extreme values (VehAge>20, BonusMalus>150, DrivAge>90) are truncated —
  standard pricing practice to keep the fit stable.
- **Endogeneity:** when the treatment (price) is assigned using the outcome's drivers (risk),
  the observed association is not causal.

## 5. Checklist
- [x] freMTPL2 validated against benchmark frequency
- [x] Severity joined and capped; severity subset isolated
- [x] Actuarial caps + log-density applied
- [x] Demand data loaded; placeholder premium flagged
- [x] Endogeneity tested, verdict recorded, scope adjusted
- [ ] `make_dataset` executed and parquet written

## 6. Common mistakes avoided
Modelling counts without exposure; fitting severity on zero-claim policies; treating the
placeholder premium as a real price; reporting a confounded price coefficient as elasticity.

## 7. Next phase
Phase 3 — frequency GLM (Poisson + offset) → severity GLM (Gamma) → Tweedie → **Bayesian
hierarchical credibility** → GBM comparison, scored on deviance / Gini / lift.
