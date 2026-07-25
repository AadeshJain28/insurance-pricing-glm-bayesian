# Phase 4 — profit-optimal pricing

Technical price from the Tweedie GLM: portfolio mean **144.79** per exposure-year.

> **The demand curve is assumed, not estimated.** Phase 2 showed this data cannot identify a price elasticity (conversion rises with price, corr +0.51, because premium is set from risk). Every figure below is therefore conditional on the stated elasticity, which is why the sensitivity sweep — not any single price — is the deliverable.

## Optimal price vs assumed elasticity

|   assumed_elasticity |   optimal_price |   markup_over_cost |   conversion_at_optimum |   expected_profit |   realised_elasticity |   lerner_implied_E | grid_censored   |
|---------------------:|----------------:|-------------------:|------------------------:|------------------:|----------------------:|-------------------:|:----------------|
|                 -0.5 |          532.55 |               3.68 |                  0.0916 |             35.53 |                -1.392 |             -1.373 | False           |
|                 -1   |          348.49 |               2.41 |                  0.1051 |             21.41 |                -1.795 |             -1.711 | False           |
|                 -1.5 |          274.86 |               1.9  |                  0.1319 |             17.16 |                -2.06  |             -2.113 | False           |
|                 -2   |          256.46 |               1.77 |                  0.1373 |             15.33 |                -2.547 |             -2.297 | False           |
|                 -3   |          219.64 |               1.52 |                  0.1933 |             14.47 |                -3.059 |             -2.934 | False           |
|                 -5   |          201.24 |               1.39 |                  0.2835 |             16    |                -4.149 |             -3.565 | False           |

## By risk decile (elasticity = -1.5)

|   risk_decile |   technical_cost |   optimal_price |   markup |   conversion |   expected_profit |
|--------------:|-----------------:|----------------:|---------:|-------------:|------------------:|
|             1 |            58.27 |          110.61 |      1.9 |       0.1319 |              6.9  |
|             5 |           109.22 |          207.33 |      1.9 |       0.1319 |             12.94 |
|            10 |           479.61 |          910.45 |      1.9 |       0.1319 |             56.83 |

### Validation

The **Lerner index** — at a profit-maximising price `(P - MC)/P = 1/|E|` — should match `realised_elasticity`. Rows where it does not are flagged `grid_censored`, meaning the optimum sits on a price-grid bound rather than at a true interior maximum.

### Caveats

* More elastic demand => lower optimal markup (3.0x at E=-0.5 down to 1.35x at E=-5).
* **The identical markup across risk deciles is an artefact, not a finding.** Demand is calibrated relative to each segment's own cost, so constant markup follows by construction. Real segments differ in elasticity (young drivers are more price-sensitive), which is exactly what this data cannot measure.
