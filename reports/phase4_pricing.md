# Phase 4 — profit-optimal pricing

Technical price from the Tweedie GLM: portfolio mean **144.79** per exposure-year.

> **The demand curve is assumed, not estimated.** Phase 2 showed this data cannot identify a price elasticity (conversion rises with price, corr +0.51, because premium is set from risk). Every figure below is therefore conditional on the stated elasticity, which is why the sensitivity sweep — not any single price — is the deliverable.

## Optimal price vs assumed elasticity

|   assumed_elasticity |   optimal_price |   markup_over_cost |   conversion_at_optimum |   expected_profit |   realised_elasticity |
|---------------------:|----------------:|-------------------:|------------------------:|------------------:|----------------------:|
|                 -0.5 |          434.38 |               3    |                  0.118  |             34.18 |                -1.102 |
|                 -1   |          342.35 |               2.36 |                  0.1085 |             21.43 |                -1.757 |
|                 -1.5 |          274.86 |               1.9  |                  0.1319 |             17.16 |                -2.06  |
|                 -2   |          244.19 |               1.69 |                  0.1549 |             15.39 |                -2.375 |
|                 -3   |          219.64 |               1.52 |                  0.1933 |             14.47 |                -3.059 |
|                 -5   |          195.1  |               1.35 |                  0.3207 |             16.13 |                -3.814 |

## By risk decile (elasticity = -1.5)

|   risk_decile |   technical_cost |   optimal_price |   markup |   conversion |   expected_profit |
|--------------:|-----------------:|----------------:|---------:|-------------:|------------------:|
|             1 |            58.27 |          110.61 |      1.9 |       0.1319 |              6.9  |
|             5 |           109.22 |          207.33 |      1.9 |       0.1319 |             12.94 |
|            10 |           479.61 |          910.45 |      1.9 |       0.1319 |             56.83 |

*More elastic demand (more negative) => lower optimal markup. Higher-risk policies carry a higher absolute price but a similar markup, because the demand curve is calibrated relative to each segment's own technical cost.*
