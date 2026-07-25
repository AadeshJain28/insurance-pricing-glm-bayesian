# Phase 3a — risk model leaderboard (test set)

Policies with a claim: 27214; with a matched claim amount: 19843 (72.9% — a known freMTPL2 table-reconciliation gap).

## Frequency

| model                     |   poisson_deviance |   gini |
|:--------------------------|-------------------:|-------:|
| Baseline (portfolio mean) |           0.627661 | 0.0151 |
| Poisson GLM               |           0.599368 | 0.2837 |
| LightGBM (Poisson)        |           0.578415 | 0.3398 |

## Severity

| model                    |   gamma_deviance |   gini |
|:-------------------------|-----------------:|-------:|
| Baseline (mean severity) |           1.4893 | 0.0787 |
| Gamma GLM                |           1.5291 | 0.0448 |

## Pure premium

| model                        |   tweedie_deviance |   gini |   mean_pred |   mean_actual |
|:-----------------------------|-------------------:|-------:|------------:|--------------:|
| Baseline (mean pure premium) |            83.5937 | 0.0582 |      196.91 |        154.04 |
| Frequency x Severity (GLM)   |            81.83   | 0.2251 |      197.07 |        154.04 |
| Tweedie GLM                  |            79.6862 | 0.2964 |      144.79 |        154.04 |
| LightGBM (Tweedie)           |            79.6717 | 0.3236 |      115.15 |        154.04 |

## Lift — Tweedie GLM

|   decile |   exposure |   predicted |   actual |     n |   a_over_e |
|---------:|-----------:|------------:|---------:|------:|-----------:|
|        1 |    7661.03 |     58.2656 |  91.0604 | 13560 |      1.563 |
|        2 |    7745.35 |     75.9582 |  91.4672 | 13560 |      1.204 |
|        3 |    7803.33 |     87.6639 |  89.9569 | 13560 |      1.026 |
|        4 |    7710.63 |     98.3312 |  85.9827 | 13561 |      0.874 |
|        5 |    7737.51 |    109.221  | 121.703  | 13560 |      1.114 |
|        6 |    7462.92 |    121.946  | 115.549  | 13560 |      0.948 |
|        7 |    7166.87 |    138.995  | 160.622  | 13561 |      1.156 |
|        8 |    6876.45 |    166.953  | 217.932  | 13560 |      1.305 |
|        9 |    5958.16 |    229.091  | 226.723  | 13560 |      0.99  |
|       10 |    5473.95 |    479.61   | 448.107  | 13561 |      0.934 |

*Deviance: lower is better. Gini: higher is better (risk ranking).*
