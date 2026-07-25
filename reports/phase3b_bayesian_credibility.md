# Phase 3b — Bayesian hierarchical credibility

MCMC: 4 chains, 3000 draws, 2000 tune, target_accept=0.95. **Max R-hat = 1.0122**.

## Frequency (test)

| cells             | model              |   poisson_deviance |   gini |
|:------------------|:-------------------|-------------------:|-------:|
| Region x VehBrand | Baseline           |           0.627661 | 0.0151 |
| Region x VehBrand | Raw cell means     |           0.626577 | 0.0923 |
| Region x VehBrand | Hierarchical Bayes |           0.624875 | 0.0926 |
| Region            | Baseline           |           0.627661 | 0.0151 |
| Region            | Raw cell means     |           0.626063 | 0.0714 |
| Region            | Hierarchical Bayes |           0.626064 | 0.0717 |

## Severity (test)

| cells             | model              |   gamma_deviance |    gini |
|:------------------|:-------------------|-----------------:|--------:|
| Region x VehBrand | Baseline           |           1.4893 |  0.0787 |
| Region x VehBrand | Raw cell means     |           1.6268 | -0.0559 |
| Region x VehBrand | Hierarchical Bayes |           1.5691 | -0.0632 |
| Region            | Baseline           |           1.4893 |  0.0787 |
| Region            | Raw cell means     |           1.5123 |  0.0158 |
| Region            | Hierarchical Bayes |           1.4989 |  0.0072 |

## Convergence

| cells             | parameter   |    mean |     sd |   eti89_lb |   eti89_ub |   ess_bulk |   ess_tail |   r_hat |   mcse_mean |   mcse_sd |   divergences | model     |
|:------------------|:------------|--------:|-------:|-----------:|-----------:|-----------:|-----------:|--------:|------------:|----------:|--------------:|:----------|
| Region x VehBrand | mu          | -2.3594 | 0.0194 |    -2.3904 |    -2.3286 |   3808.29  |   5691.1   |  1.0014 |      0.0003 |    0.0002 |             0 | frequency |
| Region x VehBrand | tau         |  0.2149 | 0.0168 |     0.1893 |     0.243  |   3299.05  |   5085.69  |  1.0009 |      0.0003 |    0.0002 |             0 | frequency |
| Region x VehBrand | mu          |  7.4466 | 0.0286 |     7.4008 |     7.4921 |   3603.82  |   4558.2   |  1.0025 |      0.0005 |    0.0003 |             0 | severity  |
| Region x VehBrand | tau         |  0.2907 | 0.0303 |     0.243  |     0.34   |   1381.95  |   2504.53  |  1.0023 |      0.0008 |    0.0006 |             0 | severity  |
| Region x VehBrand | sigma       |  1.4602 | 0.1458 |     1.2344 |     1.6991 |   1383.17  |   2262.03  |  1.0022 |      0.0039 |    0.0029 |             0 | severity  |
| Region            | mu          | -2.2884 | 0.0346 |    -2.3432 |    -2.2341 |   1702.14  |   2498.73  |  1.0028 |      0.0008 |    0.0006 |             0 | frequency |
| Region            | tau         |  0.1535 | 0.0284 |     0.1145 |     0.2043 |   2232.77  |   3268.57  |  1.0045 |      0.0006 |    0.0005 |             0 | frequency |
| Region            | mu          |  7.5135 | 0.0457 |     7.4389 |     7.5843 |   1475.99  |   1273.17  |  1.0014 |      0.0012 |    0.0009 |           136 | severity  |
| Region            | tau         |  0.1758 | 0.0622 |     0.0834 |     0.2793 |    621.551 |   1189.98  |  1.01   |      0.0025 |    0.0017 |           136 | severity  |
| Region            | sigma       |  1.8802 | 0.7139 |     0.7381 |     3.0111 |    509.615 |    646.506 |  1.0122 |      0.0308 |    0.0181 |           136 | severity  |

## Shrinkage — 5 smallest severity cells (fine)

|   n |      raw |   shrunk |   moved_toward_mean_frac |
|----:|---------:|---------:|-------------------------:|
|   1 |  1204    |  1765.58 |                     0.75 |
|   1 |  1204    |  1764.22 |                     0.74 |
|   1 | 40408.1  |  2031.67 |                     1    |
|   1 |  5912.28 |  1880.43 |                     1.02 |
|   1 |  1320    |  1772.11 |                     0.71 |

## Shrinkage — 5 largest severity cells (fine)

|    n |     raw |   shrunk |   moved_toward_mean_frac |
|-----:|--------:|---------:|-------------------------:|
| 1695 | 2313.86 |  2304.99 |                     0.02 |
| 1672 | 2272.33 |  2264.13 |                     0.03 |
|  926 | 1715.61 |  1718.8  |                     0.01 |
|  872 | 1789.13 |  1789.13 |                     0    |
|  798 | 1862.92 |  1860.64 |                    -0.02 |

*`moved_toward_mean_frac` = 1.0 means fully shrunk to the portfolio mean, 0.0 means the raw cell estimate was trusted entirely — the Bayesian counterpart of the Buhlmann-Straub credibility factor Z = n/(n+k).*
