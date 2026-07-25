# Endogeneity check — can this data identify a price elasticity?

A demand curve requires conversion to **fall** as price rises.

## Conversion by price decile

|   bin |   mean_price |   conversion |     n |
|------:|-------------:|-------------:|------:|
|     1 |      2630    |       0.1299 | 38111 |
|     2 |      7629.98 |       0.1181 | 38111 |
|     3 |     24251.5  |       0.0966 | 38111 |
|     4 |     27707.8  |       0.1016 | 38111 |
|     5 |     30370.4  |       0.1103 | 38111 |
|     6 |     33013.7  |       0.114  | 38110 |
|     7 |     35930    |       0.1242 | 38111 |
|     8 |     39483.4  |       0.1353 | 38111 |
|     9 |     44703    |       0.1439 | 38111 |
|    10 |     59924.3  |       0.1518 | 38111 |

**Overall correlation(price, conversion) = +0.513**

## Within segments

| Vehicle_Damage   | Vehicle_Age   |      n |   corr_price_conversion |
|:-----------------|:--------------|-------:|------------------------:|
| No               | 1-2 Year      |  72091 |                  -0.99  |
| No               | < 1 Year      | 116590 |                  -0.839 |
| Yes              | 1-2 Year      | 128225 |                   0.504 |
| Yes              | < 1 Year      |  48196 |                   0.887 |
| Yes              | > 2 Years     |  15992 |                   0.996 |

## Verdict

NOT IDENTIFIABLE: conversion rises with price (corr=+0.513). Price is set from risk, so this is an association, not a causal elasticity.

Consequently the pricing step uses a **stated elasticity with sensitivity bands**, and no causal elasticity is claimed from this data.
