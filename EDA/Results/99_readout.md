# Script 99 — Temporal coefficient stability audit

Source: genuine frozen Script-73 NHANES, unchanged Script-98 CV configurations.
2005–08 training; 2009–18 frozen test; 2021–23 frozen validation.
Cycle-level refits below are diagnostic comparisons only, never test/validation predictions.

## Frozen Z-main regression by cycle

| outcome   | scope   | cycle   |    n |      R2 |   calibration_slope |   mean_prediction_bias |
|:----------|:--------|:--------|-----:|--------:|--------------------:|-----------------------:|
| PHQ_total | lean    | 2009-10 | 2145 |  0.0317 |              0.8124 |                -0.0743 |
| PHQ_total | lean    | 2011-12 | 1856 |  0.0294 |              0.8312 |                 0.1825 |
| PHQ_total | lean    | 2013-14 | 2095 |  0.0398 |              0.9101 |                 0.1942 |
| PHQ_total | lean    | 2015-16 | 1913 |  0.0275 |              0.7840 |                -0.0066 |
| PHQ_total | lean    | 2017-18 | 1884 |  0.0300 |              0.7840 |                 0.0470 |
| PHQ_total | lean    | 2021-23 | 2247 |  0.0054 |              0.7645 |                -0.6211 |
| PHQ_total | full    | 2009-10 | 2145 |  0.0712 |              0.8867 |                -0.0924 |
| PHQ_total | full    | 2011-12 | 1856 |  0.0919 |              1.0338 |                 0.1429 |
| PHQ_total | full    | 2013-14 | 2095 |  0.0649 |              0.8541 |                 0.0367 |
| PHQ_total | full    | 2015-16 | 1913 |  0.0980 |              1.0979 |                -0.1589 |
| PHQ_total | full    | 2017-18 | 1884 |  0.0765 |              0.9365 |                -0.1714 |
| PHQ_total | full    | 2021-23 | 2247 | -0.0054 |              0.7703 |                -0.9445 |
| Somatic   | lean    | 2009-10 | 2145 |  0.0349 |              0.8342 |                 0.0080 |
| Somatic   | lean    | 2011-12 | 1856 |  0.0420 |              0.9774 |                 0.1617 |
| Somatic   | lean    | 2013-14 | 2095 |  0.0536 |              1.0388 |                 0.1684 |
| Somatic   | lean    | 2015-16 | 1913 |  0.0378 |              0.8647 |                 0.0383 |
| Somatic   | lean    | 2017-18 | 1884 |  0.0422 |              0.8985 |                 0.0372 |
| Somatic   | lean    | 2021-23 | 2247 |  0.0449 |              0.9459 |                -0.1489 |
| Somatic   | full    | 2009-10 | 2145 |  0.0626 |              0.8693 |                -0.0079 |
| Somatic   | full    | 2011-12 | 1856 |  0.0949 |              1.0938 |                 0.1316 |
| Somatic   | full    | 2013-14 | 2095 |  0.0785 |              0.9444 |                 0.0910 |
| Somatic   | full    | 2015-16 | 1913 |  0.0845 |              1.0223 |                -0.0402 |
| Somatic   | full    | 2017-18 | 1884 |  0.0697 |              0.9321 |                -0.0669 |
| Somatic   | full    | 2021-23 | 2247 |  0.0364 |              0.8563 |                -0.3041 |
| CogAff    | lean    | 2009-10 | 2145 |  0.0186 |              0.7923 |                -0.0822 |
| CogAff    | lean    | 2011-12 | 1856 |  0.0112 |              0.6743 |                 0.0207 |
| CogAff    | lean    | 2013-14 | 2095 |  0.0167 |              0.7600 |                 0.0258 |
| CogAff    | lean    | 2015-16 | 1913 |  0.0127 |              0.7100 |                -0.0449 |
| CogAff    | lean    | 2017-18 | 1884 |  0.0112 |              0.6609 |                 0.0098 |
| CogAff    | lean    | 2021-23 | 2247 | -0.0238 |              0.5915 |                -0.4722 |
| CogAff    | full    | 2009-10 | 2145 |  0.0573 |              0.9157 |                -0.0845 |
| CogAff    | full    | 2011-12 | 1856 |  0.0658 |              0.9799 |                 0.0113 |
| CogAff    | full    | 2013-14 | 2095 |  0.0363 |              0.7750 |                -0.0543 |
| CogAff    | full    | 2015-16 | 1913 |  0.0786 |              1.1597 |                -0.1187 |
| CogAff    | full    | 2017-18 | 1884 |  0.0592 |              0.9573 |                -0.1045 |
| CogAff    | full    | 2021-23 | 2247 | -0.0352 |              0.6791 |                -0.6404 |

## Full adjustment minus lean adjustment (Z-main; same observations)

| outcome   | cycle   |    n |   lean_R2 |   full_R2 |   full_minus_lean_R2 |
|:----------|:--------|-----:|----------:|----------:|---------------------:|
| PHQ_total | 2009-10 | 2145 |    0.0317 |    0.0712 |               0.0395 |
| PHQ_total | 2011-12 | 1856 |    0.0294 |    0.0919 |               0.0625 |
| PHQ_total | 2013-14 | 2095 |    0.0398 |    0.0649 |               0.0252 |
| PHQ_total | 2015-16 | 1913 |    0.0275 |    0.0980 |               0.0705 |
| PHQ_total | 2017-18 | 1884 |    0.0300 |    0.0765 |               0.0465 |
| PHQ_total | 2021-23 | 2247 |    0.0054 |   -0.0054 |              -0.0107 |
| Somatic   | 2009-10 | 2145 |    0.0349 |    0.0626 |               0.0277 |
| Somatic   | 2011-12 | 1856 |    0.0420 |    0.0949 |               0.0528 |
| Somatic   | 2013-14 | 2095 |    0.0536 |    0.0785 |               0.0248 |
| Somatic   | 2015-16 | 1913 |    0.0378 |    0.0845 |               0.0466 |
| Somatic   | 2017-18 | 1884 |    0.0422 |    0.0697 |               0.0275 |
| Somatic   | 2021-23 | 2247 |    0.0449 |    0.0364 |              -0.0084 |
| CogAff    | 2009-10 | 2145 |    0.0186 |    0.0573 |               0.0387 |
| CogAff    | 2011-12 | 1856 |    0.0112 |    0.0658 |               0.0546 |
| CogAff    | 2013-14 | 2095 |    0.0167 |    0.0363 |               0.0196 |
| CogAff    | 2015-16 | 1913 |    0.0127 |    0.0786 |               0.0659 |
| CogAff    | 2017-18 | 1884 |    0.0112 |    0.0592 |               0.0481 |
| CogAff    | 2021-23 | 2247 |   -0.0238 |   -0.0352 |              -0.0114 |

## Largest physiological-block coefficient shifts (Z-main refits)

| outcome   | scope   | cycle   | block     |   n_features |   rms_beta_shift |   meaningful_sign_flips |
|:----------|:--------|:--------|:----------|-------------:|-----------------:|------------------------:|
| PHQ_total | lean    | 2017-18 | A_private |            2 |           0.0996 |                       1 |
| Somatic   | lean    | 2017-18 | A_private |            2 |           0.0982 |                       0 |
| Somatic   | full    | 2017-18 | A_private |            2 |           0.0962 |                       1 |
| PHQ_total | full    | 2017-18 | A_private |            2 |           0.0962 |                       1 |
| CogAff    | lean    | 2017-18 | A_private |            2 |           0.0851 |                       1 |
| CogAff    | lean    | 2021-23 | Shared    |            2 |           0.0845 |                       2 |
| PHQ_total | lean    | 2009-10 | A_private |            2 |           0.0806 |                       1 |
| CogAff    | full    | 2017-18 | A_private |            2 |           0.0789 |                       1 |
| CogAff    | lean    | 2009-10 | A_private |            2 |           0.0776 |                       1 |
| PHQ_total | full    | 2009-10 | A_private |            2 |           0.0776 |                       1 |
| PHQ_total | lean    | 2021-23 | Shared    |            2 |           0.0772 |                       1 |
| CogAff    | full    | 2021-23 | Shared    |            2 |           0.0770 |                       1 |
| PHQ_total | lean    | 2021-23 | A_private |            2 |           0.0759 |                       1 |
| CogAff    | full    | 2009-10 | A_private |            2 |           0.0736 |                       1 |
| Somatic   | full    | 2015-16 | Shared    |            2 |           0.0729 |                       0 |
| PHQ_total | full    | 2011-12 | Shared    |            2 |           0.0713 |                       1 |
| Somatic   | lean    | 2009-10 | A_private |            2 |           0.0693 |                       0 |
| CogAff    | lean    | 2021-23 | A_private |            2 |           0.0687 |                       1 |
| Somatic   | full    | 2009-10 | A_private |            2 |           0.0675 |                       0 |
| Somatic   | lean    | 2021-23 | A_private |            2 |           0.0674 |                       0 |
| PHQ_total | full    | 2021-23 | Shared    |            2 |           0.0673 |                       1 |
| CogAff    | full    | 2011-12 | Shared    |            2 |           0.0668 |                       1 |
| PHQ_total | full    | 2021-23 | A_private |            2 |           0.0660 |                       0 |
| CogAff    | full    | 2013-14 | A_private |            2 |           0.0643 |                       1 |

## Interpretation guardrails

- Every original model and regularization strength was selected on 2005–08 only.
- Refit coefficients on later cycles are **descriptive** and use the original discovery feature scaling; their in-sample R² is not a generalization estimate.
- Coefficients are divided by the discovery outcome SD; standardized differences are comparable within the same outcome and feature.
- Sign flips are screened only when both magnitudes exceed 0.02 discovery-outcome SD; no significance test is implied.
- Performance and coefficient drift may reflect sampling variability, phenotype changes, measurement issues, cohort composition or model misspecification.
- Additional adjustment variables must not be interpreted as causes or reliable generalizable predictors solely from CV.
- Later NHANES participants are different people, not longitudinal trajectories.
- All scores are survey-weighted point metrics; no NHANES complex-design uncertainty or bootstrap CIs.
- Script-98 pooled frozen predictions numerically reproduced (maximum R² difference: 4.094e-16).
