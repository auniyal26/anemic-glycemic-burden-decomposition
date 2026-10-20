# Script 11 — MIDJA longitudinal validation and failure audit

## Study

- HbA1c panel: adjusted; MIDJA1 complete N=353; MIDJA2 complete N=303.
- Strict chronological complete pairs: 211; median follow-up interval: 53.0 months.
- Baseline CES-D mean: 12.878; later CES-D mean: 9.938.
- All predictive variables measured at MIDJA1; elapsed follow-up time is descriptive and never used to predict.
- Baseline PCA refitted inside each participant-held-out training fold; no NHANES/MIDUS loadings used.
- Repeated five-fold CV: 20 partitions; Ridge alpha=10; identical splits for comparisons.

## Repeated-CV predictive contribution

ΔMSE = baseline MSE − candidate MSE; positive favors the candidate.

| Adjustment | Model | Mean ΔMSE vs baseline | Minimum to maximum across CV splits | Fraction positive | Mean attrition-weighted ΔMSE |
|---|---|---:|---|---:|---:|
| minimal | prior_CESD | +0.0000 | [+0.0000, +0.0000] | 0.00 | +0.0000 |
| minimal | HbA1c | +1.0102 | [+0.2147, +1.3877] | 1.00 | +0.9536 |
| minimal | Hb | +0.1330 | [-0.9219, +0.5524] | 0.80 | +0.1735 |
| minimal | hematology_raw | +1.2155 | [+0.2581, +2.0350] | 1.00 | +1.8723 |
| minimal | raw_X4 | +1.9138 | [-0.0951, +2.6945] | 0.95 | +2.4341 |
| minimal | PC1_2 | +0.8352 | [+0.2324, +1.5769] | 1.00 | +1.6594 |
| minimal | PC1_2_3 | +1.4901 | [-0.5228, +2.2290] | 0.95 | +2.0057 |
| minimal | PC1_2_3_4 | +2.0153 | [+0.1427, +2.9152] | 1.00 | +2.5361 |
| minimal | A_PC1_2_plus_G | +1.4817 | [-0.5274, +2.2167] | 0.95 | +2.0013 |
| expanded | prior_CESD | +0.0000 | [+0.0000, +0.0000] | 0.00 | +0.0000 |
| expanded | HbA1c | +0.5755 | [-0.1058, +1.2617] | 0.90 | +3.4551 |
| expanded | Hb | -0.2648 | [-1.1446, +0.1825] | 0.20 | -1.8400 |
| expanded | hematology_raw | +0.3543 | [-0.5356, +1.6637] | 0.70 | -1.3975 |
| expanded | raw_X4 | +0.8321 | [-0.3762, +2.1993] | 0.95 | +2.0419 |
| expanded | PC1_2 | -0.3343 | [-1.2174, +0.9782] | 0.20 | -3.6426 |
| expanded | PC1_2_3 | +0.2307 | [-0.9148, +1.5562] | 0.65 | +0.1243 |
| expanded | PC1_2_3_4 | +1.2099 | [-0.0761, +2.6222] | 0.95 | +4.6015 |
| expanded | A_PC1_2_plus_G | +0.2187 | [-0.9215, +1.5460] | 0.65 | +0.0724 |

## Single-partition paired descriptive intervals

These bootstrap intervals resample the 211 or fewer held-out participant-level squared-error differences for one prespecified CV partition. They do not refit models or adjust for multiple testing/survey design.

| Adjustment | Weighting | Comparison | ΔMSE | Descriptive 95% interval |
|---|---|---|---:|---|
| minimal | unweighted | prior_CESD → HbA1c | +1.1268 | [-0.7370, +3.1491] |
| minimal | attrition_IPW | prior_CESD → HbA1c | +1.0227 | [-0.6122, +2.7265] |
| minimal | unweighted | prior_CESD → hematology_raw | +0.5811 | [-1.5194, +2.9046] |
| minimal | attrition_IPW | prior_CESD → hematology_raw | +1.7783 | [-1.5088, +5.5985] |
| minimal | unweighted | prior_CESD → raw_X4 | +1.5828 | [-1.3719, +5.0186] |
| minimal | attrition_IPW | prior_CESD → raw_X4 | +2.6015 | [-1.0435, +6.4466] |
| minimal | unweighted | HbA1c → raw_X4 | +0.4560 | [-1.5174, +2.5052] |
| minimal | attrition_IPW | HbA1c → raw_X4 | +1.5788 | [-1.4067, +4.7198] |
| minimal | unweighted | prior_CESD → PC1_2 | +0.2324 | [-1.9231, +2.5907] |
| minimal | attrition_IPW | prior_CESD → PC1_2 | +1.5518 | [-1.9017, +5.4788] |
| minimal | unweighted | prior_CESD → PC1_2_3_4 | +1.5041 | [-1.8176, +5.1915] |
| minimal | attrition_IPW | prior_CESD → PC1_2_3_4 | +2.5868 | [-1.7548, +6.9366] |
| minimal | unweighted | prior_CESD → A_PC1_2_plus_G | +1.2858 | [-1.5048, +4.5180] |
| minimal | attrition_IPW | prior_CESD → A_PC1_2_plus_G | +2.2464 | [-1.1840, +6.0399] |
| expanded | unweighted | prior_CESD → HbA1c | +0.6760 | [-0.8506, +2.3090] |
| expanded | attrition_IPW | prior_CESD → HbA1c | +2.2258 | [-0.6881, +6.1575] |
| expanded | unweighted | prior_CESD → hematology_raw | -0.5356 | [-2.1571, +1.0236] |
| expanded | attrition_IPW | prior_CESD → hematology_raw | -0.9203 | [-2.5357, +0.7393] |
| expanded | unweighted | prior_CESD → raw_X4 | +0.0930 | [-2.1801, +2.7875] |
| expanded | attrition_IPW | prior_CESD → raw_X4 | +0.7895 | [-2.3095, +3.7412] |
| expanded | unweighted | HbA1c → raw_X4 | -0.5830 | [-2.0756, +0.9418] |
| expanded | attrition_IPW | HbA1c → raw_X4 | -1.4363 | [-3.4147, +0.6056] |
| expanded | unweighted | prior_CESD → PC1_2 | -1.2174 | [-2.8464, +0.3081] |
| expanded | attrition_IPW | prior_CESD → PC1_2 | -4.2729 | [-10.0538, +0.0886] |
| expanded | unweighted | prior_CESD → PC1_2_3_4 | +0.3897 | [-2.6180, +3.4350] |
| expanded | attrition_IPW | prior_CESD → PC1_2_3_4 | +5.0721 | [-2.6385, +16.3812] |
| expanded | unweighted | prior_CESD → A_PC1_2_plus_G | -0.4635 | [-2.5162, +1.8216] |
| expanded | attrition_IPW | prior_CESD → A_PC1_2_plus_G | -2.4444 | [-6.8274, +1.1115] |

## Attrition

- Dated-pair eligibility: 211/353 = 0.598.
- Cross-fitted selection propensity range: [0.002, 0.883]; clipped at 0.05–0.95 for evaluation.
- Attrition-weight effective sample size: 138.6; maximum weight: 11.955; propensity clips: 1.
- Selection propensity depends on recorded baseline measures only; inverse-probability weighting does not eliminate unmeasured attrition bias.

## Prespecified robustness and interpretation

- Exclusion sensitivity: HbA1c [4,10] and Hb [8,18], retained N=211; see extreme_value_sensitivity CSV for supplementary split results.
- Expanded baseline adjusts for BMI, waist, baseline systolic blood pressure, medication count, chronic-condition count, ; missing values imputed using training folds only.
- Compare full physiology against both baseline-only and HbA1c-only models; raw four-biomarker and four-PC representations contain the same linear information.
- A positive held-out improvement is predictive association, not causality or demonstration of physiological mechanisms.
- Case-bootstrap intervals are descriptive; repetitions with different random CV splits are dependent and are not independent replications.
- BMI, blood pressure, RBC assay scale, CES-D score coding, laboratory harmonization, survey weights, and sampling/attrition mechanisms remain unverified.
- No external prospective test set exists in this audit; validation takes place within the matched MIDJA cohort.
- Strong support would require stable positive gains beyond both baseline and simple markers under expanded covariates, plausible exclusion, attrition sensitivity, and subsequent truly independent replication.
