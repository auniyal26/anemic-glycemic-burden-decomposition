# Script 11 — MIDJA longitudinal validation and failure audit

## Study

- HbA1c panel: raw; MIDJA1 complete N=353; MIDJA2 complete N=303.
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
| minimal | HbA1c | +0.9175 | [+0.1621, +1.3180] | 1.00 | +0.9731 |
| minimal | Hb | +0.1330 | [-0.9219, +0.5524] | 0.80 | +0.1744 |
| minimal | hematology_raw | +1.2155 | [+0.2581, +2.0350] | 1.00 | +1.8711 |
| minimal | raw_X4 | +1.8332 | [-0.1385, +2.6194] | 0.95 | +2.4449 |
| minimal | PC1_2 | +0.8290 | [+0.1996, +1.5739] | 1.00 | +1.6639 |
| minimal | PC1_2_3 | +1.4068 | [-0.5733, +2.1498] | 0.95 | +2.0207 |
| minimal | PC1_2_3_4 | +1.9358 | [+0.1095, +2.8406] | 1.00 | +2.5373 |
| minimal | A_PC1_2_plus_G | +1.3988 | [-0.5776, +2.1383] | 0.95 | +2.0164 |
| expanded | prior_CESD | +0.0000 | [+0.0000, +0.0000] | 0.00 | +0.0000 |
| expanded | HbA1c | +0.4991 | [-0.1830, +1.2148] | 0.90 | +3.3648 |
| expanded | Hb | -0.2648 | [-1.1446, +0.1825] | 0.20 | -1.8412 |
| expanded | hematology_raw | +0.3543 | [-0.5356, +1.6637] | 0.70 | -1.4004 |
| expanded | raw_X4 | +0.7612 | [-0.4017, +2.1165] | 0.90 | +1.9340 |
| expanded | PC1_2 | -0.3397 | [-1.2044, +0.9866] | 0.20 | -3.6297 |
| expanded | PC1_2_3 | +0.1565 | [-0.9480, +1.4695] | 0.60 | +0.0100 |
| expanded | PC1_2_3_4 | +1.1410 | [-0.0903, +2.5396] | 0.95 | +4.4982 |
| expanded | A_PC1_2_plus_G | +0.1451 | [-0.9544, +1.4602] | 0.60 | -0.0405 |

## Single-partition paired descriptive intervals

These bootstrap intervals resample the 211 or fewer held-out participant-level squared-error differences for one prespecified CV partition. They do not refit models or adjust for multiple testing/survey design.

| Adjustment | Weighting | Comparison | ΔMSE | Descriptive 95% interval |
|---|---|---|---:|---|
| minimal | unweighted | prior_CESD → HbA1c | +1.0339 | [-0.7431, +2.9641] |
| minimal | attrition_IPW | prior_CESD → HbA1c | +1.0592 | [-0.5555, +2.7006] |
| minimal | unweighted | prior_CESD → hematology_raw | +0.5811 | [-1.5194, +2.9046] |
| minimal | attrition_IPW | prior_CESD → hematology_raw | +1.7752 | [-1.5102, +5.6000] |
| minimal | unweighted | prior_CESD → raw_X4 | +1.4985 | [-1.4031, +4.9046] |
| minimal | attrition_IPW | prior_CESD → raw_X4 | +2.6077 | [-1.0861, +6.5516] |
| minimal | unweighted | HbA1c → raw_X4 | +0.4646 | [-1.5156, +2.5219] |
| minimal | attrition_IPW | HbA1c → raw_X4 | +1.5485 | [-1.4084, +4.6578] |
| minimal | unweighted | prior_CESD → PC1_2 | +0.2485 | [-1.9103, +2.6006] |
| minimal | attrition_IPW | prior_CESD → PC1_2 | +1.5803 | [-1.8818, +5.5174] |
| minimal | unweighted | prior_CESD → PC1_2_3_4 | +1.4165 | [-1.8869, +5.0255] |
| minimal | attrition_IPW | prior_CESD → PC1_2_3_4 | +2.5748 | [-1.8062, +6.9544] |
| minimal | unweighted | prior_CESD → A_PC1_2_plus_G | +1.2008 | [-1.5055, +4.3451] |
| minimal | attrition_IPW | prior_CESD → A_PC1_2_plus_G | +2.2615 | [-1.1972, +6.1634] |
| expanded | unweighted | prior_CESD → HbA1c | +0.5713 | [-0.8624, +2.1194] |
| expanded | attrition_IPW | prior_CESD → HbA1c | +2.0286 | [-0.7299, +5.6966] |
| expanded | unweighted | prior_CESD → hematology_raw | -0.5356 | [-2.1571, +1.0236] |
| expanded | attrition_IPW | prior_CESD → hematology_raw | -0.9251 | [-2.5357, +0.7304] |
| expanded | unweighted | prior_CESD → raw_X4 | -0.0043 | [-2.2600, +2.6291] |
| expanded | attrition_IPW | prior_CESD → raw_X4 | +0.5825 | [-2.3626, +3.3094] |
| expanded | unweighted | HbA1c → raw_X4 | -0.5756 | [-2.0839, +0.9700] |
| expanded | attrition_IPW | HbA1c → raw_X4 | -1.4462 | [-3.4354, +0.6049] |
| expanded | unweighted | prior_CESD → PC1_2 | -1.2044 | [-2.8453, +0.3191] |
| expanded | attrition_IPW | prior_CESD → PC1_2 | -4.2580 | [-10.0286, +0.0986] |
| expanded | unweighted | prior_CESD → PC1_2_3_4 | +0.2954 | [-2.6939, +3.3056] |
| expanded | attrition_IPW | prior_CESD → PC1_2_3_4 | +4.9149 | [-2.7116, +16.0188] |
| expanded | unweighted | prior_CESD → A_PC1_2_plus_G | -0.5650 | [-2.5698, +1.6639] |
| expanded | attrition_IPW | prior_CESD → A_PC1_2_plus_G | -2.6886 | [-7.3590, +0.9332] |

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
