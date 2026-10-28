# Script 109 — Clinical utility and reliability comparison

## Question
Does the original frozen 7D H/G representation improve PHQ-9>=10 screening, or flag missed cases better than standard biomarker and shift detectors?

## Key held-out point results
```
            period     method    n    auroc    auprc    brier  sensitivity  specificity
    test_2009-2018 Covariates 9893 0.716606 0.175077 0.067643     0.781337     0.522366
validate_2021-2023 Covariates 2247 0.618928 0.165863 0.109078     0.626962     0.603306
    test_2009-2018   Hb_HbA1c 9893 0.719432 0.174318 0.067621     0.777088     0.538722
validate_2021-2023   Hb_HbA1c 2247 0.619289 0.168689 0.108876     0.595929     0.598992
    test_2009-2018      RawX7 9893 0.723591 0.173607 0.067737     0.836113     0.465313
validate_2021-2023      RawX7 2247 0.616862 0.170222 0.108697     0.686331     0.518620
    test_2009-2018       PCA7 9893 0.723591 0.173607 0.067737     0.836113     0.465313
validate_2021-2023       PCA7 2247 0.616862 0.170222 0.108697     0.686331     0.518620
    test_2009-2018   FrozenZ7 9893 0.722606 0.172421 0.067798     0.845620     0.459383
validate_2021-2023   FrozenZ7 2247 0.614207 0.170446 0.108811     0.697122     0.504579
    test_2009-2018       PCA4 9893 0.720493 0.173432 0.067710     0.811938     0.500041
validate_2021-2023       PCA4 2247 0.617068 0.173978 0.108454     0.652234     0.569351
    test_2009-2018   FrozenZ4 9893 0.719733 0.175826 0.067598     0.808799     0.496672
validate_2021-2023   FrozenZ4 2247 0.618324 0.172280 0.108490     0.644490     0.575142
    test_2009-2018      JIVE5 9893 0.723350 0.173260 0.067754     0.828178     0.483737
validate_2021-2023      JIVE5 2247 0.618705 0.170674 0.108639     0.664365     0.541049
    test_2009-2018     AJIVE5 9893 0.719443 0.173140 0.067758     0.809681     0.508915
validate_2021-2023     AJIVE5 2247 0.611902 0.166615 0.109202     0.627393     0.554394
```

## Matched 20% weighted review capacity among predicted low-risk participants
```
            period                gate  missed_case_capture_at_20pct  weighted_flag_fraction  frozen_threshold_flag_fraction  frozen_threshold_missed_capture
    test_2009-2018     Z_group_anomaly                      0.309196                0.200201                        0.147030                         0.212596
    test_2009-2018    Raw_max_marker_z                      0.250204                0.200114                        0.137940                         0.204505
    test_2009-2018     Raw_Mahalanobis                      0.240403                0.200001                        0.156493                         0.199077
    test_2009-2018      PCA_max_axis_z                      0.306106                0.200265                        0.144627                         0.155380
    test_2009-2018 Near_risk_threshold                      0.253486                0.200243                        0.274836                         0.402573
validate_2021-2023     Z_group_anomaly                      0.241232                0.201195                        0.169892                         0.213478
validate_2021-2023    Raw_max_marker_z                      0.337943                0.201294                        0.147355                         0.229419
validate_2021-2023     Raw_Mahalanobis                      0.304541                0.200164                        0.182117                         0.304541
validate_2021-2023      PCA_max_axis_z                      0.270415                0.200036                        0.156018                         0.247709
validate_2021-2023 Near_risk_threshold                      0.082220                0.200291                        0.218984                         0.110173
```

## Direct criterion comparisons (positive values favor our method)
```
            period  delta_auc_z_minus_raw  delta_brier_raw_minus_z  Z_gate_miss_capture  best_simple_gate_miss_capture  gate_advantage  Z_gate_beats_all_controls  prediction_better_auc_and_brier
    test_2009-2018              -0.000984                -0.000061             0.309196                       0.306106        0.003089                       True                            False
validate_2021-2023              -0.002655                -0.000114             0.241232                       0.337943       -0.096711                      False                            False
```

## Paired case-bootstrap intervals (descriptive, NOT design-based)
```
            period  delta_brier_raw_minus_z  delta_brier_low  delta_brier_high  delta_auroc_z_minus_raw  delta_auroc_low  delta_auroc_high  bootstrap_reps
    test_2009-2018                -0.000070        -0.000212          0.000082                -0.001097        -0.004186          0.002299             199
validate_2021-2023                -0.000114        -0.000395          0.000201                -0.002571        -0.007507          0.001648             199
```

## What would constitute an application advantage?
- Screening: a convincing held-out improvement over RawX7 and PCA7 in AUROC and Brier, robust to uncertainty; not merely improvement over HbA1c.
- Reliability: at equal review capacity, Z group anomaly should capture more false negatives than raw marker, Mahalanobis, PCA and threshold-proximity controls in both later eras.
- Values are descriptive and exploratory; the earlier cohorts have been repeatedly inspected. This cannot certify clinical utility without independent testing.

## Interpretation boundaries
- The original Z7 is a reversible transformation of RawX7. It cannot create new biomarkers or information.
- Full 7D PCA7 and RawX7 are L2-rotation-equivalent when standardized consistently. Small Z7 differences can be caused by regularization and feature scaling.
- The structural gate is outcome-blind. Its usefulness is assessed retrospectively against missed high-PHQ cases, not validated clinical events.
- The 20% matched-budget rule uses current-cohort distributions, so it is a diagnostic comparison, not an externally calibrated review threshold.
- Original PHQ-9 measurement and cohort selection are observational and cannot explain individual causal contributors.
- Case bootstrap uncertainty does not account for PSU/stratum design, nonresponse or repeated exploratory testing.
- JIVE/AJIVE, if available, use their native discovery fits followed by train-only score adapters; projected scores are NOT native out-of-sample outputs.
- Future fundus/vitals modalities remain untested: these results only assess current blood-based application feasibility.
