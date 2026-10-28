# Script 107 — Diabetes status sensitivity (descriptive)

Doctor-reported diabetes status; **not** Type 1/Type 2 identification.
Full 2005–08 training versus discovery training excluding reported diabetes.

## Complete-case diabetes groups

```text
           period                group    n  weighted_fraction_within_period  diagnosed_insulin_yes_n  diagnosed_oral_pills_yes_n  reported_prediabetes_DIQ160_yes_n
discovery_2005_08                  all 3815                            1.000                      121                         327                                152
discovery_2005_08 no_reported_diabetes 3307                            0.907                        0                           0                                151
discovery_2005_08    reported_diabetes  434                            0.076                      121                         327                                  0
discovery_2005_08 borderline_in_DIq010   70                            0.016                        0                           0                                  0
discovery_2005_08              unknown    4                            0.001                        0                           0                                  1
     test_2009_18                  all 9893                            1.000                      397                         974                                693
     test_2009_18 no_reported_diabetes 8276                            0.876                        0                           0                                693
     test_2009_18    reported_diabetes 1352                            0.100                      397                         974                                  0
     test_2009_18 borderline_in_DIq010  261                            0.024                        0                           0                                  0
     test_2009_18              unknown    4                            0.000                        0                           0                                  0
   recent_2021_23                  all 2247                            1.000                       82                         228                                293
   recent_2021_23 no_reported_diabetes 1877                            0.865                        0                           0                                293
   recent_2021_23    reported_diabetes  294                            0.102                       82                         228                                  0
   recent_2021_23 borderline_in_DIq010   76                            0.032                        0                           0                                  0
   recent_2021_23              unknown    0                            0.000                        0                           0                                  0
```

## Frozen PHQ-total outcome transfer

Positive delta MSE indicates incremental value beyond covariates.

```text
                 training         period                group     method    n  weighted_r2  delta_mse_vs_covariates
            all_discovery   test_2009_18                  all Covariates 9893       0.0770                   0.0000
            all_discovery   test_2009_18 no_reported_diabetes Covariates 8276       0.0732                   0.0000
            all_discovery   test_2009_18    reported_diabetes Covariates 1352       0.0793                   0.0000
            all_discovery recent_2021_23                  all Covariates 2247      -0.0379                   0.0000
            all_discovery recent_2021_23 no_reported_diabetes Covariates 1877      -0.0408                   0.0000
            all_discovery recent_2021_23    reported_diabetes Covariates  294      -0.0041                   0.0000
            all_discovery   test_2009_18                  all      RawX7 9893       0.0810                   0.0654
            all_discovery   test_2009_18 no_reported_diabetes      RawX7 8276       0.0724                  -0.0134
            all_discovery   test_2009_18    reported_diabetes      RawX7 1352       0.1080                   0.5831
            all_discovery recent_2021_23                  all      RawX7 2247      -0.0050                   0.6398
            all_discovery recent_2021_23 no_reported_diabetes      RawX7 1877      -0.0151                   0.4817
            all_discovery recent_2021_23    reported_diabetes      RawX7  294       0.0526                   1.3668
            all_discovery   test_2009_18                  all   FrozenZ7 9893       0.0806                   0.0587
            all_discovery   test_2009_18 no_reported_diabetes   FrozenZ7 8276       0.0720                  -0.0187
            all_discovery   test_2009_18    reported_diabetes   FrozenZ7 1352       0.1073                   0.5678
            all_discovery recent_2021_23                  all   FrozenZ7 2247      -0.0056                   0.6298
            all_discovery recent_2021_23 no_reported_diabetes   FrozenZ7 1877      -0.0156                   0.4725
            all_discovery recent_2021_23    reported_diabetes   FrozenZ7  294       0.0525                   1.3662
without_reported_diabetes   test_2009_18                  all Covariates 9893       0.0705                   0.0000
without_reported_diabetes   test_2009_18 no_reported_diabetes Covariates 8276       0.0693                   0.0000
without_reported_diabetes   test_2009_18    reported_diabetes Covariates 1352       0.0575                   0.0000
without_reported_diabetes recent_2021_23                  all Covariates 2247      -0.0529                   0.0000
without_reported_diabetes recent_2021_23 no_reported_diabetes Covariates 1877      -0.0531                   0.0000
without_reported_diabetes recent_2021_23    reported_diabetes Covariates  294      -0.0298                   0.0000
without_reported_diabetes   test_2009_18                  all      RawX7 9893       0.0785                   0.1294
without_reported_diabetes   test_2009_18 no_reported_diabetes      RawX7 8276       0.0713                   0.0302
without_reported_diabetes   test_2009_18    reported_diabetes      RawX7 1352       0.0988                   0.8405
without_reported_diabetes recent_2021_23                  all      RawX7 2247      -0.0142                   0.7538
without_reported_diabetes recent_2021_23 no_reported_diabetes      RawX7 1877      -0.0248                   0.5281
without_reported_diabetes recent_2021_23    reported_diabetes      RawX7  294       0.0494                   1.9101
without_reported_diabetes   test_2009_18                  all   FrozenZ7 9893       0.0782                   0.1250
without_reported_diabetes   test_2009_18 no_reported_diabetes   FrozenZ7 8276       0.0710                   0.0266
without_reported_diabetes   test_2009_18    reported_diabetes   FrozenZ7 1352       0.0984                   0.8320
without_reported_diabetes recent_2021_23                  all   FrozenZ7 2247      -0.0146                   0.7458
without_reported_diabetes recent_2021_23 no_reported_diabetes   FrozenZ7 1877      -0.0252                   0.5211
without_reported_diabetes recent_2021_23    reported_diabetes   FrozenZ7  294       0.0494                   1.9097
```

## Important boundaries

- **Never** describe DIQ010/DIQ050 as confirmed Type 1 vs Type 2.
- Comparing subgroups with different outcome variance can change R² without changing biology.
- The key sensitivity is whether excluding diagnosed diabetes from discovery fitting materially changes the frozen **no-reported-diabetes** evaluation.
- This experiment is descriptive and exploratory; it does not establish diabetes as the source of any component, or formal effect modification.
- Earlier NHANES periods and the original full-sample representation have already been used; this is not an untouched confirmation.
