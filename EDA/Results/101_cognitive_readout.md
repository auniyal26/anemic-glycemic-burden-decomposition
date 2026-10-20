# Script 101 — NHANES A/G decomposition of measured cognitive outcomes

Training: NHANES 2011–2012, independently evaluated: 2013–2014.
Cognition: DSST, animal fluency, CERAD delayed recall and three-trial learning sum; age ≥60.
Physiology: real Script 73 outcome-independent frozen NHANES seven-coordinate transformation.
All models for each outcome use identical matched fully observed participants.
All hyperparameters and standardization fitted within 2011–12 only.

# Cohort availability
|   cycle | outcome               |   age60_eligible_in_source |   cognitive_observed |   fully_observed_X_and_Z |   common_complete |
|--------:|:----------------------|---------------------------:|---------------------:|-------------------------:|------------------:|
|    1112 | DSST_processing_speed |                        856 |                  720 |                      746 |               630 |
|    1112 | Animal_verbal_fluency |                        856 |                  733 |                      746 |               643 |
|    1112 | CERAD_delayed_memory  |                        856 |                  731 |                      746 |               640 |
|    1112 | CERAD_learning_sum    |                        856 |                  732 |                      746 |               641 |
|    1314 | DSST_processing_speed |                        877 |                  794 |                      812 |               736 |
|    1314 | Animal_verbal_fluency |                        877 |                  824 |                      812 |               760 |
|    1314 | CERAD_delayed_memory  |                        877 |                  832 |                      812 |               766 |
|    1314 | CERAD_learning_sum    |                        877 |                  832 |                      812 |               766 |

# Out-of-period performance
| outcome               | specification    |   n_train |   n_test |   selected_alpha |   test_MSE |   test_R2 |   test_bias |   delta_MSE_vs_covariates |
|:----------------------|:-----------------|----------:|---------:|-----------------:|-----------:|----------:|------------:|--------------------------:|
| DSST_processing_speed | covariates       |       630 |      736 |          1.00000 |  198.33943 |   0.29863 |     2.02539 |                   0.00000 |
| DSST_processing_speed | A_only           |       630 |      736 |         10.00000 |  186.02969 |   0.34216 |     0.23864 |                  12.30974 |
| DSST_processing_speed | G_only           |       630 |      736 |         10.00000 |  195.30588 |   0.30936 |     1.85874 |                   3.03355 |
| DSST_processing_speed | raw_A_plus_G     |       630 |      736 |         10.00000 |  183.43161 |   0.35135 |     0.21951 |                  14.90782 |
| DSST_processing_speed | Z_shared_private |       630 |      736 |         10.00000 |  183.75353 |   0.35021 |     0.15817 |                  14.58590 |
| Animal_verbal_fluency | covariates       |       643 |      760 |         10.00000 |   25.25827 |   0.15937 |     0.16832 |                   0.00000 |
| Animal_verbal_fluency | A_only           |       643 |      760 |         10.00000 |   25.37019 |   0.15564 |    -0.14004 |                  -0.11192 |
| Animal_verbal_fluency | G_only           |       643 |      760 |        100.00000 |   25.17901 |   0.16200 |     0.23574 |                   0.07926 |
| Animal_verbal_fluency | raw_A_plus_G     |       643 |      760 |        100.00000 |   25.32983 |   0.15698 |    -0.06434 |                  -0.07156 |
| Animal_verbal_fluency | Z_shared_private |       643 |      760 |        100.00000 |   25.26026 |   0.15930 |    -0.05305 |                  -0.00199 |
| CERAD_delayed_memory  | covariates       |       640 |      766 |         10.00000 |    4.56029 |   0.09793 |    -0.51609 |                   0.00000 |
| CERAD_delayed_memory  | A_only           |       640 |      766 |        100.00000 |    4.56151 |   0.09768 |    -0.53949 |                  -0.00123 |
| CERAD_delayed_memory  | G_only           |       640 |      766 |         10.00000 |    4.67727 |   0.07479 |    -0.52695 |                  -0.11698 |
| CERAD_delayed_memory  | raw_A_plus_G     |       640 |      766 |        100.00000 |    4.63129 |   0.08388 |    -0.53928 |                  -0.07101 |
| CERAD_delayed_memory  | Z_shared_private |       640 |      766 |        100.00000 |    4.68427 |   0.07340 |    -0.52599 |                  -0.12398 |
| CERAD_learning_sum    | covariates       |       641 |      766 |          1.00000 |   17.93463 |   0.07352 |    -1.41962 |                   0.00000 |
| CERAD_learning_sum    | A_only           |       641 |      766 |         10.00000 |   18.48079 |   0.04530 |    -1.61343 |                  -0.54616 |
| CERAD_learning_sum    | G_only           |       641 |      766 |         10.00000 |   18.19380 |   0.06013 |    -1.43055 |                  -0.25917 |
| CERAD_learning_sum    | raw_A_plus_G     |       641 |      766 |         10.00000 |   18.75000 |   0.03140 |    -1.62114 |                  -0.81537 |
| CERAD_learning_sum    | Z_shared_private |       641 |      766 |        100.00000 |   18.61927 |   0.03815 |    -1.61378 |                  -0.68464 |

# Predictive A versus G allocation
| outcome               | block   |   shapley_delta_MSE |   total_rawX_gain | note                                                                |
|:----------------------|:--------|--------------------:|------------------:|:--------------------------------------------------------------------|
| DSST_processing_speed | A       |            12.09200 |          14.90782 | predictive allocation, not individual impairment or causal fraction |
| DSST_processing_speed | G       |             2.81582 |          14.90782 | predictive allocation, not individual impairment or causal fraction |
| Animal_verbal_fluency | A       |            -0.13137 |          -0.07156 | predictive allocation, not individual impairment or causal fraction |
| Animal_verbal_fluency | G       |             0.05981 |          -0.07156 | predictive allocation, not individual impairment or causal fraction |
| CERAD_delayed_memory  | A       |             0.02237 |          -0.07101 | predictive allocation, not individual impairment or causal fraction |
| CERAD_delayed_memory  | G       |            -0.09338 |          -0.07101 | predictive allocation, not individual impairment or causal fraction |
| CERAD_learning_sum    | A       |            -0.55118 |          -0.81537 | predictive allocation, not individual impairment or causal fraction |
| CERAD_learning_sum    | G       |            -0.26419 |          -0.81537 | predictive allocation, not individual impairment or causal fraction |

# Frozen latent block attribution
| outcome               | Z_block    |   mean_absolute_predicted_deviation |   rms_predicted_deviation |   mean_signed_predicted_deviation |   n |
|:----------------------|:-----------|------------------------------------:|--------------------------:|----------------------------------:|----:|
| DSST_processing_speed | Shared     |                             2.47864 |                   3.47122 |                          -0.86893 | 736 |
| DSST_processing_speed | Discordant |                             1.00019 |                   1.31809 |                           0.30041 | 736 |
| DSST_processing_speed | A_private  |                             1.76690 |                   2.38757 |                          -1.23904 | 736 |
| DSST_processing_speed | G_private  |                             0.64947 |                   1.06880 |                          -0.03869 | 736 |
| Animal_verbal_fluency | Shared     |                             0.39384 |                   0.58160 |                          -0.13273 | 760 |
| Animal_verbal_fluency | Discordant |                             0.35092 |                   0.47590 |                           0.07650 | 760 |
| Animal_verbal_fluency | A_private  |                             0.40494 |                   0.59129 |                          -0.16568 | 760 |
| Animal_verbal_fluency | G_private  |                             0.14911 |                   0.24419 |                          -0.00670 | 760 |
| CERAD_delayed_memory  | Shared     |                             0.08584 |                   0.11026 |                          -0.02067 | 766 |
| CERAD_delayed_memory  | Discordant |                             0.13119 |                   0.16671 |                           0.07142 | 766 |
| CERAD_delayed_memory  | A_private  |                             0.18134 |                   0.26851 |                          -0.06001 | 766 |
| CERAD_delayed_memory  | G_private  |                             0.15388 |                   0.25152 |                          -0.00607 | 766 |
| CERAD_learning_sum    | Shared     |                             0.22687 |                   0.33305 |                          -0.08190 | 766 |
| CERAD_learning_sum    | Discordant |                             0.13133 |                   0.16652 |                           0.07047 | 766 |
| CERAD_learning_sum    | A_private  |                             0.31721 |                   0.45177 |                          -0.18527 | 766 |
| CERAD_learning_sum    | G_private  |                             0.23646 |                   0.38656 |                          -0.00946 | 766 |

# Descriptive participant-bootstrap intervals
| outcome               | reference   | candidate        |   delta_MSE |   descriptive_ci_low |   descriptive_ci_high |
|:----------------------|:------------|:-----------------|------------:|---------------------:|----------------------:|
| DSST_processing_speed | covariates  | A_only           |    12.30974 |              3.00108 |              22.44849 |
| DSST_processing_speed | covariates  | G_only           |     3.03355 |             -1.92546 |               8.70928 |
| DSST_processing_speed | covariates  | raw_A_plus_G     |    14.90782 |              4.74892 |              26.06669 |
| DSST_processing_speed | covariates  | Z_shared_private |    14.58590 |              4.05791 |              25.84385 |
| Animal_verbal_fluency | covariates  | A_only           |    -0.11192 |             -0.64558 |               0.43236 |
| Animal_verbal_fluency | covariates  | G_only           |     0.07926 |             -0.45671 |               0.63848 |
| Animal_verbal_fluency | covariates  | raw_A_plus_G     |    -0.07156 |             -0.83135 |               0.66207 |
| Animal_verbal_fluency | covariates  | Z_shared_private |    -0.00199 |             -0.75958 |               0.75851 |
| CERAD_delayed_memory  | covariates  | A_only           |    -0.00123 |             -0.11150 |               0.11050 |
| CERAD_delayed_memory  | covariates  | G_only           |    -0.11698 |             -0.24177 |              -0.01114 |
| CERAD_delayed_memory  | covariates  | raw_A_plus_G     |    -0.07101 |             -0.20358 |               0.06696 |
| CERAD_delayed_memory  | covariates  | Z_shared_private |    -0.12398 |             -0.28843 |               0.02391 |
| CERAD_learning_sum    | covariates  | A_only           |    -0.54616 |             -0.92566 |              -0.20396 |
| CERAD_learning_sum    | covariates  | G_only           |    -0.25917 |             -0.59606 |               0.09331 |
| CERAD_learning_sum    | covariates  | raw_A_plus_G     |    -0.81537 |             -1.35000 |              -0.30755 |
| CERAD_learning_sum    | covariates  | Z_shared_private |    -0.68464 |             -1.29314 |              -0.07355 |

# Limits
- All A/G allocation figures are about a *fitted prediction*, not causal shares of cognitive impairment.
- Shared/discordant coordinates mix A and G, so cannot be attributed uniquely to either physiology block.
- Full-rank Z preserves seven-dimensional information; it cannot create additional information beyond raw X.
- Outcomes are contemporaneous cognitive test performances, not observed cognitive decline.
- 2011–12 and 2013–14 contain different NHANES participants, not individual follow-up.
- Fasting glucose and insulin restrict sample size; selection and test nonresponse may bias associations.
- SURVEY_WT is inherited from Script 73; verify its fasting-weight construction before design-based inference.
- This analysis estimates weighted point metrics but not NHANES strata/PSU-adjusted uncertainty.
- Bootstrap confidence bands resample people, not the complex survey design, and are descriptive.
- Clinical assay harmonization and missingness assumptions require independent verification.
