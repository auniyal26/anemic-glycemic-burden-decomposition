# Script123 — Biological specificity and leakage audit

## Completion

```text
{
  "completed": true,
  "available_anchor_cycle_samples": 11,
  "expected_anchor_cycle_samples": 11,
  "planned_primary_tests": 44,
  "available_primary_tests": 44,
  "holm_significant_primary_tests": 21,
  "provenance_failures": 0,
  "provenance_reviews": 0,
  "complete_anchor_coverage": true,
  "frozen_paired_contrasts": 84,
  "rotation_comparisons": 56,
  "fold_cv_models": 9,
  "causal_mechanisms_proven": false,
  "independent_clinical_validation": false
}
```

## Provenance

```text
                                                 test     status                                                                                       evidence
                   Script72_frozen_score_reproduction       PASS                                                                         1.7763568394002505e-15
                   Script73_frozen_score_reproduction       PASS                                                                         1.7763568394002505e-15
                                discovery_sample_size       PASS                                                                                           4088
         discovery_canonical_correlation_reproduction       PASS                                                                         2.7755575615628914e-16
                            discovery_subspace_Shared       PASS                                                                                            1.0
                        discovery_subspace_Discordant       PASS                                                                                            1.0
                         discovery_subspace_H_private       PASS                                                                                            1.0
                         discovery_subspace_G_private       PASS                                                                                            1.0
                        participant_overlap_2009-2018       PASS                                                                                              0
                        participant_overlap_2021-2023       PASS                                                                                              0
hash_72_shared_private_decomposition_and_phenotype.py       PASS                               29a5b6b7604e08dda7bb9d30edcacce0b4133774eae0e5c3cb3439f45a3fbdd8
                     hash_103_jive_ajive_benchmark.py       PASS                               5ff4d5329f2a7b8e2061a796ffd96e9a829485cef5d53abd983ac22c849b339d
           hash_104_representation_validation_gate.py       PASS                               6f5f4076bbe9dcbd9dcdf9824d88828f665f57b70e851fad0ac8c67956075f05
                   hash_106_survey_ferritin_anchor.py       PASS                               94cc7de9c1f1686999c68c20048efe244f4a5532b782ae821474fd4d3df401fb
                        hash_112_finish_benchmarks.py       PASS                               613ebe67f6d94ccd5b09859b5f5ea4b23cd3f34d9be455fa281a4ec14a39a501
                       hash_114_structure_recovery.py       PASS                               730029aad69209a2471397ba754cfffbcbc72f84b096b2c14c45a9c8102f4d06
             hash_118_biomedical_utility_benchmark.py       PASS                               c072c0e86ab2fbdfc0d434488913fd8a1109f384462cf7f86b276059b898a74c
                                 direct_outcome_input       PASS                              Independent refit uses only seven raw markers and survey weights.
            prior_outcome_informed_method_development LIMITATION Earlier exploratory feature and method selection cannot be made outcome-blind retrospectively.
                        untouched_external_population LIMITATION                                                   All NHANES periods were previously examined.
                               unmeasured_confounding UNTESTABLE                                                          Cross-sectional observational design.
                              complete_case_selection LIMITATION                                                        Inherited eligible laboratory scaffold.
                                  assay_harmonisation LIMITATION                               Historical and modern assay methods are not formally harmonised.
```

## Anchor coverage

```text
cycle   anchor                  status  eligible_scaffold_n      n  effective_n   weight  represented_psus     outcome
 0506 Ferritin               available                440.0  416.0   298.447505 WTSAF2YR              30.0 natural_log
 0506     sTfR               available                440.0  417.0   298.867640 WTSAF2YR              30.0 natural_log
 0506      CRP               available               1796.0 1688.0  1170.783671 WTSAF2YR              30.0 natural_log
 0506     OGTT               available               1796.0 1407.0   988.946803 WTSOG2YR              30.0 natural_log
 0708 Ferritin               available                555.0  513.0   350.389394 WTSAF2YR              32.0 natural_log
 0708     sTfR               available                555.0  511.0   348.674814 WTSAF2YR              32.0 natural_log
 0708      CRP               available               2292.0 2070.0  1288.271324 WTSAF2YR              32.0 natural_log
 0708     OGTT               available               2292.0 1659.0  1060.630987 WTSOG2YR              32.0 natural_log
 2123 Ferritin               available                597.0  519.0   358.494323 WTSAF2YR              30.0 natural_log
 2123     sTfR               available                597.0  519.0   358.494323 WTSAF2YR              30.0 natural_log
 2123      CRP               available               2783.0 2415.0  1519.259679 WTSAF2YR              30.0 natural_log
 2123     OGTT not_available_by_design                  NaN    NaN          NaN      NaN               NaN         NaN
```

## All primary conditional tests

```text
  anchor cycle   model      group    n  partial_R2  incremental_R2          F            p  df_num  design_df    status       p_holm
Ferritin  0506 Primary     Shared  416    0.119008        0.090101  12.103150 7.421906e-04       2         15 available 1.929696e-02
Ferritin  0506 Primary Discordant  416    0.151529        0.119120  38.779294 1.181841e-06       2         15 available 3.900075e-05
Ferritin  0506 Primary  H_private  416    0.103498        0.077003   9.961621 1.767361e-03       2         15 available 4.418402e-02
Ferritin  0506 Primary  G_private  416    0.003434        0.002298   3.202956 9.370992e-02       1         15 available 1.000000e+00
    sTfR  0506 Primary     Shared  417    0.261705        0.177753  40.746379 8.649162e-07       2         15 available 3.027207e-05
    sTfR  0506 Primary Discordant  417    0.186495        0.114959  42.310086 6.808944e-07       2         15 available 2.451220e-05
    sTfR  0506 Primary  H_private  417    0.127350        0.073180  30.897565 4.793821e-06       2         15 available 1.486085e-04
    sTfR  0506 Primary  G_private  417    0.000655        0.000329   0.058327 8.124308e-01       1         15 available 1.000000e+00
     CRP  0506 Primary     Shared 1688    0.031473        0.023146  14.552336 3.069375e-04       2         15 available 8.901186e-03
     CRP  0506 Primary Discordant 1688    0.014092        0.010181   9.624116 2.045965e-03       2         15 available 4.910316e-02
     CRP  0506 Primary  H_private 1688    0.006321        0.004531   3.366607 6.198046e-02       2         15 available 8.677264e-01
     CRP  0506 Primary  G_private 1688    0.000281        0.000200   0.422287 5.256257e-01       1         15 available 1.000000e+00
    OGTT  0506 Primary     Shared 1407    0.144450        0.098085 112.511781 9.306371e-10       2         15 available 3.908676e-08
    OGTT  0506 Primary Discordant 1407    0.064519        0.040067 117.607512 6.812932e-10       2         15 available 2.929561e-08
    OGTT  0506 Primary  H_private 1407    0.000435        0.000253   0.152334 8.600132e-01       2         15 available 1.000000e+00
    OGTT  0506 Primary  G_private 1407    0.182517        0.129704 127.701080 9.779152e-09       1         15 available 3.813869e-07
Ferritin  0708 Primary     Shared  513    0.095500        0.072066  30.232657 3.674887e-06       2         16 available 1.175964e-04
Ferritin  0708 Primary Discordant  513    0.085527        0.063836   8.183242 3.566118e-03       2         16 available 8.093262e-02
Ferritin  0708 Primary  H_private  513    0.024106        0.016860   2.583266 1.066012e-01       2         16 available 1.000000e+00
Ferritin  0708 Primary  G_private  513    0.005459        0.003747   1.902457 1.867791e-01       1         16 available 1.000000e+00
    sTfR  0708 Primary     Shared  511    0.170798        0.096302  89.951285 1.979880e-09       2         16 available 8.117507e-08
    sTfR  0708 Primary Discordant  511    0.178232        0.101403  67.428128 1.601228e-08       2         16 available 6.084667e-07
    sTfR  0708 Primary  H_private  511    0.035624        0.017271   5.202263 1.817744e-02       2         16 available 3.138550e-01
    sTfR  0708 Primary  G_private  511    0.000803        0.000376   0.127667 7.255328e-01       1         16 available 1.000000e+00
     CRP  0708 Primary     Shared 2070    0.011435        0.008609   8.210280 3.518810e-03       2         16 available 8.093262e-02
     CRP  0708 Primary Discordant 2070    0.003538        0.002643   5.271131 1.743639e-02       2         16 available 3.138550e-01
     CRP  0708 Primary  H_private 2070    0.003114        0.002325   4.229131 3.353939e-02       2         16 available 5.030908e-01
     CRP  0708 Primary  G_private 2070    0.000009        0.000007   0.042058 8.400947e-01       1         16 available 1.000000e+00
    OGTT  0708 Primary     Shared 1659    0.130013        0.091615  46.909085 2.030328e-07       2         16 available 7.512214e-06
    OGTT  0708 Primary Discordant 1659    0.083529        0.055874  87.893152 2.346492e-09       2         16 available 9.385970e-08
    OGTT  0708 Primary  H_private 1659    0.000387        0.000237   0.421559 6.631016e-01       2         16 available 1.000000e+00
    OGTT  0708 Primary  G_private 1659    0.134479        0.095251 255.126508 2.967489e-11       1         16 available 1.305695e-09
Ferritin  2123 Primary     Shared  519    0.110074        0.087693  13.836832 3.930836e-04       2         15 available 1.100634e-02
Ferritin  2123 Primary Discordant  519    0.056844        0.042731   6.250450 1.060676e-02       2         15 available 2.227419e-01
Ferritin  2123 Primary  H_private  519    0.017387        0.012545   4.653091 2.677989e-02       2         15 available 4.284783e-01
Ferritin  2123 Primary  G_private  519    0.001903        0.001351   0.877784 3.636502e-01       1         15 available 1.000000e+00
    sTfR  2123 Primary     Shared  519    0.168184        0.083711  39.958649 9.785752e-07       2         15 available 3.327156e-05
    sTfR  2123 Primary Discordant  519    0.212216        0.111530  13.209079 4.917604e-04       2         15 available 1.327753e-02
    sTfR  2123 Primary  H_private  519    0.072035        0.032139   6.043996 1.188114e-02       2         15 available 2.280865e-01
    sTfR  2123 Primary  G_private  519    0.019080        0.008053   3.196615 9.400606e-02       1         15 available 1.000000e+00
     CRP  2123 Primary     Shared 2415    0.025990        0.018268  30.891294 4.799697e-06       2         15 available 1.486085e-04
     CRP  2123 Primary Discordant 2415    0.006453        0.004446   6.118167 1.140433e-02       2         15 available 2.280865e-01
     CRP  2123 Primary  H_private 2415    0.001957        0.001342   0.940757 4.121915e-01       2         15 available 1.000000e+00
     CRP  2123 Primary  G_private 2415    0.001160        0.000795   1.922643 1.858296e-01       1         15 available 1.000000e+00
```

## Holm-supported associations

```text
  anchor cycle   model      group    n  partial_R2  incremental_R2          F            p  df_num  design_df    status       p_holm
Ferritin  0506 Primary     Shared  416    0.119008        0.090101  12.103150 7.421906e-04       2         15 available 1.929696e-02
Ferritin  0506 Primary Discordant  416    0.151529        0.119120  38.779294 1.181841e-06       2         15 available 3.900075e-05
Ferritin  0506 Primary  H_private  416    0.103498        0.077003   9.961621 1.767361e-03       2         15 available 4.418402e-02
    sTfR  0506 Primary     Shared  417    0.261705        0.177753  40.746379 8.649162e-07       2         15 available 3.027207e-05
    sTfR  0506 Primary Discordant  417    0.186495        0.114959  42.310086 6.808944e-07       2         15 available 2.451220e-05
    sTfR  0506 Primary  H_private  417    0.127350        0.073180  30.897565 4.793821e-06       2         15 available 1.486085e-04
     CRP  0506 Primary     Shared 1688    0.031473        0.023146  14.552336 3.069375e-04       2         15 available 8.901186e-03
     CRP  0506 Primary Discordant 1688    0.014092        0.010181   9.624116 2.045965e-03       2         15 available 4.910316e-02
    OGTT  0506 Primary     Shared 1407    0.144450        0.098085 112.511781 9.306371e-10       2         15 available 3.908676e-08
    OGTT  0506 Primary Discordant 1407    0.064519        0.040067 117.607512 6.812932e-10       2         15 available 2.929561e-08
    OGTT  0506 Primary  G_private 1407    0.182517        0.129704 127.701080 9.779152e-09       1         15 available 3.813869e-07
Ferritin  0708 Primary     Shared  513    0.095500        0.072066  30.232657 3.674887e-06       2         16 available 1.175964e-04
    sTfR  0708 Primary     Shared  511    0.170798        0.096302  89.951285 1.979880e-09       2         16 available 8.117507e-08
    sTfR  0708 Primary Discordant  511    0.178232        0.101403  67.428128 1.601228e-08       2         16 available 6.084667e-07
    OGTT  0708 Primary     Shared 1659    0.130013        0.091615  46.909085 2.030328e-07       2         16 available 7.512214e-06
    OGTT  0708 Primary Discordant 1659    0.083529        0.055874  87.893152 2.346492e-09       2         16 available 9.385970e-08
    OGTT  0708 Primary  G_private 1659    0.134479        0.095251 255.126508 2.967489e-11       1         16 available 1.305695e-09
Ferritin  2123 Primary     Shared  519    0.110074        0.087693  13.836832 3.930836e-04       2         15 available 1.100634e-02
    sTfR  2123 Primary     Shared  519    0.168184        0.083711  39.958649 9.785752e-07       2         15 available 3.327156e-05
    sTfR  2123 Primary Discordant  519    0.212216        0.111530  13.209079 4.917604e-04       2         15 available 1.327753e-02
     CRP  2123 Primary     Shared 2415    0.025990        0.018268  30.891294 4.799697e-06       2         15 available 1.486085e-04
```

## Frozen paired contrasts

```text
  anchor cycle    n representation_reference anchor_training_cycle      group                     comparison      gain     low95    high95  replicates
Ferritin  0708  513                2005-2008                  0506     Shared            group_vs_covariates  0.172578  0.112522  0.241275         999
Ferritin  0708  513                2005-2008                  0506     Shared                   group_vs_PCA -0.065823 -0.157203  0.009612         999
Ferritin  0708  513                2005-2008                  0506     Shared conditional_group_contribution  0.073860  0.025879  0.126394         999
Ferritin  0708  513                2005-2008                  0506 Discordant            group_vs_covariates  0.165612  0.080484  0.263200         999
Ferritin  0708  513                2005-2008                  0506 Discordant                   group_vs_PCA -0.072789 -0.159940  0.002627         999
Ferritin  0708  513                2005-2008                  0506 Discordant conditional_group_contribution  0.080406  0.016640  0.154421         999
Ferritin  0708  513                2005-2008                  0506  H_private            group_vs_covariates  0.079554  0.019598  0.153487         999
Ferritin  0708  513                2005-2008                  0506  H_private                   group_vs_PCA -0.158847 -0.237883 -0.091061         999
Ferritin  0708  513                2005-2008                  0506  H_private conditional_group_contribution  0.029831 -0.033419  0.098302         999
Ferritin  0708  513                2005-2008                  0506  G_private            group_vs_covariates  0.005400 -0.005547  0.015548         999
Ferritin  0708  513                2005-2008                  0506  G_private                   group_vs_PCA -0.140543 -0.209519 -0.079509         999
Ferritin  0708  513                2005-2008                  0506  G_private conditional_group_contribution  0.004140 -0.001290  0.009320         999
Ferritin  2123  519                2005-2008                  0506     Shared            group_vs_covariates  0.114509  0.039204  0.208334         999
Ferritin  2123  519                2005-2008                  0506     Shared                   group_vs_PCA  0.128133 -0.031099  0.277202         999
Ferritin  2123  519                2005-2008                  0506     Shared conditional_group_contribution -0.070202 -0.148520 -0.002129         999
Ferritin  2123  519                2005-2008                  0506 Discordant            group_vs_covariates  0.057691 -0.064161  0.188463         999
Ferritin  2123  519                2005-2008                  0506 Discordant                   group_vs_PCA  0.071316 -0.042383  0.174078         999
Ferritin  2123  519                2005-2008                  0506 Discordant conditional_group_contribution -0.152755 -0.331609  0.013601         999
Ferritin  2123  519                2005-2008                  0506  H_private            group_vs_covariates -0.022183 -0.146576  0.121271         999
Ferritin  2123  519                2005-2008                  0506  H_private                   group_vs_PCA -0.008558 -0.071395  0.046768         999
Ferritin  2123  519                2005-2008                  0506  H_private conditional_group_contribution -0.198098 -0.349726 -0.052854         999
Ferritin  2123  519                2005-2008                  0506  G_private            group_vs_covariates  0.000576 -0.011558  0.012770         999
Ferritin  2123  519                2005-2008                  0506  G_private                   group_vs_PCA -0.067813 -0.135572 -0.014669         999
Ferritin  2123  519                2005-2008                  0506  G_private conditional_group_contribution  0.015608  0.000392  0.030276         999
    sTfR  0708  511                2005-2008                  0506     Shared            group_vs_covariates  0.048096  0.032472  0.065759         999
    sTfR  0708  511                2005-2008                  0506     Shared                   group_vs_PCA -0.006963 -0.019517  0.003828         999
    sTfR  0708  511                2005-2008                  0506     Shared conditional_group_contribution  0.024237  0.014411  0.035343         999
    sTfR  0708  511                2005-2008                  0506 Discordant            group_vs_covariates  0.044884  0.028980  0.061532         999
    sTfR  0708  511                2005-2008                  0506 Discordant                   group_vs_PCA -0.010175 -0.028988  0.008229         999
    sTfR  0708  511                2005-2008                  0506 Discordant conditional_group_contribution  0.025903  0.015865  0.036951         999
    sTfR  0708  511                2005-2008                  0506  H_private            group_vs_covariates  0.008742 -0.006293  0.023987         999
    sTfR  0708  511                2005-2008                  0506  H_private                   group_vs_PCA -0.046318 -0.065237 -0.030789         999
    sTfR  0708  511                2005-2008                  0506  H_private conditional_group_contribution -0.004769 -0.017229  0.007582         999
    sTfR  0708  511                2005-2008                  0506  G_private            group_vs_covariates -0.000030 -0.000163  0.000108         999
    sTfR  0708  511                2005-2008                  0506  G_private                   group_vs_PCA -0.040674 -0.059670 -0.023542         999
    sTfR  0708  511                2005-2008                  0506  G_private conditional_group_contribution  0.000153 -0.000260  0.000582         999
    sTfR  2123  519                2005-2008                  0506     Shared            group_vs_covariates  0.031525  0.006737  0.060802         999
    sTfR  2123  519                2005-2008                  0506     Shared                   group_vs_PCA -0.004473 -0.033753  0.019835         999
    sTfR  2123  519                2005-2008                  0506     Shared conditional_group_contribution -0.015037 -0.029243 -0.003135         999
    sTfR  2123  519                2005-2008                  0506 Discordant            group_vs_covariates  0.045079  0.009134  0.086457         999
    sTfR  2123  519                2005-2008                  0506 Discordant                   group_vs_PCA  0.009080 -0.006267  0.023775         999
    sTfR  2123  519                2005-2008                  0506 Discordant conditional_group_contribution  0.000652 -0.020269  0.019002         999
    sTfR  2123  519                2005-2008                  0506  H_private            group_vs_covariates  0.022544 -0.003330  0.053840         999
    sTfR  2123  519                2005-2008                  0506  H_private                   group_vs_PCA -0.013455 -0.037956  0.010314         999
    sTfR  2123  519                2005-2008                  0506  H_private conditional_group_contribution -0.017138 -0.030614 -0.003167         999
    sTfR  2123  519                2005-2008                  0506  G_private            group_vs_covariates  0.000049 -0.000062  0.000170         999
    sTfR  2123  519                2005-2008                  0506  G_private                   group_vs_PCA -0.034114 -0.062977 -0.008938         999
    sTfR  2123  519                2005-2008                  0506  G_private conditional_group_contribution  0.000032 -0.000441  0.000518         999
     CRP  0708 2070                2005-2008                  0506     Shared            group_vs_covariates -0.003685 -0.025064  0.016630         999
     CRP  0708 2070                2005-2008                  0506     Shared                   group_vs_PCA  0.011382 -0.000097  0.021013         999
     CRP  0708 2070                2005-2008                  0506     Shared conditional_group_contribution -0.011449 -0.037818  0.013570         999
     CRP  0708 2070                2005-2008                  0506 Discordant            group_vs_covariates -0.009582 -0.020765  0.002953         999
     CRP  0708 2070                2005-2008                  0506 Discordant                   group_vs_PCA  0.005486 -0.016651  0.029250         999
     CRP  0708 2070                2005-2008                  0506 Discordant conditional_group_contribution -0.003680 -0.017822  0.010968         999
     CRP  0708 2070                2005-2008                  0506  H_private            group_vs_covariates -0.009635 -0.017129 -0.002674         999
     CRP  0708 2070                2005-2008                  0506  H_private                   group_vs_PCA  0.005432 -0.007969  0.018520         999
     CRP  0708 2070                2005-2008                  0506  H_private conditional_group_contribution -0.022931 -0.036754 -0.009887         999
     CRP  0708 2070                2005-2008                  0506  G_private            group_vs_covariates -0.000774 -0.001849  0.000332         999
     CRP  0708 2070                2005-2008                  0506  G_private                   group_vs_PCA  0.014979 -0.001324  0.031736         999
     CRP  0708 2070                2005-2008                  0506  G_private conditional_group_contribution  0.000234 -0.001458  0.002052         999
     CRP  2123 2415                2005-2008                  0506     Shared            group_vs_covariates -0.019866 -0.037718  0.000015         999
     CRP  2123 2415                2005-2008                  0506     Shared                   group_vs_PCA  0.015600  0.004521  0.027789         999
     CRP  2123 2415                2005-2008                  0506     Shared conditional_group_contribution -0.048820 -0.075703 -0.020936         999
     CRP  2123 2415                2005-2008                  0506 Discordant            group_vs_covariates  0.003432 -0.012240  0.017492         999
     CRP  2123 2415                2005-2008                  0506 Discordant                   group_vs_PCA  0.038899  0.003488  0.071540         999
     CRP  2123 2415                2005-2008                  0506 Discordant conditional_group_contribution  0.013366 -0.001887  0.026455         999
     CRP  2123 2415                2005-2008                  0506  H_private            group_vs_covariates -0.016044 -0.028903 -0.002424         999
     CRP  2123 2415                2005-2008                  0506  H_private                   group_vs_PCA  0.019422  0.004001  0.036375         999
     CRP  2123 2415                2005-2008                  0506  H_private conditional_group_contribution -0.045495 -0.065380 -0.024243         999
     CRP  2123 2415                2005-2008                  0506  G_private            group_vs_covariates -0.001160 -0.002902  0.000516         999
     CRP  2123 2415                2005-2008                  0506  G_private                   group_vs_PCA  0.036879  0.012378  0.058888         999
     CRP  2123 2415                2005-2008                  0506  G_private conditional_group_contribution -0.003423 -0.005670 -0.001365         999
    OGTT  0708 1659                2005-2008                  0506     Shared            group_vs_covariates  0.006859  0.002086  0.011876         999
    OGTT  0708 1659                2005-2008                  0506     Shared                   group_vs_PCA -0.004171 -0.007895 -0.000821         999
    OGTT  0708 1659                2005-2008                  0506     Shared conditional_group_contribution  0.014642  0.008432  0.021255         999
    OGTT  0708 1659                2005-2008                  0506 Discordant            group_vs_covariates  0.001334  0.000605  0.002173         999
    OGTT  0708 1659                2005-2008                  0506 Discordant                   group_vs_PCA -0.009696 -0.017423 -0.002601         999
    OGTT  0708 1659                2005-2008                  0506 Discordant conditional_group_contribution  0.007033  0.005164  0.009444         999
    OGTT  0708 1659                2005-2008                  0506  H_private            group_vs_covariates -0.000363 -0.000552 -0.000185         999
    OGTT  0708 1659                2005-2008                  0506  H_private                   group_vs_PCA -0.011393 -0.019522 -0.003969         999
    OGTT  0708 1659                2005-2008                  0506  H_private conditional_group_contribution  0.000021 -0.000143  0.000158         999
    OGTT  0708 1659                2005-2008                  0506  G_private            group_vs_covariates  0.005340  0.001690  0.009416         999
    OGTT  0708 1659                2005-2008                  0506  G_private                   group_vs_PCA -0.000051 -0.005813  0.005835         999
    OGTT  0708 1659                2005-2008                  0506  G_private conditional_group_contribution  0.012734  0.008786  0.017137         999
```

## Random rotation controls

```text
  anchor cycle    n representation_reference anchor_training_cycle      group  dimensions  comparison   HG_gain  rotation_median  rotation_q025  rotation_q975  HG_percentile  rotations
Ferritin  0708  513                2005-2008                  0506     Shared           2    marginal  0.172578         0.125032       0.006063       0.252620      74.371859        199
Ferritin  0708  513                2005-2008                  0506     Shared           2 conditional  0.073860         0.036138      -0.003557       0.185435      77.889447        199
Ferritin  0708  513                2005-2008                  0506 Discordant           2    marginal  0.165612         0.125032       0.006063       0.252620      70.854271        199
Ferritin  0708  513                2005-2008                  0506 Discordant           2 conditional  0.080406         0.036138      -0.003557       0.185435      79.396985        199
Ferritin  0708  513                2005-2008                  0506  H_private           2    marginal  0.079554         0.125032       0.006063       0.252620      31.658291        199
Ferritin  0708  513                2005-2008                  0506  H_private           2 conditional  0.029831         0.036138      -0.003557       0.185435      44.723618        199
Ferritin  0708  513                2005-2008                  0506  G_private           1    marginal  0.005400         0.038611      -0.002564       0.222002      23.115578        199
Ferritin  0708  513                2005-2008                  0506  G_private           1 conditional  0.004140         0.009142      -0.005724       0.109075      37.688442        199
Ferritin  2123  519                2005-2008                  0506     Shared           2    marginal  0.114509         0.020155      -0.206054       0.131732      93.467337        199
Ferritin  2123  519                2005-2008                  0506     Shared           2 conditional -0.070202        -0.080230      -0.272399       0.192845      52.763819        199
Ferritin  2123  519                2005-2008                  0506 Discordant           2    marginal  0.057691         0.020155      -0.206054       0.131732      69.849246        199
Ferritin  2123  519                2005-2008                  0506 Discordant           2 conditional -0.152755        -0.080230      -0.272399       0.192845      31.155779        199
Ferritin  2123  519                2005-2008                  0506  H_private           2    marginal -0.022183         0.020155      -0.206054       0.131732      21.105528        199
Ferritin  2123  519                2005-2008                  0506  H_private           2 conditional -0.198098        -0.080230      -0.272399       0.192845      16.080402        199
Ferritin  2123  519                2005-2008                  0506  G_private           1    marginal  0.000576         0.012008      -0.066530       0.115459      32.663317        199
Ferritin  2123  519                2005-2008                  0506  G_private           1 conditional  0.015608        -0.006327      -0.225184       0.151317      71.356784        199
    sTfR  0708  511                2005-2008                  0506     Shared           2    marginal  0.048096         0.029497      -0.001165       0.064351      79.899497        199
    sTfR  0708  511                2005-2008                  0506     Shared           2 conditional  0.024237         0.007789      -0.003552       0.042997      80.904523        199
    sTfR  0708  511                2005-2008                  0506 Discordant           2    marginal  0.044884         0.029497      -0.001165       0.064351      74.874372        199
    sTfR  0708  511                2005-2008                  0506 Discordant           2 conditional  0.025903         0.007789      -0.003552       0.042997      83.417085        199
    sTfR  0708  511                2005-2008                  0506  H_private           2    marginal  0.008742         0.029497      -0.001165       0.064351      19.597990        199
    sTfR  0708  511                2005-2008                  0506  H_private           2 conditional -0.004769         0.007789      -0.003552       0.042997       1.005025        199
    sTfR  0708  511                2005-2008                  0506  G_private           1    marginal -0.000030         0.008204      -0.002501       0.056812      18.592965        199
    sTfR  0708  511                2005-2008                  0506  G_private           1 conditional  0.000153         0.001163      -0.004399       0.029711      36.683417        199
    sTfR  2123  519                2005-2008                  0506     Shared           2    marginal  0.031525         0.027729      -0.005548       0.047583      59.296482        199
    sTfR  2123  519                2005-2008                  0506     Shared           2 conditional -0.015037        -0.003536      -0.018579       0.020361      10.050251        199
    sTfR  2123  519                2005-2008                  0506 Discordant           2    marginal  0.045079         0.027729      -0.005548       0.047583      95.979899        199
    sTfR  2123  519                2005-2008                  0506 Discordant           2 conditional  0.000652        -0.003536      -0.018579       0.020361      67.336683        199
    sTfR  2123  519                2005-2008                  0506  H_private           2    marginal  0.022544         0.027729      -0.005548       0.047583      40.201005        199
    sTfR  2123  519                2005-2008                  0506  H_private           2 conditional -0.017138        -0.003536      -0.018579       0.020361       5.025126        199
    sTfR  2123  519                2005-2008                  0506  G_private           1    marginal  0.000049         0.011042      -0.005646       0.045661      21.608040        199
    sTfR  2123  519                2005-2008                  0506  G_private           1 conditional  0.000032        -0.000616      -0.014637       0.011843      60.301508        199
     CRP  0708 2070                2005-2008                  0506     Shared           2    marginal -0.003685        -0.005720      -0.038459       0.011549      57.788945        199
     CRP  0708 2070                2005-2008                  0506     Shared           2 conditional -0.011449        -0.005568      -0.039010       0.022191      37.688442        199
     CRP  0708 2070                2005-2008                  0506 Discordant           2    marginal -0.009582        -0.005720      -0.038459       0.011549      40.703518        199
     CRP  0708 2070                2005-2008                  0506 Discordant           2 conditional -0.003680        -0.005568      -0.039010       0.022191      55.778894        199
     CRP  0708 2070                2005-2008                  0506  H_private           2    marginal -0.009635        -0.005720      -0.038459       0.011549      40.703518        199
     CRP  0708 2070                2005-2008                  0506  H_private           2 conditional -0.022931        -0.005568      -0.039010       0.022191      19.095477        199
     CRP  0708 2070                2005-2008                  0506  G_private           1    marginal -0.000774        -0.001929      -0.028883       0.009743      59.798995        199
     CRP  0708 2070                2005-2008                  0506  G_private           1 conditional  0.000234        -0.001873      -0.030437       0.022482      65.829146        199
     CRP  2123 2415                2005-2008                  0506     Shared           2    marginal -0.019866        -0.004207      -0.047965       0.012231      19.095477        199
     CRP  2123 2415                2005-2008                  0506     Shared           2 conditional -0.048820        -0.021213      -0.061274       0.016878      14.070352        199
     CRP  2123 2415                2005-2008                  0506 Discordant           2    marginal  0.003432        -0.004207      -0.047965       0.012231      80.904523        199
     CRP  2123 2415                2005-2008                  0506 Discordant           2 conditional  0.013366        -0.021213      -0.061274       0.016878      93.467337        199
     CRP  2123 2415                2005-2008                  0506  H_private           2    marginal -0.016044        -0.004207      -0.047965       0.012231      23.618090        199
     CRP  2123 2415                2005-2008                  0506  H_private           2 conditional -0.045495        -0.021213      -0.061274       0.016878      16.582915        199
     CRP  2123 2415                2005-2008                  0506  G_private           1    marginal -0.001160        -0.000865      -0.036133       0.011419      47.738693        199
     CRP  2123 2415                2005-2008                  0506  G_private           1 conditional -0.003423        -0.006655      -0.055178       0.020312      56.281407        199
    OGTT  0708 1659                2005-2008                  0506     Shared           2    marginal  0.006859         0.002426      -0.001643       0.012117      85.427136        199
    OGTT  0708 1659                2005-2008                  0506     Shared           2 conditional  0.014642         0.012458       0.000829       0.022544      68.844221        199
    OGTT  0708 1659                2005-2008                  0506 Discordant           2    marginal  0.001334         0.002426      -0.001643       0.012117      39.698492        199
    OGTT  0708 1659                2005-2008                  0506 Discordant           2 conditional  0.007033         0.012458       0.000829       0.022544      28.643216        199
    OGTT  0708 1659                2005-2008                  0506  H_private           2    marginal -0.000363         0.002426      -0.001643       0.012117      13.065327        199
    OGTT  0708 1659                2005-2008                  0506  H_private           2 conditional  0.000021         0.012458       0.000829       0.022544       0.000000        199
    OGTT  0708 1659                2005-2008                  0506  G_private           1    marginal  0.005340         0.000364      -0.002126       0.007686      91.959799        199
    OGTT  0708 1659                2005-2008                  0506  G_private           1 conditional  0.012734         0.005682      -0.000041       0.019238      80.402010        199
```

## Fold-specific CV sensitivity

```text
  outcome            method  alpha  validation_MSE
  Somatic DiscoveryWide_HG7    100        3.409680
  Somatic  FoldRefitted_HG7    100        3.409720
  Somatic              Raw7     10        3.411155
   CogAff DiscoveryWide_HG7    100        4.430357
   CogAff  FoldRefitted_HG7    100        4.430369
   CogAff              Raw7     10        4.432889
PHQ_total DiscoveryWide_HG7    100       12.292577
PHQ_total  FoldRefitted_HG7    100       12.292686
PHQ_total              Raw7     10       12.298484
```

## Interpretation

Conditional associations support construct validity only when interpreted against comparator evidence.

Neither significant associations nor superiority over random subspaces establishes biological causality.

CV refits the HG representation within each fold, while retaining the existing covariate encoding.

Holdout bootstrap intervals condition on fitted models and are exploratory and unadjusted for multiplicity.

The available scaffold may retain selection bias. Assay harmonisation and unmeasured confounding remain unresolved.