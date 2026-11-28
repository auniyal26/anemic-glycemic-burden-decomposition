# Script122 — Statistical robustness audit

## Completion

```text
{
  "completed": true,
  "per_model_deletion_fraction": 0.01,
  "regression_models": 45,
  "successful_refits": 3795,
  "unsuccessful_or_limited_records": 0,
  "structural_failed_or_limited_records": 0,
  "full_rank_invariance_passed": true,
  "existing_inference_files_found": 8,
  "existing_inference_files_expected": 8,
  "interpretation": [
    "Classical WLS influence statistics are descriptive.",
    "PSU deletions are sensitivity analyses, not confidence intervals.",
    "Regression coefficients use standardized predictors.",
    "Within-period fits are not frozen prediction models.",
    "Structural refits retain fixed shared rank two.",
    "Historical survey inference is copied, not recomputed.",
    "No causal or clinical robustness claim is established."
  ]
}
```

## Full-rank invariance

```text
   period   outcome         pred     leverage        cooks      student  passed
2005-2008 PHQ_total 2.486900e-14 3.417405e-16 4.250073e-17 9.325873e-15    True
2005-2008   Somatic 6.439294e-15 3.417405e-16 5.290907e-17 4.329870e-15    True
2005-2008    CogAff 9.547918e-15 3.417405e-16 7.979728e-17 6.272760e-15    True
2009-2018 PHQ_total 1.687539e-14 1.379105e-16 2.081668e-17 4.884981e-15    True
2009-2018   Somatic 7.993606e-15 1.379105e-16 3.122502e-17 3.774758e-15    True
2009-2018    CogAff 1.509903e-14 1.379105e-16 2.775558e-17 6.217249e-15    True
2021-2023 PHQ_total 1.687539e-14 2.012279e-16 2.428613e-16 5.773160e-15    True
2021-2023   Somatic 9.325873e-15 2.012279e-16 1.665335e-16 5.051515e-15    True
2021-2023    CogAff 6.883383e-15 2.012279e-16 1.387779e-16 3.330669e-15    True
```

## Structural robustness

```text
               perturbation               group  count      min   median      max
                 delete_PSU          Discordant     62 0.998544 0.999675 0.999959
                 delete_PSU   Glycaemia-private     62 0.996525 0.999852 0.999997
                 delete_PSU Haematology-private     62 0.998894 0.999714 0.999995
                 delete_PSU              Shared     62 0.999229 0.999807 0.999980
high_physiological_leverage          Discordant      1 0.985496 0.985496 0.985496
high_physiological_leverage   Glycaemia-private      1 0.982295 0.982295 0.982295
high_physiological_leverage Haematology-private      1 0.994799 0.994799 0.994799
high_physiological_leverage              Shared      1 0.992018 0.992018 0.992018
         high_survey_weight          Discordant      1 0.999753 0.999753 0.999753
         high_survey_weight   Glycaemia-private      1 0.999956 0.999956 0.999956
         high_survey_weight Haematology-private      1 0.999664 0.999664 0.999664
         high_survey_weight              Shared      1 0.999863 0.999863 0.999863
```

## Regression deletion sensitivity

```text
   period   outcome            method   sensitivity  successful_refits  maximum_prediction_shift  maximum_beta_change  maximum_original_sample_MSE_change
2005-2008    CogAff       Continuous2    delete_PSU                 62                  0.060081             0.042121                            0.003610
2005-2008    CogAff       Continuous2     high_cook                  1                  0.223100             0.086098                            0.049773
2005-2008    CogAff       Continuous2   high_dfbeta                  1                  0.201575             0.078271                            0.040632
2005-2008    CogAff       Continuous2 high_leverage                  1                  0.043762             0.042357                            0.001915
2005-2008    CogAff               HG7    delete_PSU                 62                  0.067442             0.036633                            0.004548
2005-2008    CogAff               HG7     high_cook                  1                  0.229125             0.073938                            0.052498
2005-2008    CogAff               HG7   high_dfbeta                  1                  0.207891             0.067246                            0.043219
2005-2008    CogAff               HG7 high_leverage                  1                  0.072821             0.060472                            0.005303
2005-2008    CogAff              Raw7    delete_PSU                 62                  0.067442             0.089062                            0.004548
2005-2008    CogAff              Raw7     high_cook                  1                  0.229125             0.182319                            0.052498
2005-2008    CogAff              Raw7   high_dfbeta                  1                  0.207891             0.209088                            0.043219
2005-2008    CogAff              Raw7 high_leverage                  1                  0.072821             0.073086                            0.005303
2005-2008    CogAff            Scalar    delete_PSU                 62                  0.059998             0.038318                            0.003600
2005-2008    CogAff            Scalar     high_cook                  1                  0.219995             0.082557                            0.048398
2005-2008    CogAff            Scalar   high_dfbeta                  1                  0.196576             0.067761                            0.038642
2005-2008    CogAff            Scalar high_leverage                  1                  0.040300             0.029207                            0.001624
2005-2008    CogAff ScalarInteraction    delete_PSU                 62                  0.059882             0.039046                            0.003586
2005-2008    CogAff ScalarInteraction     high_cook                  1                  0.220465             0.080369                            0.048605
2005-2008    CogAff ScalarInteraction   high_dfbeta                  1                  0.197454             0.065040                            0.038988
2005-2008    CogAff ScalarInteraction high_leverage                  1                  0.079257             0.072309                            0.006282
2005-2008 PHQ_total       Continuous2    delete_PSU                 62                  0.107053             0.074867                            0.011460
2005-2008 PHQ_total       Continuous2     high_cook                  1                  0.364632             0.117907                            0.132956
2005-2008 PHQ_total       Continuous2   high_dfbeta                  1                  0.347670             0.136329                            0.120875
2005-2008 PHQ_total       Continuous2 high_leverage                  1                  0.079207             0.045403                            0.006274
2005-2008 PHQ_total               HG7    delete_PSU                 62                  0.130339             0.062559                            0.016988
2005-2008 PHQ_total               HG7     high_cook                  1                  0.368233             0.096560                            0.135595
2005-2008 PHQ_total               HG7   high_dfbeta                  1                  0.351418             0.125005                            0.123494
2005-2008 PHQ_total               HG7 high_leverage                  1                  0.141836             0.094732                            0.020118
2005-2008 PHQ_total              Raw7    delete_PSU                 62                  0.130339             0.158927                            0.016988
2005-2008 PHQ_total              Raw7     high_cook                  1                  0.368233             0.250802                            0.135595
2005-2008 PHQ_total              Raw7   high_dfbeta                  1                  0.351418             0.256278                            0.123494
2005-2008 PHQ_total              Raw7 high_leverage                  1                  0.141836             0.155946                            0.020118
2005-2008 PHQ_total            Scalar    delete_PSU                 62                  0.103837             0.065942                            0.010782
2005-2008 PHQ_total            Scalar     high_cook                  1                  0.352780             0.075117                            0.124454
2005-2008 PHQ_total            Scalar   high_dfbeta                  1                  0.330886             0.121460                            0.109486
2005-2008 PHQ_total            Scalar high_leverage                  1                  0.106229             0.082590                            0.011285
2005-2008 PHQ_total ScalarInteraction    delete_PSU                 62                  0.103440             0.066594                            0.010700
2005-2008 PHQ_total ScalarInteraction     high_cook                  1                  0.354255             0.070645                            0.125497
2005-2008 PHQ_total ScalarInteraction   high_dfbeta                  1                  0.330523             0.121830                            0.109245
2005-2008 PHQ_total ScalarInteraction high_leverage                  1                  0.108851             0.085548                            0.011849
2005-2008   Somatic       Continuous2    delete_PSU                 62                  0.058424             0.033060                            0.003413
2005-2008   Somatic       Continuous2     high_cook                  1                  0.165515             0.036935                            0.027395
2005-2008   Somatic       Continuous2   high_dfbeta                  1                  0.141980             0.051481                            0.020158
2005-2008   Somatic       Continuous2 high_leverage                  1                  0.054702             0.011444                            0.002992
2005-2008   Somatic               HG7    delete_PSU                 62                  0.073313             0.037316                            0.005375
2005-2008   Somatic               HG7     high_cook                  1                  0.171670             0.050417                            0.029471
2005-2008   Somatic               HG7   high_dfbeta                  1                  0.141263             0.041430                            0.019955
2005-2008   Somatic               HG7 high_leverage                  1                  0.081143             0.042037                            0.006584
2005-2008   Somatic              Raw7    delete_PSU                 62                  0.073313             0.069866                            0.005375
2005-2008   Somatic              Raw7     high_cook                  1                  0.171670             0.053474                            0.029471
2005-2008   Somatic              Raw7   high_dfbeta                  1                  0.141263             0.030175                            0.019955
2005-2008   Somatic              Raw7 high_leverage                  1                  0.081143             0.084317                            0.006584
2005-2008   Somatic            Scalar    delete_PSU                 62                  0.055256             0.027624                            0.003053
2005-2008   Somatic            Scalar     high_cook                  1                  0.161563             0.026865                            0.026103
2005-2008   Somatic            Scalar   high_dfbeta                  1                  0.136292             0.028946                            0.018575
2005-2008   Somatic            Scalar high_leverage                  1                  0.081574             0.060569                            0.006654
2005-2008   Somatic ScalarInteraction    delete_PSU                 62                  0.055088             0.027601                            0.003035
2005-2008   Somatic ScalarInteraction     high_cook                  1                  0.161555             0.025438                            0.026100
2005-2008   Somatic ScalarInteraction   high_dfbeta                  1                  0.137078             0.033815                            0.018790
2005-2008   Somatic ScalarInteraction high_leverage                  1                  0.079231             0.050642                            0.006278
2009-2018    CogAff       Continuous2    delete_PSU                152                  0.034441             0.021973                            0.001186
2009-2018    CogAff       Continuous2     high_cook                  1                  0.254362             0.016802                            0.064700
2009-2018    CogAff       Continuous2   high_dfbeta                  1                  0.250173             0.024429                            0.062587
2009-2018    CogAff       Continuous2 high_leverage                  1                  0.051163             0.027567                            0.002618
2009-2018    CogAff               HG7    delete_PSU                152                  0.042044             0.024100                            0.001768
2009-2018    CogAff               HG7     high_cook                  1                  0.262538             0.060325                            0.068926
2009-2018    CogAff               HG7   high_dfbeta                  1                  0.259537             0.069490                            0.067360
2009-2018    CogAff               HG7 high_leverage                  1                  0.069085             0.038177                            0.004773
2009-2018    CogAff              Raw7    delete_PSU                152                  0.042044             0.057862                            0.001768
2009-2018    CogAff              Raw7     high_cook                  1                  0.262538             0.195138                            0.068926
2009-2018    CogAff              Raw7   high_dfbeta                  1                  0.259537             0.223480                            0.067360
2009-2018    CogAff              Raw7 high_leverage                  1                  0.069085             0.142289                            0.004773
2009-2018    CogAff            Scalar    delete_PSU                152                  0.034526             0.015542                            0.001192
2009-2018    CogAff            Scalar     high_cook                  1                  0.256835             0.037508                            0.065964
2009-2018    CogAff            Scalar   high_dfbeta                  1                  0.253878             0.041250                            0.064454
2009-2018    CogAff            Scalar high_leverage                  1                  0.054772             0.028617                            0.003000
2009-2018    CogAff ScalarInteraction    delete_PSU                152                  0.034572             0.017651                            0.001195
2009-2018    CogAff ScalarInteraction     high_cook                  1                  0.257107             0.043438                            0.066104
2009-2018    CogAff ScalarInteraction   high_dfbeta                  1                  0.257567             0.057892                            0.066341
2009-2018    CogAff ScalarInteraction high_leverage                  1                  0.064498             0.035885                            0.004160
2009-2018 PHQ_total       Continuous2    delete_PSU                152                  0.054251             0.039591                            0.002943
2009-2018 PHQ_total       Continuous2     high_cook                  1                  0.357665             0.017280                            0.127925
2009-2018 PHQ_total       Continuous2   high_dfbeta                  1                  0.362977             0.026718                            0.131752
2009-2018 PHQ_total       Continuous2 high_leverage                  1                  0.081051             0.041802                            0.006569
2009-2018 PHQ_total               HG7    delete_PSU                152                  0.061598             0.032669                            0.003794
2009-2018 PHQ_total               HG7     high_cook                  1                  0.373445             0.102823                            0.139461
2009-2018 PHQ_total               HG7   high_dfbeta                  1                  0.382883             0.124029                            0.146599
2009-2018 PHQ_total               HG7 high_leverage                  1                  0.110015             0.050741                            0.012103
2009-2018 PHQ_total              Raw7    delete_PSU                152                  0.061598             0.095660                            0.003794
2009-2018 PHQ_total              Raw7     high_cook                  1                  0.373445             0.213177                            0.139461
2009-2018 PHQ_total              Raw7   high_dfbeta                  1                  0.382883             0.271730                            0.146599
2009-2018 PHQ_total              Raw7 high_leverage                  1                  0.110015             0.124735                            0.012103
2009-2018 PHQ_total            Scalar    delete_PSU                152                  0.054308             0.030977                            0.002949
2009-2018 PHQ_total            Scalar     high_cook                  1                  0.361858             0.058203                            0.130941
2009-2018 PHQ_total            Scalar   high_dfbeta                  1                  0.364677             0.037016                            0.132990
2009-2018 PHQ_total            Scalar high_leverage                  1                  0.090225             0.052752                            0.008141
2009-2018 PHQ_total ScalarInteraction    delete_PSU                152                  0.054369             0.035237                            0.002956
2009-2018 PHQ_total ScalarInteraction     high_cook                  1                  0.374056             0.116343                            0.139918
2009-2018 PHQ_total ScalarInteraction   high_dfbeta                  1                  0.383390             0.140544                            0.146988
2009-2018 PHQ_total ScalarInteraction high_leverage                  1                  0.105426             0.057869                            0.011115
2009-2018   Somatic       Continuous2    delete_PSU                152                  0.022127             0.017619                            0.000490
2009-2018   Somatic       Continuous2     high_cook                  1                  0.174795             0.028435                            0.030553
2009-2018   Somatic       Continuous2   high_dfbeta                  1                  0.162437             0.023510                            0.026386
2009-2018   Somatic       Continuous2 high_leverage                  1                  0.040320             0.031445                            0.001626
2009-2018   Somatic               HG7    delete_PSU                152                  0.024676             0.017220                            0.000609
2009-2018   Somatic               HG7     high_cook                  1                  0.179492             0.040361                            0.032217
2009-2018   Somatic               HG7   high_dfbeta                  1                  0.170009             0.053488                            0.028903
2009-2018   Somatic               HG7 high_leverage                  1                  0.053992             0.040484                            0.002915
2009-2018   Somatic              Raw7    delete_PSU                152                  0.024676             0.038887                            0.000609
2009-2018   Somatic              Raw7     high_cook                  1                  0.179492             0.086146                            0.032217
2009-2018   Somatic              Raw7   high_dfbeta                  1                  0.170009             0.128365                            0.028903
2009-2018   Somatic              Raw7 high_leverage                  1                  0.053992             0.059390                            0.002915
2009-2018   Somatic            Scalar    delete_PSU                152                  0.022156             0.015434                            0.000491
2009-2018   Somatic            Scalar     high_cook                  1                  0.173751             0.018562                            0.030189
2009-2018   Somatic            Scalar   high_dfbeta                  1                  0.160188             0.006549                            0.025660
2009-2018   Somatic            Scalar high_leverage                  1                  0.039604             0.024135                            0.001569
2009-2018   Somatic ScalarInteraction    delete_PSU                152                  0.022721             0.017586                            0.000516
2009-2018   Somatic ScalarInteraction     high_cook                  1                  0.177614             0.042547                            0.031547
2009-2018   Somatic ScalarInteraction   high_dfbeta                  1                  0.164002             0.038784                            0.026897
2009-2018   Somatic ScalarInteraction high_leverage                  1                  0.044593             0.021985                            0.001989
2021-2023    CogAff       Continuous2    delete_PSU                 30                  0.141705             0.064541                            0.020080
2021-2023    CogAff       Continuous2     high_cook                  1                  0.265604             0.061646                            0.070545
2021-2023    CogAff       Continuous2   high_dfbeta                  1                  0.254123             0.083999                            0.064578
2021-2023    CogAff       Continuous2 high_leverage                  1                  0.141674             0.100398                            0.020072
2021-2023    CogAff               HG7    delete_PSU                 30                  0.163246             0.089463                            0.026649
2021-2023    CogAff               HG7     high_cook                  1                  0.292676             0.115845                            0.085659
2021-2023    CogAff               HG7   high_dfbeta                  1                  0.272155             0.107876                            0.074068
2021-2023    CogAff               HG7 high_leverage                  1                  0.191535             0.097526                            0.036686
2021-2023    CogAff              Raw7    delete_PSU                 30                  0.163246             0.187019                            0.026649
2021-2023    CogAff              Raw7     high_cook                  1                  0.292676             0.203565                            0.085659
2021-2023    CogAff              Raw7   high_dfbeta                  1                  0.272155             0.185516                            0.074068
2021-2023    CogAff              Raw7 high_leverage                  1                  0.191535             0.185577                            0.036686
2021-2023    CogAff            Scalar    delete_PSU                 30                  0.136772             0.048021                            0.018707
2021-2023    CogAff            Scalar     high_cook                  1                  0.262920             0.046201                            0.069127
2021-2023    CogAff            Scalar   high_dfbeta                  1                  0.250082             0.064941                            0.062541
2021-2023    CogAff            Scalar high_leverage                  1                  0.168919             0.120441                            0.028534
2021-2023    CogAff ScalarInteraction    delete_PSU                 30                  0.138561             0.053666                            0.019199
2021-2023    CogAff ScalarInteraction     high_cook                  1                  0.264099             0.055854                            0.069748
2021-2023    CogAff ScalarInteraction   high_dfbeta                  1                  0.252249             0.072876                            0.063630
2021-2023    CogAff ScalarInteraction high_leverage                  1                  0.159642             0.114759                            0.025485
2021-2023 PHQ_total       Continuous2    delete_PSU                 30                  0.235978             0.115361                            0.055686
2021-2023 PHQ_total       Continuous2     high_cook                  1                  0.418805             0.102629                            0.175397
2021-2023 PHQ_total       Continuous2   high_dfbeta                  1                  0.337363             0.059707                            0.113814
2021-2023 PHQ_total       Continuous2 high_leverage                  1                  0.241782             0.174901                            0.058458
2021-2023 PHQ_total               HG7    delete_PSU                 30                  0.258331             0.149302                            0.066735
2021-2023 PHQ_total               HG7     high_cook                  1                  0.483653             0.263237                            0.233920
2021-2023 PHQ_total               HG7   high_dfbeta                  1                  0.381048             0.180822                            0.145197
2021-2023 PHQ_total               HG7 high_leverage                  1                  0.342972             0.172710                            0.117630
2021-2023 PHQ_total              Raw7    delete_PSU                 30                  0.258331             0.268849                            0.066735
2021-2023 PHQ_total              Raw7     high_cook                  1                  0.483653             0.281757                            0.233920
2021-2023 PHQ_total              Raw7   high_dfbeta                  1                  0.381048             0.181700                            0.145197
2021-2023 PHQ_total              Raw7 high_leverage                  1                  0.342972             0.326921                            0.117630
2021-2023 PHQ_total            Scalar    delete_PSU                 30                  0.222414             0.080303                            0.049468
2021-2023 PHQ_total            Scalar     high_cook                  1                  0.426654             0.118892                            0.182034
2021-2023 PHQ_total            Scalar   high_dfbeta                  1                  0.348257             0.116064                            0.121283
2021-2023 PHQ_total            Scalar high_leverage                  1                  0.283231             0.210577                            0.080220
2021-2023 PHQ_total ScalarInteraction    delete_PSU                 30                  0.225018             0.088759                            0.050633
2021-2023 PHQ_total ScalarInteraction     high_cook                  1                  0.430507             0.141772                            0.185336
2021-2023 PHQ_total ScalarInteraction   high_dfbeta                  1                  0.351384             0.131241                            0.123471
2021-2023 PHQ_total ScalarInteraction high_leverage                  1                  0.277355             0.213297                            0.076926
2021-2023   Somatic       Continuous2    delete_PSU                 30                  0.129017             0.058023                            0.016645
2021-2023   Somatic       Continuous2     high_cook                  1                  0.172176             0.031296                            0.029645
2021-2023   Somatic       Continuous2   high_dfbeta                  1                  0.135712             0.032963                            0.018418
2021-2023   Somatic       Continuous2 high_leverage                  1                  0.128329             0.074503                            0.016468
2021-2023   Somatic               HG7    delete_PSU                 30                  0.136071             0.059839                            0.018515
2021-2023   Somatic               HG7     high_cook                  1                  0.213463             0.097137                            0.045567
2021-2023   Somatic               HG7   high_dfbeta                  1                  0.184067             0.099085                            0.033881
2021-2023   Somatic               HG7 high_leverage                  1                  0.175346             0.078382                            0.030746
2021-2023   Somatic              Raw7    delete_PSU                 30                  0.136071             0.202470                            0.018515
2021-2023   Somatic              Raw7     high_cook                  1                  0.213463             0.108246                            0.045567
2021-2023   Somatic              Raw7   high_dfbeta                  1                  0.184067             0.110383                            0.033881
2021-2023   Somatic              Raw7 high_leverage                  1                  0.175346             0.141344                            0.030746
2021-2023   Somatic            Scalar    delete_PSU                 30                  0.124303             0.036913                            0.015451
2021-2023   Somatic            Scalar     high_cook                  1                  0.184842             0.073402                            0.034166
2021-2023   Somatic            Scalar   high_dfbeta                  1                  0.143719             0.051696                            0.020655
2021-2023   Somatic            Scalar high_leverage                  1                  0.142327             0.090136                            0.020257
2021-2023   Somatic ScalarInteraction    delete_PSU                 30                  0.124646             0.041887                            0.015537
2021-2023   Somatic ScalarInteraction     high_cook                  1                  0.197631             0.101241                            0.039058
2021-2023   Somatic ScalarInteraction   high_dfbeta                  1                  0.148789             0.065882                            0.022138
2021-2023   Somatic ScalarInteraction high_leverage                  1                  0.144759             0.098538                            0.020955
```

## Interpretation boundaries

Influence thresholds are screening heuristics.
No observations are removed from the primary analysis.
PSU perturbations use the available pre-domain frame.
Residual diagnostics are descriptive and not formal
survey-adjusted specification tests.