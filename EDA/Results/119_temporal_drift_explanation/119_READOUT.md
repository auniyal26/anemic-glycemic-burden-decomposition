# Script119 — Temporal drift explanation

## Reproduction

Original Script118 weighted MSE results reproduced.

## Error decomposition

Positive differences favour HG shared2.

```text
cohort      reference  candidate            metric  estimate  psu_low95  psu_high95
  0708          RawX7 HG_Shared2               mse  -0.14311   -0.25868    -0.03789
  0708          RawX7 HG_Shared2   mean_bias_error  -0.02316   -0.04699    -0.00176
  0708          RawX7 HG_Shared2       slope_error  -0.03689   -0.08827     0.01194
  0708          RawX7 HG_Shared2 calibration_error  -0.06004   -0.12092    -0.00004
  0708          RawX7 HG_Shared2    residual_error  -0.08307   -0.18498    -0.00957
  0708           PCA2 HG_Shared2               mse  -0.10403   -0.20098    -0.02091
  0708           PCA2 HG_Shared2   mean_bias_error  -0.02180   -0.03887    -0.00428
  0708           PCA2 HG_Shared2       slope_error  -0.04070   -0.08502    -0.00236
  0708           PCA2 HG_Shared2 calibration_error  -0.06250   -0.11092    -0.01793
  0708           PCA2 HG_Shared2    residual_error  -0.04153   -0.14264     0.02480
  0708         H_PCA2 HG_Shared2               mse  -0.14241   -0.25285    -0.04061
  0708         H_PCA2 HG_Shared2   mean_bias_error  -0.02306   -0.04846     0.00142
  0708         H_PCA2 HG_Shared2       slope_error  -0.03830   -0.08817     0.00773
  0708         H_PCA2 HG_Shared2 calibration_error  -0.06136   -0.12162    -0.00404
  0708         H_PCA2 HG_Shared2    residual_error  -0.08105   -0.18040    -0.01007
  0708    JIVE_Joint2 HG_Shared2               mse   0.06072    0.02697     0.09803
  0708    JIVE_Joint2 HG_Shared2   mean_bias_error  -0.00448   -0.01827     0.00331
  0708    JIVE_Joint2 HG_Shared2       slope_error  -0.03511   -0.07856    -0.01122
  0708    JIVE_Joint2 HG_Shared2 calibration_error  -0.03959   -0.09017    -0.01072
  0708    JIVE_Joint2 HG_Shared2    residual_error   0.10032    0.04424     0.18120
  0708   AJIVE_Joint2 HG_Shared2               mse  -0.05627   -0.10262    -0.01325
  0708   AJIVE_Joint2 HG_Shared2   mean_bias_error  -0.01755   -0.03128    -0.00623
  0708   AJIVE_Joint2 HG_Shared2       slope_error  -0.02098   -0.04647     0.00569
  0708   AJIVE_Joint2 HG_Shared2 calibration_error  -0.03854   -0.06666    -0.01333
  0708   AJIVE_Joint2 HG_Shared2    residual_error  -0.01773   -0.08380     0.03561
  0708 VAE2_mean_loss HG_Shared2               mse  -0.09529   -0.17503    -0.02357
  0708 VAE2_mean_loss HG_Shared2   mean_bias_error  -0.01952   -0.03573    -0.00425
  0708 VAE2_mean_loss HG_Shared2       slope_error  -0.03912   -0.07836    -0.00430
  0708 VAE2_mean_loss HG_Shared2 calibration_error  -0.05864   -0.10148    -0.01665
  0708 VAE2_mean_loss HG_Shared2    residual_error  -0.03665   -0.12127     0.02067
  0708     Threshold2 HG_Shared2               mse  -0.03032   -0.08942     0.01821
  0708     Threshold2 HG_Shared2   mean_bias_error  -0.01103   -0.02249    -0.00338
  0708     Threshold2 HG_Shared2       slope_error  -0.04029   -0.07582    -0.00850
  0708     Threshold2 HG_Shared2 calibration_error  -0.05132   -0.09315    -0.01495
  0708     Threshold2 HG_Shared2    residual_error   0.02101   -0.04583     0.07486
  2123          RawX7 HG_Shared2               mse   0.17168    0.04997     0.30613
  2123          RawX7 HG_Shared2   mean_bias_error   0.18005    0.13755     0.23317
  2123          RawX7 HG_Shared2       slope_error   0.04401   -0.00633     0.12145
  2123          RawX7 HG_Shared2 calibration_error   0.22406    0.14551     0.33112
  2123          RawX7 HG_Shared2    residual_error  -0.05238   -0.11009     0.00552
  2123           PCA2 HG_Shared2               mse   0.02957   -0.03855     0.09241
  2123           PCA2 HG_Shared2   mean_bias_error   0.08936    0.06347     0.11869
  2123           PCA2 HG_Shared2       slope_error  -0.00606   -0.04291     0.01806
  2123           PCA2 HG_Shared2 calibration_error   0.08330    0.04537     0.11710
  2123           PCA2 HG_Shared2    residual_error  -0.05373   -0.11530     0.00743
  2123         H_PCA2 HG_Shared2               mse   0.11161    0.02424     0.20298
  2123         H_PCA2 HG_Shared2   mean_bias_error   0.15787    0.12046     0.20181
  2123         H_PCA2 HG_Shared2       slope_error   0.01868   -0.02432     0.06591
  2123         H_PCA2 HG_Shared2 calibration_error   0.17655    0.11761     0.24241
  2123         H_PCA2 HG_Shared2    residual_error  -0.06494   -0.12077    -0.01340
  2123    JIVE_Joint2 HG_Shared2               mse   0.04407    0.00209     0.08859
  2123    JIVE_Joint2 HG_Shared2   mean_bias_error  -0.01349   -0.02464    -0.00481
  2123    JIVE_Joint2 HG_Shared2       slope_error  -0.01163   -0.03969     0.00820
  2123    JIVE_Joint2 HG_Shared2 calibration_error  -0.02512   -0.05498    -0.00568
  2123    JIVE_Joint2 HG_Shared2    residual_error   0.06919    0.01333     0.13950
  2123   AJIVE_Joint2 HG_Shared2               mse   0.00328   -0.03293     0.03611
  2123   AJIVE_Joint2 HG_Shared2   mean_bias_error   0.03493    0.02299     0.04788
  2123   AJIVE_Joint2 HG_Shared2       slope_error  -0.01104   -0.04194     0.00170
  2123   AJIVE_Joint2 HG_Shared2 calibration_error   0.02389   -0.01027     0.04215
  2123   AJIVE_Joint2 HG_Shared2    residual_error  -0.02061   -0.05651     0.02104
  2123 VAE2_mean_loss HG_Shared2               mse   0.06152   -0.01158     0.13533
  2123 VAE2_mean_loss HG_Shared2   mean_bias_error   0.07948    0.05663     0.10551
  2123 VAE2_mean_loss HG_Shared2       slope_error   0.01420   -0.02650     0.05390
  2123 VAE2_mean_loss HG_Shared2 calibration_error   0.09369    0.04584     0.14237
  2123 VAE2_mean_loss HG_Shared2    residual_error  -0.03217   -0.08705     0.02216
  2123     Threshold2 HG_Shared2               mse   0.00051   -0.05107     0.04682
  2123     Threshold2 HG_Shared2   mean_bias_error  -0.01550   -0.03589    -0.00292
  2123     Threshold2 HG_Shared2       slope_error  -0.01062   -0.04451     0.00621
  2123     Threshold2 HG_Shared2 calibration_error  -0.02612   -0.05522    -0.00734
  2123     Threshold2 HG_Shared2    residual_error   0.02663   -0.02451     0.08485
```

## Change in relative performance across periods

```text
     reference            metric  estimate  psu_low95  psu_high95
         RawX7               mse   0.31479    0.13847     0.50628
         RawX7   mean_bias_error   0.20321    0.15431     0.25879
         RawX7       slope_error   0.08089    0.00288     0.17009
         RawX7 calibration_error   0.28410    0.17537     0.39407
         RawX7    residual_error   0.03069   -0.07191     0.16080
          PCA2               mse   0.13360    0.01773     0.25464
          PCA2   mean_bias_error   0.11116    0.07762     0.14322
          PCA2       slope_error   0.03464   -0.02075     0.08599
          PCA2 calibration_error   0.14580    0.08404     0.20164
          PCA2    residual_error  -0.01221   -0.10676     0.11899
        H_PCA2               mse   0.25402    0.10787     0.40662
        H_PCA2   mean_bias_error   0.18093    0.13426     0.22911
        H_PCA2       slope_error   0.05698   -0.00973     0.12301
        H_PCA2 calibration_error   0.23791    0.14910     0.32227
        H_PCA2    residual_error   0.01611   -0.08044     0.13407
   JIVE_Joint2               mse  -0.01665   -0.07657     0.04412
   JIVE_Joint2   mean_bias_error  -0.00901   -0.02242     0.00792
   JIVE_Joint2       slope_error   0.02349   -0.01773     0.06915
   JIVE_Joint2 calibration_error   0.01448   -0.02904     0.06879
   JIVE_Joint2    residual_error  -0.03113   -0.13443     0.06254
  AJIVE_Joint2               mse   0.05955    0.00185     0.11945
  AJIVE_Joint2   mean_bias_error   0.05249    0.03517     0.07243
  AJIVE_Joint2       slope_error   0.00995   -0.03153     0.04082
  AJIVE_Joint2 calibration_error   0.06243    0.01877     0.09840
  AJIVE_Joint2    residual_error  -0.00288   -0.06941     0.07265
VAE2_mean_loss               mse   0.15681    0.04118     0.27134
VAE2_mean_loss   mean_bias_error   0.09900    0.07001     0.12897
VAE2_mean_loss       slope_error   0.05333   -0.00445     0.11032
VAE2_mean_loss calibration_error   0.15233    0.08740     0.21588
VAE2_mean_loss    residual_error   0.00448   -0.08132     0.11755
    Threshold2               mse   0.03083   -0.04682     0.11293
    Threshold2   mean_bias_error  -0.00447   -0.02545     0.01230
    Threshold2       slope_error   0.02968   -0.02071     0.07138
    Threshold2 calibration_error   0.02520   -0.02302     0.07094
    Threshold2    residual_error   0.00563   -0.06529     0.10044
```

## Raw-marker error contribution by coordinate group

```text
cohort      group  estimate  psu_low95  psu_high95
  0708  intercept   0.00000    0.00000     0.00000
  0708 covariates   0.00064   -0.00277     0.00437
  0708     shared   0.01319   -0.00635     0.03499
  0708 discordant  -0.07713   -0.13892    -0.02253
  0708  H_private  -0.07703   -0.16943     0.00651
  0708  G_private  -0.00277   -0.00954     0.00442
  0708      total  -0.14311   -0.25868    -0.03789
  2123  intercept  -0.00000   -0.00000    -0.00000
  2123 covariates   0.00336   -0.00118     0.00759
  2123     shared  -0.05018   -0.07207    -0.03013
  2123 discordant   0.07911    0.03010     0.13830
  2123  H_private   0.13538    0.04213     0.23549
  2123  G_private   0.00401   -0.00318     0.01273
  2123      total   0.17168    0.04997     0.30613
```

## Change in coordinate-level error contribution

```text
     group  estimate  psu_low95  psu_high95
 intercept  -0.00000   -0.00000    -0.00000
covariates   0.00272   -0.00322     0.00825
    shared  -0.06337   -0.09502    -0.03291
discordant   0.15624    0.07774     0.24413
 H_private   0.21241    0.07692     0.35479
 G_private   0.00679   -0.00449     0.01805
     total   0.31479    0.13847     0.50628
```

## Distribution shift

```text
cohort     variable  weighted_mean  weighted_sd  mean_shift_in_reference_SD  sd_ratio_to_reference
  2123       LBXHGB       13.12802      1.18030                    -0.52666                1.04782
  2123     LBXRBCSI        4.53304      0.35427                    -0.02725                0.98884
  2123     LBXMCVSI       86.62752      6.19260                    -0.51013                1.05529
  2123       LBXRDW       13.88408      1.95666                     1.06183                1.73225
  2123        LBXGH        5.39582      0.88056                     0.11492                0.92767
  2123       LBXGLU       99.05348     22.04381                     0.10188                0.89921
  2123       LOG_IN        2.23164      0.70615                     0.35314                0.95988
  2123           S1        0.27702      1.07397                     0.59323                1.11168
  2123           S2        0.25216      0.75979                    -0.06638                0.89591
  2123           D1        0.39355      1.06510                     0.33772                1.23078
  2123           D2        0.42875      0.70930                     0.26256                0.81081
  2123           H1       -0.91191      1.05839                    -0.68639                1.10306
  2123           H2        0.03071      1.23499                     0.67030                1.62495
  2123           G1        0.23918      0.62661                     0.04748                0.82510
  2123          age       34.51026      8.16727                    -0.18060                0.99091
  2123          BMI       29.60062      8.93365                     0.18741                1.22582
  2123       income        3.01178      1.65449                    -0.08825                1.07960
  2123 log_ferritin        3.75621      0.86946                     0.05558                1.00390
```

## Interpretation boundaries

- Affine recalibration is fitted descriptively on the evaluation distribution; it is not a prospective correction.
- Residual error is the remaining variance after optimal affine recalibration.
- Lower residual error does not imply causality.
- Coordinate contributions depend on the specified reference parameterization.
- Contributions are signed algebraic terms, not independent biological effects.
- VAE metrics average the five fixed-seed decompositions, not predictions.
- Population shift, assay changes and selection cannot be causally separated by this analysis.
- PSU intervals condition on frozen model fits.
- Exploratory comparisons are unadjusted for multiplicity.
