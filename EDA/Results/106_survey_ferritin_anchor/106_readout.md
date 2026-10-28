# Script 106 — Survey-aware structural uncertainty and ferritin anchoring

## Reproducibility: shared and discordant subspaces

Full NHANES masked PSU/stratum frame, fasting subsample weights, and rescaled PSU bootstrap; intervals approximate and exploratory.

```text
reference target       group  observed_similarity  psu_rescaled_low_2p5  psu_rescaled_high_97p5  bootstrap_sd  bootstrap_reps  stratum_psu_design_used
     0506   0708     Shared2               0.9663                0.8297                  0.9818        0.0424             199                     True
     0506   0708 Discordant2               0.9464                0.7551                  0.9643        0.0563             199                     True
     0506   0708 H_Isolated2               0.9862                0.8458                  0.9892        0.0394             199                     True
     0506   0708 G_Isolated1               0.9964                0.8477                  0.9973        0.0424             199                     True
     0506   2123     Shared2               0.9688                0.8247                  0.9879        0.0407             199                     True
     0506   2123 Discordant2               0.9544                0.7648                  0.9753        0.0548             199                     True
     0506   2123 H_Isolated2               0.9347                0.7912                  0.9832        0.0563             199                     True
     0506   2123 G_Isolated1               0.7872                0.4177                  0.9970        0.1673             199                     True
```

## Ferritin anchor: external assay, internal NHANES dataset

Frozen 0506-only ridge predictor (not a ferritin-trained physiological decomposition).
Positive delta MSE indicates benefit over covariates-only on log ferritin.

```text
cohort         0506    0708    2123
method                             
Covariates   0.0000  0.0000  0.0000
Discordant2  0.1134  0.1597  0.0463
G3           0.0095 -0.0046 -0.0061
G_Isolated1  0.0066  0.0063 -0.0061
H4           0.2232  0.2835 -0.0850
H_Isolated2  0.0862  0.1067  0.0367
PCA2         0.1777  0.2463  0.0536
RawX7        0.2335  0.2865 -0.0866
Shared2      0.0614  0.1588  0.0869
```

## PSU resampling of ferritin transfer (heldout uncertainty only)

```text
cohort      method  delta_mse_vs_covariates  psu_rescaled_low_2p5  psu_rescaled_high_97p5  bootstrap_sd  bootstrap_reps                                                     note
  0708 H_Isolated2                   0.1067                0.0362                  0.1823        0.0428             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  0708     Shared2                   0.1588                0.1001                  0.2264        0.0339             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  0708 Discordant2                   0.1597                0.0641                  0.2539        0.0526             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  0708 G_Isolated1                   0.0063               -0.0064                  0.0184        0.0079             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  0708          H4                   0.2835                0.1418                  0.4388        0.0927             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  0708          G3                  -0.0046               -0.0203                  0.0092        0.0087             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  0708        PCA2                   0.2463                0.1265                  0.3822        0.0722             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  0708       RawX7                   0.2865                0.1385                  0.4466        0.0948             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123 H_Isolated2                   0.0367               -0.0420                  0.1223        0.0441             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123     Shared2                   0.0869                0.0058                  0.1586        0.0407             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123 Discordant2                   0.0463               -0.0280                  0.1309        0.0418             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123 G_Isolated1                  -0.0061               -0.0202                  0.0072        0.0077             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123          H4                  -0.0850               -0.2229                  0.0287        0.0594             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123          G3                  -0.0061               -0.0310                  0.0125        0.0111             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123        PCA2                   0.0536               -0.0430                  0.1434        0.0508             199 Holdout PSU uncertainty only; frozen discovery fit fixed
  2123       RawX7                  -0.0866               -0.2385                  0.0324        0.0627             199 Holdout PSU uncertainty only; frozen discovery fit fixed
```

## Frozen PHQ transfer with masked PSU resampling

```text
cohort     method  point_frozen_phq_r2  psu_rescaled_low_2p5  psu_rescaled_high_97p5  bootstrap_reps                                                         note
  0708 Covariates               0.0757                0.0504                  0.1111             199 Holdout PSU resampling; reference map and coefficients fixed
  0708      RawX7               0.0857                0.0617                  0.1128             199 Holdout PSU resampling; reference map and coefficients fixed
  0708   H_G_CCA7               0.0860                0.0622                  0.1132             199 Holdout PSU resampling; reference map and coefficients fixed
  2123 Covariates              -0.0714               -0.1123                 -0.0362             199 Holdout PSU resampling; reference map and coefficients fixed
  2123      RawX7              -0.0208               -0.0545                  0.0121             199 Holdout PSU resampling; reference map and coefficients fixed
  2123   H_G_CCA7              -0.0206               -0.0543                  0.0123             199 Holdout PSU resampling; reference map and coefficients fixed
```

## Claim boundaries

- Ferritin is an independent laboratory anchor, but the same NHANES survey system; NOT external-population validation.
- Early ferritin sampling targets primarily girls/women and children; adult comparison is women 20-49.
- Ferritin reflects iron storage and inflammation; a link is not unique biological source identification.
- Model transfer across 2021-23 is sensitive to assay change and population/weighting differences.
- Full-rank RawX7 and CCA7 contain the same linear information. CCA gains in ridge are regularization/conditioning dependent.
- Complete-case restriction and earlier NHANES inspection persist; results exploratory and not clinical validation.
- Rescaled PSU intervals are approximate; survey methodologist should review NCHS weights and variance approach.
- The 0506 training and 0708 test inherit earlier 2005-08 categorical encoding from Script98; 0708 is NOT fully untouched preprocessing.
- An independent *population* with seven harmonized biomarkers remains an unmet publication requirement for strong transport claims.
