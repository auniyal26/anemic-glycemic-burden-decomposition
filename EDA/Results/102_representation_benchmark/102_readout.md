# Script 102 — NHANES representation benchmark

## Protocol
- Genuine Script-98 common-complete cohort; original PHQ item definitions and covariates.
- Original outcome-independent frozen Z is NOT refitted or altered.
- Baseline PCA/CCA fitted on 2005–08 physiology only; no target data used in fitting.
- Penalization alpha chosen exclusively with training-cycle-stratified five-fold CV (or --folds).
- Fixed representations and regression coefficients applied to 2009–18 and 2021–23.
- These later periods have been studied before: exploratory temporal stress tests, NOT untouched confirmation.
- NHANES cycles contain different individuals; results are NOT within-person longitudinal evidence.
- R² values are survey-weighted point estimates, without complex survey design variance estimation.

Observed counts: `{"train_2005-2008": 3815, "test_2009-2018": 9893, "validate_2021-2023": 2247}`.
Independent discovery CCA correlations: `[0.31327, 0.25645, 0.02705]`.

## Reconstruction (test and validation)
```text
period                   test_2009-2018  validate_2021-2023
method                                                     
BlockPCA6                       0.99612             0.99740
FrozenZ4_shared_discord         0.61246             0.59760
FrozenZ7                        1.00000             1.00000
IndependentCCA7                 1.00000             1.00000
PCA4                            0.85848             0.83593
PCA5                            0.97132             0.97450
PCA7                            1.00000             1.00000
RawX7                           1.00000             1.00000
```

## Phenotype prediction on temporally held-out periods
```text
period                             test_2009-2018  validate_2021-2023
outcome   method                                                     
CogAff    BlockPCA6                       0.06004            -0.03251
          CovariatesOnly                  0.05695            -0.05679
          FrozenZ4_shared_discord         0.05815            -0.05052
          FrozenZ7                        0.05922            -0.03518
          Hb_HbA1c                        0.05734            -0.05167
          IndependentCCA7                 0.05922            -0.03517
          PCA4                            0.05844            -0.04175
          PCA5                            0.06007            -0.03235
          PCA7                            0.05922            -0.03517
          RawX7                           0.05933            -0.03509
PHQ_total BlockPCA6                       0.08290            -0.00134
          CovariatesOnly                  0.07702            -0.03702
          FrozenZ4_shared_discord         0.08032            -0.02431
          FrozenZ7                        0.08094            -0.00537
          Hb_HbA1c                        0.07901            -0.02842
          IndependentCCA7                 0.08094            -0.00536
          PCA4                            0.08067            -0.01538
          PCA5                            0.08294            -0.00123
          PCA7                            0.08094            -0.00537
          RawX7                           0.08239            -0.00319
Somatic   BlockPCA6                       0.08143             0.04067
          CovariatesOnly                  0.07438             0.00519
          FrozenZ4_shared_discord         0.07945             0.02172
          FrozenZ7                        0.07895             0.03643
          Hb_HbA1c                        0.07796             0.01499
          IndependentCCA7                 0.07894             0.03644
          PCA4                            0.07947             0.02634
          PCA5                            0.08147             0.04058
          PCA7                            0.07894             0.03644
          RawX7                           0.08050             0.03854
```

## Training-only cross-validation
```text
outcome                   CogAff  PHQ_total  Somatic
method                                              
BlockPCA6                4.43722   12.33069  3.41924
CovariatesOnly           4.45911   12.46448  3.45983
FrozenZ4_shared_discord  4.46408   12.46192  3.45602
FrozenZ7                 4.42488   12.29818  3.41591
Hb_HbA1c                 4.44401   12.39190  3.44056
IndependentCCA7          4.42487   12.29812  3.41589
PCA4                     4.43399   12.32521  3.41884
PCA5                     4.43548   12.32629  3.41821
PCA7                     4.42486   12.29811  3.41590
RawX7                    4.42735   12.30361  3.41540
```

## What the numbers cannot establish
- A full rank 7->7 linear transform is not intrinsically more informative than the original 7 biomarkers.
- Equal or slightly better held-out Ridge scores can reflect regularization geometry or model selection, not new information.
- A high reconstruction R² for seven full-rank features is mathematically expected, not proof of biological identifiability.
- Private block zeros are structural properties defined by the transformation, not independently verified physiological mechanisms.
- Drift scores are coordinate dependent, descriptive, and must not be interpreted as formal subspace alignment.
- This script does not benchmark true JIVE, AJIVE, MOFA+, external MIDUS/MIDJA, or clinical utility; they require distinct protocols.
- No causal inference, mechanistic identification, untouched external replication, or established framework novelty follows from this run.

## Decision rule
Judge the method on matched 4D/5D compression, held-out phenotype metrics, and independent interpretation; treat 7D reconstruction as a sanity check. If IndependentCCA7 and FrozenZ7 behave nearly identically, the distinguishing methodological claim remains unproven.
