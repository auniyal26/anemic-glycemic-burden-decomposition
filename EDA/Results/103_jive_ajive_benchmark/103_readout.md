# Script 103 — Head-to-head decomposition comparison

## Method integrity
- Official CRAN r.jive and original Python AJIVE (`jive`). Neither is approximated or relabeled.
- JIVE/AJIVE fixed to 2 joint + 2 A-individual + 1 G-individual = 5 factors.
- Frozen Z5 retains both shared, one discord, one A-private and G-private; choose discord and A-private by discovery-only X reconstruction (not phenotype). See manifest for indices.
- 7D RawX/FrozenZ/PCA are controls; not fair compression competitors for 5D.
- Common 2005–08 unweighted standardization for all unsupervised methods; later rows unchanged.
- Official JIVE/AJIVE native scores are regressed on discovery X to form an OUT-OF-SAMPLE adapter.
- Adapter R² is measured: a poor adapter invalidates external comparison.
- Training-only CV selects Ridge alpha; PHQ outcomes are never used to learn unsupervised scores.
- MSE differences computed on same patients and weighted point estimates.
- Later NHANES periods previously explored; evidence is exploratory, not untouched confirmation.

Failures/omissions: `{}`.

## Rank-matched physiology reconstruction (5D versus 5D)
```text
period           test_2009-2018  train_2005-2008  validate_2021-2023
method                                                              
AJIVE5_pyjive           0.71140          0.69925             0.73878
FrozenZ5_trainX         0.81289          0.81735             0.80132
JIVE5_rjive             0.97638          0.97484             0.98041
PCA5                    0.97648          0.97499             0.98048
```

## All-7D reconstruction controls
```text
period    test_2009-2018  train_2005-2008  validate_2021-2023
method                                                       
FrozenZ7             1.0              1.0                 1.0
PCA7                 1.0              1.0                 1.0
RawX7                1.0              1.0                 1.0
```

## Frozen biomarker-fingerprint transport (higher cosine = more similar)
```text
period           test_2009-2018  validate_2021-2023
method                                             
AJIVE5_pyjive           0.99317             0.98721
FrozenZ5_trainX         0.99798             0.99495
FrozenZ7                0.99682             0.99495
JIVE5_rjive             0.99808             0.99432
PCA5                    0.99248             0.99104
PCA7                    0.98288             0.98953
RawX7                   0.99428             0.99402
```

Note: This measures similarity of biomarker correlations for FROZEN axes. It is not independent re-discovery or biological identifiability.

## Training-CV phenotype weighted MSE (lower better)
```text
outcome           CogAff  PHQ_total  Somatic
method                                      
AJIVE5_pyjive    4.44387   12.37104  3.43554
FrozenZ5_trainX  4.45918   12.44655  3.45763
FrozenZ7         4.43036   12.29258  3.40968
JIVE5_rjive      4.44132   12.33146  3.41629
PCA5             4.44130   12.33123  3.41618
PCA7             4.43034   12.29262  3.40970
RawX7            4.43289   12.29848  3.41115
```

## Later phenotype weighted R² (higher better)
```text
period                     test_2009-2018  validate_2021-2023
outcome   method                                             
CogAff    AJIVE5_pyjive           0.05711            -0.04528
          FrozenZ5_trainX         0.05395            -0.06127
          FrozenZ7                0.05922            -0.03518
          JIVE5_rjive             0.06008            -0.03227
          PCA5                    0.06007            -0.03231
          PCA7                    0.05921            -0.03520
          RawX7                   0.05933            -0.03509
PHQ_total AJIVE5_pyjive           0.07847            -0.02042
          FrozenZ5_trainX         0.07376            -0.04184
          FrozenZ7                0.08094            -0.00537
          JIVE5_rjive             0.08297            -0.00123
          PCA5                    0.08293            -0.00121
          PCA7                    0.08093            -0.00539
          RawX7                   0.08099            -0.00507
Somatic   AJIVE5_pyjive           0.07736             0.02128
          FrozenZ5_trainX         0.07238             0.00267
          FrozenZ7                0.07895             0.03643
          JIVE5_rjive             0.08152             0.04041
          PCA5                    0.08145             0.04054
          PCA7                    0.07893             0.03642
          RawX7                   0.07883             0.03649
```

## Paired MSE gain over equal-dimension PCA baseline (positive = better)
```text
  outcome          method reference             period  delta_MSE_positive_better  descriptive_CI_low  descriptive_CI_high  interval_includes_zero
PHQ_total FrozenZ5_trainX      PCA5     test_2009-2018                   -0.14968            -0.28294             -0.04231                   False
PHQ_total FrozenZ5_trainX      PCA5 validate_2021-2023                   -0.79231            -1.06055             -0.56141                   False
PHQ_total     JIVE5_rjive      PCA5     test_2009-2018                    0.00074            -0.00111              0.00269                    True
PHQ_total     JIVE5_rjive      PCA5 validate_2021-2023                   -0.00033            -0.00543              0.00462                    True
PHQ_total   AJIVE5_pyjive      PCA5     test_2009-2018                   -0.07274            -0.17230              0.01348                    True
PHQ_total   AJIVE5_pyjive      PCA5 validate_2021-2023                   -0.37461            -0.53031             -0.19799                   False
  Somatic FrozenZ5_trainX      PCA5     test_2009-2018                   -0.03695            -0.06739             -0.00816                   False
  Somatic FrozenZ5_trainX      PCA5 validate_2021-2023                   -0.16472            -0.22979             -0.09530                   False
  Somatic     JIVE5_rjive      PCA5     test_2009-2018                    0.00029            -0.00018              0.00082                    True
  Somatic     JIVE5_rjive      PCA5 validate_2021-2023                   -0.00059            -0.00184              0.00059                    True
  Somatic   AJIVE5_pyjive      PCA5     test_2009-2018                   -0.01663            -0.04084              0.00385                    True
  Somatic   AJIVE5_pyjive      PCA5 validate_2021-2023                   -0.08377            -0.13207             -0.03077                   False
   CogAff FrozenZ5_trainX      PCA5     test_2009-2018                   -0.03677            -0.07471             -0.00669                   False
   CogAff FrozenZ5_trainX      PCA5 validate_2021-2023                   -0.22184            -0.30073             -0.14153                   False
   CogAff     JIVE5_rjive      PCA5     test_2009-2018                    0.00009            -0.00039              0.00054                    True
   CogAff     JIVE5_rjive      PCA5 validate_2021-2023                    0.00034            -0.00088              0.00217                    True
   CogAff   AJIVE5_pyjive      PCA5     test_2009-2018                   -0.01779            -0.04216              0.00817                    True
   CogAff   AJIVE5_pyjive      PCA5 validate_2021-2023                   -0.09935            -0.15495             -0.05118                   False
```

## Interpretation guardrails
- Do not equate an enforced private loading zero with a discovered independent biological factor.
- Shared/private scores differ by method definition; raw cross-block sensitivity is descriptive, not a mechanistic proof.
- Rank selection is fixed here; test alternative prespecified ranks and official automatic rank methods next.
- AJIVE/JIVE adapters are linear extensions of discovery scores. Refit ranks only in discovery in future analyses.
- Reconstruction R² is relative to discovery mean and normalized by frozen raw biomarker standardization.
- Bootstrap intervals resample participants rather than NHANES PSUs/strata; not design-based inference.
- Equal or better performance on these metrics alone cannot prove methodological novelty or clinical relevance.
- The best case is consistent improvement in matched-rank reconstruction and temporal prediction WITH credible physiological interpretability, then fully independent data.
- The null is also scientifically useful: JIVE/AJIVE may perform equally well or better, narrowing our actual contribution.

## Required next step for defensible novelty
Repeat JIVE/AJIVE rank settings, add population-shift stress tests and external MIDUS/MIDJA, and compare physiological interpretations with domain experts.
