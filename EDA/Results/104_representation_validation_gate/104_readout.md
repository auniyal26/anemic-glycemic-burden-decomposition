# Script 104 — Representation validation gate

## What was actually run

- Cohort sizes (sampled without outcome information): `{'0506': 1705, '0708': 1800, '0918': 1800, '2123': 1800}`.
- Original Script-72 map was **not** changed. Its CCA recipe was **refit independently** per cohort, on the matched Script-98 sample.
- JIVE/AJIVE were fit using official packages; training-only linear projections allow fixed-anchor comparison.
- Every subspace was evaluated on exactly the same reference participants (0506).
- Whole 7D subspace stability deliberately **not** scored (trivially identical for invertible maps).
- Original Script 103 showed our 5D reduction underperformed JIVE/PCA compression.

## Independent group rediscovery (principal subspace similarity; 1=perfect)

```text
cohort_rediscovered        0708     0918     2123
method  group                                    
AJIVE5  G_Individual1   0.89904  0.97110  0.98401
        H_Individual2   0.52578  0.51194  0.69154
        Joint2          0.96356  0.88857  0.77573
HG_CCA7 CoupledShared2  0.96496  0.95784  0.98302
        Discordant2     0.94378  0.93971  0.97323
        G_Isolated1     0.99673  0.94795  0.85002
        H_Isolated2     0.97793  0.92004  0.92464
JIVE5   G_Individual1   0.86327  0.90333  0.97017
        H_Individual2   0.58093  0.66025  0.62415
        Joint2          0.54350  0.74794  0.75589
PCA5    Leading2        0.92010  0.87874  0.95751
        Residual1       0.94848  0.92420  0.87586
        Residual2       0.85792  0.81497  0.90589
```

## Individual axis rediscovery (Hungarian matched absolute correlation)

```text
cohort_rediscovered        0708     0918     2123
method  group                                    
AJIVE5  G_Individual1   0.94818  0.98545  0.99197
        H_Individual2   0.59989  0.57860  0.81344
        Joint2          0.97866  0.88305  0.83621
HG_CCA7 CoupledShared2  0.83227  0.97296  0.97971
        Discordant2     0.81467  0.96456  0.97397
        G_Isolated1     0.99836  0.97363  0.92197
        H_Isolated2     0.97847  0.87014  0.94109
JIVE5   G_Individual1   0.92912  0.95044  0.98497
        H_Individual2   0.62138  0.77573  0.75540
        Joint2          0.62955  0.83494  0.85300
PCA5    Leading2        0.92941  0.89941  0.90125
        Residual1       0.97390  0.96135  0.93587
        Residual2       0.91700  0.88812  0.92469
```

## Cross-block correlations versus G-row permutation null

```text
cohort  axis     rho  permutation_p  null95_rho
  0506     1 0.36524          0.005     0.10659
  0506     2 0.23578          0.005     0.06913
  0708     1 0.30654          0.005     0.11143
  0708     2 0.22650          0.005     0.06858
  0918     1 0.25314          0.005     0.11727
  0918     2 0.19657          0.005     0.08029
  2123     1 0.37880          0.005     0.12032
  2123     2 0.21502          0.005     0.07456
```

## Claim ledger (read before drafting conclusions)

| Proposed claim | Status | Grounds |
|---|---|---|
| Exact 7D reconstruction | SUPPORTED ALREADY | Scripts 72-73 and 103; full-rank linear map. Mathematical, not novel. |
| Cross-block nonrandom correlation | EVALUATE 104 | See 104_crossblock_permutation.csv. Noncausal coupling. |
| Independent rediscovery by cohort | EVALUATE 104 | See 104_independent_rediscovery.csv. Statistical subspace/axis, not biological identification. |
| Superior 5D reconstruction to PCA/JIVE | NOT SUPPORTED | 103: Z5 around .80-.82 vs PCA/JIVE around .97-.98. |
| Superior 7D phenotype prediction | NOT SUPPORTED | 103: practically tied with PCA7/RawX7. |
| More biological meaning than JIVE/AJIVE | OPEN | Needs independent physiological anchors, expert review, falsification. |
| Longitudinal subject-level evidence | OPEN | NHANES time transfer is repeated cross-sectional; use true repeated-person data. |
| Generalisable validation framework | PROPOSED | Only credible after independent multicohort demonstrations + reproducible criteria. |

## What this can and cannot establish

- Statistical subspace reproducibility is a valuable diagnostic, **not** evidence of unique biological source identification.
- Shared/isolated partitions are imposed by construction. Compare JIVE and AJIVE as distinct models, not identical biological decompositions.
- Comparisons depend on fixed ranks (JIVE/AJIVE 2 joint, 2 H individual, 1 G individual) and their affine score adapters.
- Positive cross-block association can be caused by shared confounders or biomarker measurement effects; not causal.
- Disjoint periods mean different people; within-person longitudinal claims remain unsupported.
- 0506 and 0708 are independent patient groups here, but the ORIGINAL 72 map had been learned using 2005–08 data.
- Do not turn high similarities into significance claims: these estimates are descriptive and lack survey-design uncertainty.
- The strong paper angle is a transparently reported **validation protocol plus empirical case study**, not automatic algorithm superiority.

## Closure decision

If component rediscovery is strong across 0918/2123, report it as an exploratory robustness result. 
If it fails, report lack of generality. In either case, proceed with the review/framework draft while marking
external biological anchoring, independent dataset and true longitudinal validation as open gates.
