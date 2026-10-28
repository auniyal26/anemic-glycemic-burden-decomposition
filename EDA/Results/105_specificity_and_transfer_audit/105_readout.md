# Script 105 — Falsification and frozen-transfer audit

This script does NOT demonstrate biological source identifiability or superior prediction.

## Conditional null — observed rediscovery versus age/sex-preserving placebo

The crucial comparison is observed similarity vs. the 95th percentile of conditioned-null similarity.
A small/null margin means high absolute similarity by itself is not informative.

```text
cohort       group  observed_similarity  conditioned_null_high95  specificity_margin_vs_null95  exploratory_permutation_tail_p
  0708     Shared2              0.96627                  0.51051                       0.45577                           0.005
  0708 Discordant2              0.94417                  0.46435                       0.47983                           0.005
  0708 H_Isolated2              0.97453                  0.58337                       0.39116                           0.005
  0708 G_Isolated1              0.99728                  0.35715                       0.64013                           0.005
  0918     Shared2              0.96744                  0.51040                       0.45704                           0.005
  0918 Discordant2              0.94866                  0.48223                       0.46642                           0.005
  0918 H_Isolated2              0.92993                  0.64888                       0.28105                           0.010
  0918 G_Isolated1              0.89257                  0.51146                       0.38111                           0.010
  2123     Shared2              0.95945                  0.56195                       0.39750                           0.005
  2123 Discordant2              0.94192                  0.49014                       0.45178                           0.005
  2123 H_Isolated2              0.94279                  0.69509                       0.24770                           0.005
  2123 G_Isolated1              0.73986                  0.73760                       0.00226                           0.055
```

## Frozen outcome models (NOT untouched test cohorts)

Every model is fitted on 0506 only. Compare covariate-only vs full-rank RawX7/PCA7/CCA7.
```text
cohort                  0506    0708    0918    2123
outcome   method                                    
CogAff    Covariates  0.0488  0.0475  0.0276 -0.0610
          H_G_CCA7    0.0607  0.0575  0.0412 -0.0373
          PCA7        0.0599  0.0554  0.0409 -0.0393
          RawX7       0.0599  0.0554  0.0409 -0.0393
PHQ_total Covariates  0.0699  0.0701  0.0454 -0.0460
          H_G_CCA7    0.0784  0.0705  0.0523 -0.0102
          PCA7        0.0773  0.0713  0.0520 -0.0120
          RawX7       0.0773  0.0713  0.0520 -0.0120
Somatic   Covariates  0.0639  0.0508  0.0480 -0.0070
          H_G_CCA7    0.0888  0.0573  0.0621  0.0511
          PCA7        0.0885  0.0618  0.0626  0.0512
          RawX7       0.0885  0.0618  0.0626  0.0512
```

## Interpretation rules

- Observed stability exceeding an age/sex-conditioned placebo is evidence of a nontrivial STATISTICAL grouping, not biological identification.
- Null preserving coarse age and sex does not remove continuous age, metabolic, renal or measurement confounding.
- Participant resampling is NOT NHANES design-based uncertainty. Obtain strata/PSU for publication-grade confidence intervals.
- If all 7D models deteriorate together, phenotype transfer failure is not unique to the CCA representation.
- Score rotations of equal full dimension cannot increase information; regularized model comparisons may vary numerically.
- A fully independent cohort and independent physiological anchors are still required to establish biological relevance.
- Previously inspected future cohorts make these analyses exploratory; no confirmatory p-value claims.

## Scientific outcome

The method is useful only if this falsification test adds discriminatory evidence beyond an unconditional correlation or a trivial full-rank reconstruction.
