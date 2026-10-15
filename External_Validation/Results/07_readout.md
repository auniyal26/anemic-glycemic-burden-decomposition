# Script 07 — Hematology-associated HbA1c/glucose discordance

**Question:** Do Hb/RBC/MCV/RDW add transferable information for measured HbA1c after conditioning on fasting glucose, log-insulin and basic covariates?

**Not causal identification.** Residual HbA1c after fasting-glucose adjustment is a proxy for *discordance*, not verified glucose-independent HbA1c bias.

## Locked experiment

- Discovery: NHANES 2005–08; outcomes not used to choose variables.
- Baseline: fasting glucose, centered glucose squared, log-insulin, age, sex, BMI and estimated GFR.
- Augmented: same variables plus Hb, RBC count, MCV and RDW.
- Fixed Ridge alpha=20; weighted fitting; five-fold discovery-only CV; no holdout refitting.
- All comparisons are matched-complete-case and use frozen source `73_frozen_phenotype_map_input.csv`.

## Discovery five-fold CV (within same participants)

| Model | Weighted CV MSE | Weighted CV R² |
|---|---:|---:|
| Baseline | 0.259577 | 0.67703 |
| + Hematology | 0.242898 | 0.69778 |

## Frozen temporal tests

Positive ΔMSE favors adding hematology. 95% intervals are **descriptive case bootstraps only**, not NHANES survey-design inferential intervals.

| Period | Complete N | Baseline MSE | +H MSE | ΔMSE | Descriptive 95% CI |
|---|---:|---:|---:|---:|---|
| discovery_2005_08 | 4413 | 0.25491 | 0.23790 | +0.01701 | [+0.01310, +0.02103] |
| holdout_2009_18 | 12065 | 0.28216 | 0.27036 | +0.01180 | [+0.00853, +0.01514] |
| holdout_2021_23 | 2959 | 0.33939 | 0.31372 | +0.02568 | [+0.01417, +0.03821] |

## Contextual expansion

Optional labs are evaluated only when the **exact source columns** exist with sufficient data in ALL periods. Otherwise the experiment is explicitly skipped.

- CRP (LBXCRP): PRESENT_BUT_INSUFFICIENT_COVERAGE
- CREATININE (LBXSCR): ABSENT_FROM_SCRIPT73
- TOTAL_CHOLESTEROL (LBXTC): ABSENT_FROM_SCRIPT73
- HDL (LBDHDD): ABSENT_FROM_SCRIPT73
- TRIGLYCERIDES (LBXTR): PRESENT_BUT_INSUFFICIENT_COVERAGE

## Critical limitations

1. **Fasting glucose is not mean glucose over the preceding 2–3 months**; HbA1c discrepancy cannot be attributed to RBC lifespan or an HbA1c assay effect on this evidence alone.
2. Associations can arise from iron deficiency, kidney disease, medications, diabetes, assay drift, inflammation, age and selection; physiology is observational.
3. Survey-weighted point metrics and naive participant-bootstrap intervals **do not** account for PSU/stratum sampling. Use design-based methods before publication-level inference.
4. Age, eGFR and BMI adjustment cannot remove all confounding. Hematological indices are correlated and non-interventional.
5. The source file may be a selected analytical extract rather than the full fasting NHANES population; sample restriction needs external audit.
6. MIDJA in this batch has HbA1c/CBC but no matched fasting glucose; it cannot confirm glucose–HbA1c discordance. Unit and assay documentation remains outstanding.
7. This is a new diagnostic study, not a reconstruction of the original frozen 7D NHANES physiological decomposition.

## Outputs

- `07_frozen_transfer_metrics.csv` (including per-cycle diagnostics)
- `07_hematology_coefficients.csv` (conditional, ridge-shrunk associations)
- `07_context_marker_audit.csv`, `07_context_sensitivity.csv`
- `07_manifest.json`

**Scientific gate:** If the frozen hematology increment fails in later periods, report non-transfer and reassess assays/phenotype before adding model complexity.
