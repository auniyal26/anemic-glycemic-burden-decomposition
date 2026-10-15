# Script 03 — MIDUS/MIDJA data atlas and visit chronology

**Purpose:** Understand available marker domains, how much they overlap, and when assessments occurred.
**Not yet done:** Clinical assay/unit verification, missing-code validation, BTACT/CES-D invariance, NHANES transfer, modeling, or causal inference.
**Privacy:** Counts from 0 to 10 suppressed; no individual records or identifiers exported.

## What is measured (selected variables, not an exhaustive catalog)

The detailed `03_measurement_atlas.csv` gives the source label, numeric coverage and unvalidated distribution for every curated variable.

- **US MIDUS:** HbA1c, fasting glucose, fasting insulin, CRP, fibrinogen, creatinine, lipids; CES-D in biomarker files; BTACT in cognitive files.
- **Japan MIDJA:** Hb, RBC, MCV, hematocrit, MCH, MCHC, WBC, raw/adjusted HbA1c, CRP, fibrinogen, lipids; CES-D; no equivalent BTACT download in this batch.
- **Survey files (all waves):** sampled age, sex, education, BMI, and self-reported depressive history, where available. These are questionnaire fields, not validated confounder definitions.
- Do **not** presume shared measurement units from shared variable names; verify original and adjusted HbA1c independently.

## Panel availability — numeric-only (not screened valid)

| Dataset | Proposed marker panel | N with numeric measurements | Source N |
|---|---|---:|---:|
| M2_BIO | G3 | 1228 | 1255 |
| M2_BIO | G3+inflammation | 1224 | 1255 |
| M2_BIO | G3+metabolic | 1226 | 1255 |
| M2_BIO | G3+CESD | 1220 | 1255 |
| M3_BIO | G3 | 733 | 747 |
| M3_BIO | G3+inflammation | 731 | 747 |
| M3_BIO | G3+metabolic | 731 | 747 |
| M3_BIO | G3+CESD | 732 | 747 |
| MR1_BIO | G3 | 846 | 863 |
| MR1_BIO | G3+inflammation | 844 | 863 |
| MR1_BIO | G3+metabolic | 845 | 863 |
| MR1_BIO | G3+CESD | 846 | 863 |
| MIDJA_1_BIO | H3+HbA1c_raw | 382 | 382 |
| MIDJA_1_BIO | H7+HbA1c_raw | 382 | 382 |
| MIDJA_1_BIO | H3+HbA1c_raw+CESD | 353 | 382 |
| MIDJA_1_BIO | H7+HbA1c_raw+inflammation | 382 | 382 |
| MIDJA_2_BIO | H3+HbA1c_raw | 326 | 328 |
| MIDJA_2_BIO | H7+HbA1c_raw | 326 | 328 |
| MIDJA_2_BIO | H3+HbA1c_raw+CESD | 306 | 328 |
| MIDJA_2_BIO | H7+HbA1c_raw+inflammation | 325 | 328 |

## Visit chronology (source → target)

Positive date gap means source was assessed earlier than target. Same month does not establish ordering.

| Comparison | Linked IDs | Dated pairs | Source earlier | Same month | Source later | Median target−source (months) |
|---|---:|---:|---:|---:|---:|---:|
| Core: M2 survey vs M2 biomarker | 1054 | 1054 | 1054 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 26 |
| Core: M3 survey vs M3 biomarker | 644 | 644 | 644 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 63 |
| Refresher: MR1 survey vs MR1 biomarker | 746 | 746 | 746 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 22 |
| Core: M2 biomarker vs M2 cognition | 1152 | 1152 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 1152 | -21 |
| Core: M2 biomarker vs M3 cognition | 1049 | 1049 | 1049 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 92 |
| Core: M3 biomarker vs M3 cognition | 731 | 731 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 731 | -56 |
| Core: M2 biomarker vs M3 biomarker | 687 | 687 | 687 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 146 |
| Core: M2 cognition vs M3 cognition | 2940 | 2940 | 2940 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 111 |
| Refresher: MR1 biomarker vs MR1 cognition | 801 | 792 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 792 | -18 |
| Refresher: MR1 biomarker vs MR2 cognition | 689 | 689 | 689 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 102 |
| Refresher: MR1 cognition vs MR2 cognition | 1602 | 1588 | 1588 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 126 |
| Japan: MIDJA1 biomarker vs MIDJA2 biomarker | 243 | 243 | 243 | SUPPRESSED_0_TO_10 | SUPPRESSED_0_TO_10 | 54 |

## Joint date + measurement diagnostic

- **Core: M2 survey vs M2 biomarker:** NOT_EVALUATED.
- **Core: M3 survey vs M3 biomarker:** NOT_EVALUATED.
- **Refresher: MR1 survey vs MR1 biomarker:** NOT_EVALUATED.
- **Core: M2 biomarker vs M2 cognition:** 966 measured-and-dated; SUPPRESSED_0_TO_10 source-before-target.
- **Core: M2 biomarker vs M3 cognition:** 981 measured-and-dated; 981 source-before-target.
- **Core: M3 biomarker vs M3 cognition:** 688 measured-and-dated; SUPPRESSED_0_TO_10 source-before-target.
- **Core: M2 biomarker vs M3 biomarker:** 670 measured-and-dated; 670 source-before-target.
- **Core: M2 cognition vs M3 cognition:** NOT_EVALUATED.
- **Refresher: MR1 biomarker vs MR1 cognition:** 665 measured-and-dated; SUPPRESSED_0_TO_10 source-before-target.
- **Refresher: MR1 biomarker vs MR2 cognition:** 636 measured-and-dated; 636 source-before-target.
- **Refresher: MR1 cognition vs MR2 cognition:** NOT_EVALUATED.
- **Japan: MIDJA1 biomarker vs MIDJA2 biomarker:** 241 measured-and-dated; 241 source-before-target.

## Hypotheses to choose from AFTER this audit

1. US glycemic physiological signatures and later BTACT, controlling for baseline BTACT, age and measured confounders (temporal ordering required).
2. Stability of the US 3-marker glycemic representation across M2/M3 and Refresher; report construct changes rather than relabeling as the full A/G decomposition.
3. Japan Hb/RBC/MCV/HbA1c profiles and CES-D, within-person changes; calibrate laboratory units and inspect collinearity before decomposition.
4. Sensitivity to CRP/fibrinogen or lipid markers as prespecified contextual variables, not an unbounded feature search.

## Decision gates

- Verify full metadata, laboratory units, assay harmonization and missingness (especially Japanese RBC and HbA1c adjusted vs raw).
- Check CES-D subscale scoring and whether BTACT standardizations are on a common longitudinal reference.
- Study attrition/selection, causal ordering and potential confounders; no causal conclusions from pairwise correlations.
- Do not train or evaluate on wave M3 physiological measurements to predict earlier M3 cognition; use participant-specific chronology.
- Preserve all frozen NHANES maps; adapted US and Japan representations must be labeled separately.

## Outputs

- `03_readout.md`
- `03_measurement_atlas.csv`
- `03_panel_feasibility.csv`
- `03_visit_chronology.csv`
