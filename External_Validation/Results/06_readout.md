# Script 06 — Executive-function transfer: does frozen PC2 add beyond HbA1c?

**Primary comparison:** the already-tested Core-trained `NHANES_PC2` model versus the Core-trained `HbA1c` model, evaluated in the same Refresher participants.
**Status:** post-hoc validation/sensitivity analysis, not an independent confirmation of a prespecified hypothesis.
**Data privacy:** no subject IDs, individual rows, fitted values or residuals exported; N <= 10 suppressed.

## Locked executive outcome: paired comparisons

Positive ΔMSE favors the **comparison** model. Intervals are descriptive participant-paired bootstrap percentiles; they do not account for cohort sampling, repeated model fitting, confounder bias, or multiple comparisons.

| Comparison (external Refresher) | N | ΔMSE | 95% bootstrap CI | Centered-error Δ | Squared-bias Δ |
|---|---:|---:|---|---:|---:|
| HbA1c → NHANES_PC2 | 576 | 0.00093 | [-0.00055, 0.00251] | -0.00032 | 0.00126 |
| covariates → NHANES_PC2 | 576 | 0.00639 | [0.00291, 0.01015] | 0.00235 | 0.00404 |
| covariates → HbA1c | 576 | 0.00546 | [0.00240, 0.00867] | 0.00267 | 0.00279 |

**Exact identity:** ΔMSE = Δ(centered prediction-error variance) + Δ(squared mean prediction bias).
Centered-error MSE is an **outcome-informed, hypothetical perfect mean-bias correction** and is a diagnostic, not prospective deployable performance.

## Score-reference / outcome-definition sensitivity (executive)

| Score treatment | ΔMSE PC2 versus HbA1c | 95% paired CI | Centered-error Δ |
|---|---:|---|---:|
| native_frozen | 0.00093 | [-0.00055, 0.00251] | -0.00032 |
| baseline_anchored | 0.00149 | [-0.00058, 0.00366] | -0.00047 |
| within_person_change | 0.00105 | [-0.00059, 0.00257] | -0.00032 |

**Interpretation of scenarios:**
- `native_frozen`: exactly reproduces Script 05's cohort-native BTACT source scoring and entirely frozen Core→Refresher model.
- `baseline_anchored`: scores in each cohort are expressed relative to that cohort's *earlier* BTACT mean/SD; Refresher predictor-baseline distribution is used (without later outcomes). This is cohort-baseline adaptation, not a strictly frozen single-person deployment.
- `within_person_change`: outcome is follow-up minus earlier BTACT, retaining each study's native standardization; baseline BTACT stays a covariate.
- Different BTACT national reference populations, practice/retest effects and unknown measurement invariance remain unresolved by any of these checks. Do NOT claim proven cross-cohort phenotype equivalence.

## Domain-specificity check (native locked transfer)

| Domain | Refresher N | ΔMSE PC2 versus HbA1c | 95% paired CI |
|---|---:|---:|---|
| comp | 522 | -0.00054 | [-0.00163, 0.00066] |
| memory | 573 | -0.00284 | [-0.00597, 0.00032] |
| executive | 576 | 0.00093 | [-0.00055, 0.00251] |

## What the experiment cannot establish

1. Does not establish causal physiology→cognition effects or clinical usefulness; the first glycemic assessment followed the earlier BTACT assessment.
2. No MIDUS lab-assay cross-calibration, sampling/attrition weighting, medication/fasting validation, or complete psychometric invariance test.
3. The two-PC model was selected for this comparison **after inspecting Script 05's executive result**. A positive paired bootstrap does not become independent external confirmation.
4. Uses the NEW NHANES-frozen G3 PCA benchmark, not the original full seven-marker A/G shared–discordant map.
5. Does not use Refresher follow-up outcomes to fit predictor coefficients, select a model, or choose bootstrap settings; within-person-change and baseline-anchored scores are sensitivity definitions.
6. Refresher and Core remain different population and scoring references; re-anchoring is diagnostic, not a formal crosswalk.

## Reproducibility

- Script 05 outputs exactly reproduced on native-scenario Core CV and locked Refresher, for all three domains/models checked.
- Bootstrap replicates: 3000; seed: 20261015; no participant-level output.
- NHANES frozen reference SHA256: `92b1266d63e8841f173683b01a42a5051db6434c03fd3ecc895b872af2a92962`.

## Files

- `06_readout.md` (this report)
- `06_paired_model_comparisons.csv` (all three external and Core CV models, domain/score sensitivity and paired intervals)
- `06_error_decomposition.csv` (model-specific bias and centered errors; no participant rows)
- `06_manifest.json` (reproducibility details)

**Decision gate:** Judge added information over HbA1c only after the paired difference, centered-error contribution and score-reference sensitivities are read together. No result is interpreted as a novel external confirmation solely because an interval excludes zero.
