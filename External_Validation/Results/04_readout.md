# Script 04 — US MIDUS prospective measurement/scoring audit

**Nature:** Local SPSS metadata + participant-level checks conducted in memory; only aggregate outputs saved.
**This is NOT a fitted model, proof of score invariance, or independent clinical assay certification.**
**Disclosure:** All cells N <= 10 (including 0) suppressed.

## Prospective sample sizes

| Cohort | G3 + baseline/follow-up composite + correct dates | + clinic age/sex | + prior survey education | + prior BMI |
|---|---:|---:|---:|---:|
| CORE_M2_TO_M3 | 785 | 785 | 784 | 757 |
| REFRESHER_MR1_TO_MR2 | 523 | 523 | 522 | 504 |

## Three-domain measurement checks

| Cohort | Composite N | Episodic memory N | Executive function N |
|---|---:|---:|---:|
| CORE_M2_TO_M3 | 785 | 844 | 848 |
| REFRESHER_MR1_TO_MR2 | 523 | 574 | 577 |

## Timing in analyzable composite-score sample

| Cohort | N | Baseline cognition → blood (median months) | Blood → follow-up cognition (median months) |
|---|---:|---:|---:|
| CORE_M2_TO_M3 | 785 | 22.000 | 89.000 |
| REFRESHER_MR1_TO_MR2 | 523 | 19.000 | 99.000 |

## Explicit source-score crosswalk (candidate)

- **Core:** baseline `B3TCOMPZ1`, `B3TEMZ1`, `B3TEFZ1` (MIDUS 2 national z scores); follow-up `C3TCOMP`, `C3TEM`, `C3TEF` (labels state standardized against MIDUS 2 national mean/SD).
- **Refresher:** baseline `RA3TCOMPZ1`, `RA3TEMZ1`, `RA3TEFZ1` (Refresher 1 national z scores); follow-up `RB3TCOMPZ_Long`, `RB3TEMZ_Long`, `RB3TEFZC_Long` (labels state Refresher 1 national mean/SD).
- Do NOT pool raw composite values from Core and Refresher: they reference DIFFERENT national cohorts.
- Longitudinal *score-reference* compatibility is suggested by labels, not proven. Source scoring manuals, item definition changes, invariance/retest effects still require review.

## Source marker meanings (not independent laboratory verification)

- HbA1c source label: percentage; fasting glucose: mg/dL; fasting insulin: uIU/mL.
- Positive insulin is required for log(Insulin). No log transform, scaling or outcome modelling was performed.
- `04_variable_scoring_qc.csv` documents numeric availability, selected quantiles, provisional extreme-value flags, and the SPSS source labels.
- SPSS-declared user-missing codes were parsed as missing; undocumented numeric sentinels remain possible.

## Methodological gates before the first model

1. **Clinical lab confirmation required:** check the official biomarker assay protocol, fasting compliance, laboratory harmonization, and exclusions.
2. **BTACT manual confirmation required:** verify composite construction and whether longitudinal reference-standardization and measurement invariance support comparison.
3. **Selection/attrition:** linked blood/cognition subsamples are not the population cohort; evaluate weighting and attrition strategy.
4. **Chronology:** only treat strictly `baseline cognition < blood < later cognition` at month resolution as temporally ordered; same-month ambiguous.
5. **Covariates:** survey education/BMI were collected before blood; clinic age is measured at blood visit. Their scientific role and missingness must be assessed.
6. **NHANES:** the US G3 panel is an ADAPTED glycemic study; do not call it a replication of the frozen 7-marker A/G decomposition.
7. **No causal claim:** temporally ordered associations do not identify causal physiological effects.

## Files

- `04_variable_scoring_qc.csv` (source labels, unit candidate, broad QC diagnostics)
- `04_prospective_sample_flow.csv` (aggregate cohort gates)
- `04_prospective_chronology.csv` (median gaps by cognitive domain)
