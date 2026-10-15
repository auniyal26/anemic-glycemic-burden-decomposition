# Script 02 — MIDUS measurement and longitudinal coverage audit

**Scope:** Local SPSS reads of explicit candidate variables; aggregate-only outputs. No subject records exported.
**Status:** Feasibility audit only. No complete clinical harmonization, frozen NHANES transfer, modelling, or causal inference.
**Disclosure:** All computed table cells with N <= 10, including zero, are suppressed.

## Candidate measurements, by study

| Wave | Profile | Complete physiology | Physiology + CES-D | Physiology + BTACT |
|---|---|---:|---:|---:|
| M2 | G3 (HbA1c, glucose, insulin) | 1227 | 1219 | 965 |
| M3 | G3 (HbA1c, glucose, insulin) | 732 | 731 | 687 |
| MR1 | G3 (HbA1c, glucose, insulin) | 846 | 846 | 665 |
| MIDJA_1 | H3+A1c (Hb, RBC, MCV, HbA1c) | 382 | 353 | NO_PERFORMANCE_COGNITION_FILE |
| MIDJA_2 | H3+A1c (Hb, RBC, MCV, HbA1c) | 323 | 303 | NO_PERFORMANCE_COGNITION_FILE |

## Additional within-wave profiles

| Wave | Panel | Valid participant N |
|---|---|---:|
| M2 | G3+CESD_somatic+affect | 1222 |
| M3 | G3+CESD_somatic+affect | 731 |
| MR1 | G3+CESD_somatic+affect | 846 |
| MIDJA_1 | H3+A1c_adjusted | 382 |
| MIDJA_2 | H3+A1c_adjusted | 323 |
| M2 | G3+CESD_total+BTACT_comp | 962 |
| M3 | G3+CESD_total+BTACT_comp | 687 |
| MR1 | G3+CESD_total+BTACT_comp | 665 |

**MIDJA panel note:** H3+A1c counts reflect RBC numeric availability (units unverified), not certified physiological validity. Do not use these raw RBC values for modelling until unit harmonization.

## Longitudinal/linkage complete-case candidates

| Cohort | Prerequisites | Valid participant N |
|---|---|---:|
| CORE_M2 | G3 + same-wave CESD + BTACT | 962 |
| CORE_M3 | G3 + same-wave CESD + BTACT | 687 |
| CORE_M2_M3 | G3 at both biomarker waves | 669 |
| CORE_M2_M3 | M2 G3 -> M3 BTACT | 980 |
| CORE_M2_M3 | M2 G3 + M2 BTACT -> M3 BTACT | 784 |
| CORE_M2_M3 | M2 G3 + M2 CESD -> M3 BTACT | 975 |
| CORE_M2_M3 | G3 + BTACT measured at both waves | 509 |
| REFRESHER_MR1 | MR1 G3 + CESD + BTACT | 665 |
| REFRESHER_MR1_MR2 | MR1 G3 -> MR2 BTACT | 636 |
| REFRESHER_MR1_MR2 | MR1 G3 + MR1 BTACT -> MR2 BTACT | 523 |
| MIDJA_1 | H3+A1c + same-wave CESD | 353 |
| MIDJA_2 | H3+A1c + same-wave CESD | 303 |
| MIDJA_1_2 | H3+A1c measured at both waves | 239 |
| MIDJA_1_2 | H3+A1c + CESD at both waves | 211 |

## Important interpretation limits

- These are **valid-measurement ID intersections**, not an approved analytical cohort. No exclusions for medications, pregnancy, diagnoses, fasting verification, age, sampling design, or time gaps have been applied.
- Physiology uses provisional bounds where units are reasonably identifiable. Both the nonmissing count and the screened-valid count appear in `02_measurement_coverage.csv`.
- **MIDJA RBC correction:** the original 1-9 numeric screen was inappropriate without confirmation of Japanese RBC units. RBC is now audited on finite nonmissing data only, with **no unit conversion**. Japanese panel counts are **measurement-available**, NOT clinically validated; do not fit models using this raw RBC scale until laboratory units and sentinel values are verified.
- SPSS user-missing values are parsed as NaN by `pyreadstat` (`user_missing=False`); unflagged sentinel codes require separate codebook review.
- For insulin, positive screened values are **log-transform-compatible**; no log transformation or model transfer has occurred.
- US waves provide a possible glycemic triad, not the full frozen seven-marker NHANES A/G space. Japan provides a distinct provisional Hb/RBC/MCV/HbA1c panel, not an exact NHANES transfer.
- `CESD_total`, `CESD_somatic`, and `CESD_affect` are separate measures. CES-D is not PHQ-9; depressive affect is not interchangeable with PHQ-derived CogAff.
- BTACT score standardizations vary by wave and sample (national-only, full-sample, longitudinal). Explicit mappings are candidates only; audit the source scoring and measurement invariance before calculating change.
- MIDJA lacks downloaded performance-based cognitive assessments. A `Cognition Control` questionnaire response is not interchangeable with BTACT.
- Two observations per person do not in themselves identify a causal effect. Check visit dates and temporal ordering before building longitudinal equations.
- `HbA1c_adj` in MIDJA is a separately reported adjusted assay; no substitution is made without source documentation.

## Diagnostic details

- SPSS files staged: 15; recognized cohort/role files: 15.
- Mapped variables reviewed: 58; linkage candidates evaluated: 14.
- Files with duplicate IDs (unsafe for row-level intersections): 0.
- Errors: 0.

**Next scientific gate:** Confirm assay units, sentinel codes, questionnaire scoring and timing against codebooks before attempting an adapted decomposition or frozen transfer.
