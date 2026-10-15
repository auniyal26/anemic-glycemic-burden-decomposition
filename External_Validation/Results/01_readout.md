# Script 01 — MIDUS/MIDJA label and linkage audit

**Scope:** All SPSS variable labels + local ID-column joins. No individual records exported.
**Interpretation:** Hits are provisional candidates, NOT validated measurements. Review codebooks and units.
**Privacy:** Source-policy cell counts <=10 suppressed, including zero.

## File coverage

| Wave | Kind | Variables | Candidate marker groups | Unique IDs |
|---|---|---:|---|---:|
| M2 | BIO | 3516 | HbA1c, Glucose, Insulin | 1255 |
| M2 | SURVEY | 2189 | none matched | 4963 |
| M2 | COG | 223 | none matched | 4512 |
| M3 | BIO | 3205 | HbA1c, Glucose, Insulin | 747 |
| M3 | SURVEY | 2613 | none matched | 3294 |
| M3 | COG | 245 | none matched | 3291 |
| MIDJA_2 | BIO | 1000 | Hb, RBC, MCV, HbA1c | 328 |
| MIDJA_1 | SURVEY | 703 | none matched | 1027 |
| MIDJA_1 | BIO | 946 | Hb, RBC, MCV, HbA1c | 382 |
| MIDJA_2 | SURVEY | 760 | none matched | 657 |
| MR1 | BIO | 3469 | HbA1c, Glucose, Insulin | 863 |
| MR1 | SURVEY | 2605 | none matched | 3577 |
| MR1 | COG | 252 | none matched | 2763 |
| MR2 | SURVEY | 2493 | none matched | 2154 |
| MR2 | COG | 399 | none matched | 1934 |

## Candidate physiological variables by file

### M2_BIO_AGGREGATE_N1255_20240405.sav
- **HbA1c** (1 possible): `B4BHA1C` (Blood Hemoglobin A1c %)
- **Glucose** (1 possible): `B4BGLUC` (Blood Fasting Glucose levels mg/dL)
- **Insulin** (2 possible): `B4BINSLN` (Blood Fasting Insulin levels uIU/mL); `B4BIGF1` (Blood Fasting IGF1 (Insulin-like Growth Factor 1) ng/mL))

### M3_BIO_AGGREGATE_N747_20231016.sav
- **HbA1c** (1 possible): `C4BHA1C` (Blood Hemoglobin A1c %)
- **Glucose** (1 possible): `C4BGLUC` (Blood Fasting Glucose levels mg/dL)
- **Insulin** (2 possible): `C4BINSLN` (Blood Fasting Insulin levels uIU/mL); `C4BIGF1` (Blood Fasting IGF1 (Insulin-like Growth Factor 1) ng/mL))

### MIDJA 2_Biomarker_Public_N=328_10-06-2016.sav
- **Hb** (3 possible): `K2CHGB` (Tokyo Lab: Hemoglobin (HGB)); `K2CMCH` (Tokyo Lab: Mean Cell Hemoglobin (MCH)); `K2CMCHC` (Tokyo Lab: Mean Cell Hemoglobin Concentration (MCHC))
- **RBC** (1 possible): `K2CRBC` (Tokyo Lab: Red Blood Cell Count)
- **MCV** (1 possible): `K2CMCV` (Tokyo Lab: Mean Cell Volume (MCV))
- **HbA1c** (2 possible): `K2CHBA1C` (Tokyo Lab: Hemoglobin A1c (%)); `K2BHA1CA` (MIDUS Adjusted Tokyo Lab: Hemoglobin A1c (%))

### MIDJA_1_Biomarker_Public_N382_20180816.sav
- **Hb** (3 possible): `J2CHGB` (Tokyo Lab Anemia: Hemoglobin (HGB)); `J2CMCH` (Tokyo Lab Anemia: Mean Cell Hemoglobin (MCH)); `J2CMCHC` (Tokyo Lab Anemia: Mean Cell Hemoglobin Concentration (MCHC))
- **RBC** (1 possible): `J2CRBC` (Tokyo Lab Anemia: Red Blood Cell Count)
- **MCV** (1 possible): `J2CMCV` (Tokyo Lab Anemia: Mean Cell Volume (MCV))
- **HbA1c** (2 possible): `J2CHBA1C` (Tokyo Lab Diabetes: Hemoglobin A1c (%)); `J2BHA1CA` (Tokyo Lab Diabetes: MIDUS adjusted Hemoglobin A1c (%))

### MR1_BIO_AGGREGATE_N863_20240405.sav
- **HbA1c** (1 possible): `RA4BHA1C` (Blood Hemoglobin A1c %)
- **Glucose** (1 possible): `RA4BGLUC` (Blood Fasting Glucose levels mg/dL)
- **Insulin** (2 possible): `RA4BINSLN` (Blood Fasting Insulin levels uIU/mL); `RA4BIGF1` (Blood Fasting IGF1 (Insulin-like Growth Factor 1) ng/mL))

## Candidate depression and cognitive outcomes

These are label-based leads, not confirmed equivalent scales across waves.

### M2 — BIO
- **CESD:** 25 candidate variables; `B4Q3A` (CESD I was bothered by things that usually don't bother me); `B4Q3B` (CESD I did not feel like eating; my appetite was poor); `B4Q3C` (CESD I felt that I could not shake off the blues even with the help of my f); `B4Q3D` (CESD I felt that I was just as good as other people); ...
- **Depression:** 82 candidate variables; `B4Q1A` (MASQ Felt sad); `B4Q1B` (MASQ Startled easily); `B4Q1C` (MASQ Felt cheerful); `B4Q1D` (MASQ Felt Afraid); ...
- **Cognition:** 9 candidate variables; `B4ZB3CLG` (Lag Cognitive Battery to Biomarker complete in months, rounded); `B4QSC_CC` (Cognition Control); `B4BSCL21` (Saliva cortisol post-cognitive dupl#1 (nmol/L)); `B4BSCL22` (Saliva cortisol Post-cognitive dupl#2 (nmol/L)); ...

### M2 — SURVEY
- **Depression:** 17 candidate variables; `B1PA60` (Felt sad/depressed for 2 or more weeks (12 months)); `B1PA70` (Number weeks sad/depressed (12 months)); `B1PA71MO` (Most recent month sad/depressed); `B1PA71YR` (Most recent year sad/depressed); ...

### M2 — COG
- **Cognition:** 9 candidate variables; `B3TCOMPZ1` (Zscore BTACT Composite Score - computed for MIDUS national sample only (N=4); `B3TEMZ1` (Zscore Episodic Memory - computed for MIDUS national sample only (N=4206)); `B3TEFZ1` (Zscore Executive Functioning - computed for MIDUS national sample only (N=4); `B3TCOMPZ2` (Zscore BTACT Composite Score - computed for Milwaukee sample only (N=306)); ...

### M3 — BIO
- **CESD:** 25 candidate variables; `C4Q3A` (CESD I was bothered by things that usually don't bother me); `C4Q3B` (CESD I did not feel like eating; my appetite was poor); `C4Q3C` (CESD I felt that I could not shake off the blues even with the help of my f); `C4Q3D` (CESD I felt that I was just as good as other people); ...
- **Depression:** 81 candidate variables; `C4Q1A` (MASQ Felt sad); `C4Q1B` (MASQ Startled easily); `C4Q1C` (MASQ Felt cheerful); `C4Q1D` (MASQ Felt afraid); ...
- **Cognition:** 6 candidate variables; `C4ZC3CLG` (Lag Cognitive Battery to Biomarker complete in months,rounded); `C4QSC_CC` (Cognition Control); `C4VCS1` (Cognitive stressor #1); `C4VSRCS1` (Stress rating, cognitive stressor #1); ...

### M3 — SURVEY
- **Depression:** 16 candidate variables; `C1PA60` (Felt sad/depressed for 2 or more weeks (12 months)); `C1PA70` (Number weeks sad/depressed (12 months)); `C1PA71MO` (Most recent month sad/depressed); `C1PA71YR` (Most recent year sad/depressed); ...

### M3 — COG
- **Cognition:** 4 candidate variables; `C3PHONTYPE` (Type of phone on which R completed BTACT); `C3TCOMP` (MIDUS 3 BTACT Composite Score standardized based on main national MIDUS 2 M); `C3TEM` (MIDUS 3 Episodic Memory standardized based on main national MIDUS 2 M and S); `C3TEF` (MIDUS 3 Executive Functioning standardized based on main national MIDUS 2 M)

### MIDJA_2 — BIO
- **CESD:** 21 candidate variables; `K2Q43A` (CESD Unusually bothered by things); `K2Q43B` (CESD Appetite was poor); `K2Q43C` (CESD Could not shake off blues); `K2Q43D` (CESD Felt I was as good as others); ...
- **Depression:** 4 candidate variables; `K2Q1V` (Ever had depression); `K2Q1VD` (Physician diagnosed depression); `K2Q43F` (CESD Felt depressed); `K2QCESD` (CESD: Center for Epidemiologic Studies Depression Scale)

### MIDJA_1 — SURVEY
- **Depression:** 3 candidate variables; `J1SA8T` (Anxiety/depression ever (12 mo)); `J1SA9K` (RX anxiety/depression ever (30 days)); `J1SA9KY` (RX anxiety/depression freq (30 days))
- **Cognition:** 1 candidate variables; `J1SSC_CC` (Cognition Control)

### MIDJA_1 — BIO
- **CESD:** 25 candidate variables; `J2Q43A` (CESD Unusually bothered by things); `J2Q43B` (CESD Appetite was poor); `J2Q43C` (CESD Could not shake off blues); `J2Q43D` (CESD Felt I was as good as others); ...
- **Depression:** 5 candidate variables; `J2Q1V` (Ever had depression); `J2Q1VD` (Physician diagnosed depression); `J2Q43F` (CESD Felt depressed); `J2QCESDDA` (CESD Depressive Affect Subscale (C,F,I,J,N,Q,R)); ...

### MIDJA_2 — SURVEY
- **Depression:** 5 candidate variables; `K1SA9T` (Anxiety/depression ever (12 months)); `K1SA10K` (RX anxiety/depression ever (30 days)); `K1SA10KY` (RX anxiety/depression frequency (30 days)); `K1SD6` (Feeling sad, blue, depressed (12 months)); ...
- **Cognition:** 1 candidate variables; `K1SSC_CC` (Cognition Control)

### MR1 — BIO
- **CESD:** 25 candidate variables; `RA4Q3A` (CESD I was bothered by things that usually don't bother me); `RA4Q3B` (CESD I did not feel like eating; my appetite was poor); `RA4Q3C` (CESD I felt that I could not shake off the blues even with the help of my f); `RA4Q3D` (CESD I felt that I was just as good as other people); ...
- **Depression:** 82 candidate variables; `RA4Q1A` (MASQ Felt sad); `RA4Q1B` (MASQ Startled easily); `RA4Q1C` (MASQ Felt cheerful); `RA4Q1D` (MASQ Felt afraid); ...
- **Cognition:** 7 candidate variables; `RA4ZRA3CLG` (Lag Cognitive Battery to Biomarker complete in months, rounded); `RA4QSC_CC` (Cognition Control); `RA4BSCL3` (Saliva cortisol number 3: post cognitive (nmol/L)); `RA4VCS1` (Cognitive stressor #1); ...

### MR1 — SURVEY
- **Depression:** 15 candidate variables; `RA1PA60` (Felt sad/depressed for 2 or more weeks (12 months)); `RA1PA70` (Number weeks sad/depressed (12 months)); `RA1PA71MO` (Most recent month sad/depressed); `RA1PA71YR` (Most recent year sad/depressed); ...

### MR1 — COG
- **Cognition:** 13 candidate variables; `RA3PHONTYPE` (Type of phone on which R completed BTACT); `RA3TCOMPZ1` (Zscore BTACT Composite Score - computed for MIDUS Refresher national sample); `RA3TEMZ1` (Zscore Episodic Memory - computed for MIDUS Refresher national sample only ); `RA3TEFZ1` (Zscore Executive Functioning - computed for MIDUS Refresher national sample); ...

### MR2 — SURVEY
- **Depression:** 16 candidate variables; `RB1PA60` (Felt sad/depressed for 2 or more weeks (12 months)); `RB1PA70` (Number weeks sad/depressed (12 months)); `RB1PA71MO` (Most recent month sad/depressed); `RB1PA71YR` (Most recent year sad/depressed); ...

### MR2 — COG
- **Cognition:** 162 candidate variables; `RB3PHONTYPE` (Type of phone on which R completed BTACT and MoCA); `RB3TEMZ_Long` (Longitudinal BTACT Episodic Memory Factor (standardized based on main natio); `RB3TEFZC_Long` (Longitudinal BTACT Executive Functioning Factor (standardized based on main); `RB3TCOMPZ_Long` (Longitudinal BTACT Composite Score (standardized based on main national MR1); ...

## Aggregate participant linkage (not complete-case samples)

| Family | From | To | Shared participant IDs |
|---|---|---|---:|
| MIDUS_CORE | M2_BIO | M2_SURVEY | 1054 |
| MIDUS_CORE | M2_BIO | M2_COG | 1152 |
| MIDUS_CORE | M2_BIO | M3_BIO | 687 |
| MIDUS_CORE | M2_BIO | M3_COG | 1049 |
| MIDUS_CORE | M2_COG | M3_COG | 2940 |
| MIDUS_CORE | M3_BIO | M3_SURVEY | 644 |
| MIDUS_CORE | M3_BIO | M3_COG | 731 |
| MIDUS_REFRESHER | MR1_BIO | MR1_SURVEY | 746 |
| MIDUS_REFRESHER | MR1_BIO | MR1_COG | 801 |
| MIDUS_REFRESHER | MR1_BIO | MR2_COG | 689 |
| MIDUS_REFRESHER | MR1_COG | MR2_COG | 1602 |
| MIDJA | MIDJA_1_BIO | MIDJA_1_SURVEY | 382 |
| MIDJA | MIDJA_1_BIO | MIDJA_2_BIO | 243 |
| MIDJA | MIDJA_1_BIO | MIDJA_2_SURVEY | 317 |
| MIDJA | MIDJA_2_BIO | MIDJA_2_SURVEY | 328 |
| MIDUS_CORE | M2_BIO+M2_COG | M3_COG | 972 |
| MIDUS_CORE | M2_BIO+M3_BIO | M3_COG | 673 |
| MIDUS_REFRESHER | MR1_BIO+MR1_COG | MR2_COG | 649 |

## Decision gates

1. Verify every biomarker variable against the downloaded SPSS/PDF codebook, including laboratory units, fasting status, timing and missing-value codes.
2. Determine seven-biomarker complete cases on-device; ID overlap alone is not enough.
3. Keep the original MIDUS core, Refresher and Japanese MIDJA cohorts distinct.
4. For longitudinal models confirm actual visit time/order and repeated measurements; don't assume biomarker and cognitive measurement dates coincide.
5. Report exact seven-marker transfer only if all original markers can be harmonized. Otherwise prespecify reduced-variable tests and call them adaptations.
6. Preserve frozen NHANES scripts and maps. Do not refit them on external outcomes.
