# Script 00 — MIDUS local file and schema inventory

**Scope:** American MIDUS Core/Refresher + Japanese MIDJA; KNHANES deferred.
**Method:** SPSS metadata only. No participant rows read, no variable harmonization claimed.

- SPSS files staged: **15**
- Schemas inspected: **15**

## Study coverage

| Wave | Survey | Biomarker | Cognitive | Notes |
|---|---|---|---|---|
| M2 | present | present | present | 3 files |
| M3 | present | present | present | 3 files |
| MR1 | present | present | present | 3 files |
| MR2 | present | not part of downloaded wave | present | 2 files |
| MIDJA_1 | present | present | not part of downloaded wave | 2 files |
| MIDJA_2 | present | present | not part of downloaded wave | 2 files |

## File inventory

| Wave | Role | File | Columns | Participant ID field | Status |
|---|---|---|---:|---|---|
| M2 | BIO | `M2_BIO_AGGREGATE_N1255_20240405.sav` | 3516 | `M2ID` | ok |
| M2 | SURVEY | `M2_P1_SURVEY_N4963_20200720.sav` | 2189 | `M2ID` | ok |
| M2 | COG | `M2_P3_BTACT_N4512_20211123.sav` | 223 | `M2ID` | ok |
| M3 | BIO | `M3_BIO_AGGREGATE_N747_20231016.sav` | 3205 | `M2ID` | ok |
| M3 | SURVEY | `M3_P1_SURVEY_N3294_20251029.sav` | 2613 | `M2ID` | ok |
| M3 | COG | `M3_P3_BTACT_N3291_20210922.sav` | 245 | `M2ID` | ok |
| MIDJA_2 | BIO | `MIDJA 2_Biomarker_Public_N=328_10-06-2016.sav` | 1000 | `MIDJA_IDs` | ok |
| MIDJA_1 | SURVEY | `MIDJA_ SURVEY_ DATA_Public_01_6-13-11.sav` | 703 | `MIDJA_IDs` | ok |
| MIDJA_1 | BIO | `MIDJA_1_Biomarker_Public_N382_20180816.sav` | 946 | `MIDJA_IDs` | ok |
| MIDJA_2 | SURVEY | `MIDJA_2_SURVEY_DATA_Public_01_03-09-16.sav` | 760 | `MIDJA_IDs` | ok |
| MR1 | BIO | `MR1_BIO_AGGREGATE_N863_20240405.sav` | 3469 | `MRID` | ok |
| MR1 | SURVEY | `MR1_P1_SURVEY_N3577_20250407.sav` | 2605 | `MRID` | ok |
| MR1 | COG | `MR1_P3_BTACT_N2763_20251119.SAV` | 252 | `MRID` | ok |
| MR2 | SURVEY | `MR2_P1_SURVEY_N2154_20251003.sav` | 2493 | `MRID` | ok |
| MR2 | COG | `MR2_P3_BTACT+MOCA_N1934_20260709.sav` | 399 | `MRID` | ok |

## Next action

Run `01_midus_schema_linkage_audit.py` to inspect complete SPSS variable labels,
candidate blood markers, depressive/cognitive phenotypes and aggregate ID overlap.
Presence of files or linkage IDs is **not** evidence of seven-marker complete cases.
