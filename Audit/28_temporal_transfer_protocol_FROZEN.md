# Frozen Temporal Transfer Protocol

**Frozen:** 2026-08-22  
**Purpose:** test whether the 2005-2008 NHANES finding transfers forward in time without changing the model after seeing new outcomes.

## Discovery
- NHANES 2005-2006
- NHANES 2007-2008

## Validation
- Temporal replication: 2009-2018
- Final modern holdout: August 2021-August 2023

## Frozen primary outcome
`SOMATIC_SCORE = DPQ030 + DPQ040 + DPQ050`

## Frozen exposure definitions
- `A = max(HB_THRESHOLD - LBXHGB, 0)`
  - male threshold = 13.0 g/dL
  - female threshold = 12.0 g/dL
- `G_HBA1C = max(LBXGH - 5.7, 0)`
- `AG_HBA1C = A * G_HBA1C` — exploratory interaction

## Frozen X3 adjustment
`RIDAGEYR + RIAGENDR + RIDRETH1 + INDFMPIR + EDUC3 + SMOKING3 + BMXBMI + EGFR_2021`

## Frozen primary model
`SOMATIC_SCORE ~ A + G_HBA1C + X3`

## Frozen secondary models
- `PHQ9_TOTAL ~ A + G_HBA1C + X3`
- `SOMATIC_SCORE ~ A + G_HBA1C + AG_HBA1C + X3`

## Transfer criteria
A minimum replication requires:
- A remains positive in 2009-2018 and 2021-2023.
- G remains positive and adds non-zero weighted incremental information in both periods.
- No thresholds, PHQ items, or confounder definitions are changed after outcome inspection.

A strong replication additionally requires design-based confidence intervals excluding zero where the survey design has adequate precision.

A×G remains exploratory unless it independently stabilizes.

## Survey design
- 2009-2018 pooled full-sample weight: `WTMEC2YR / 5`
- 2021-2023: released `WTMEC2YR`
- fasting sensitivity: corresponding `WTSAF2YR` weights
- inference: R `survey` design-based models

## Rule
The validation data are tests of the frozen model, not a new model-selection dataset.

Full machine-readable protocol and discovery file hashes:
`Audit/28_temporal_transfer_protocol_FROZEN.json`
