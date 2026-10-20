# Script 10 — Independent MIDJA A/G decomposition and CES-D audits

## Study design

- Panel: Hb, RBC, MCV and raw HbA1c.
- MIDJA1 complete N=353; MIDJA2 complete N=303.
- MIDJA2 with no MIDJA1 biomarker record: 78; repeated MIDJA1 participants: 225.
- Components are learned without NHANES/MIDUS loadings or CES-D supervision.
- MIDJA waves contain overlapping participants; wave-to-wave structure is not independent-person replication.
- Biomarker and CES-D measurement comparability, attrition, and assay harmonization are not established.

## Statistical structure

| PCs | MIDJA1 cumulative variance | MIDJA2 cumulative variance | Subspace cosine | MIDJA2 frozen reconstruction energy |
|---|---:|---:|---:|---:|
| 1 | 0.4547 | 0.4745 | 0.9710 | 0.4150 |
| 2 | 0.7491 | 0.7644 | 0.9692 | 0.7203 |
| 3 | 0.9920 | 0.9884 | 0.9993 | 0.9882 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## CES-D forward transfer (MIDJA1 fitted, MIDJA2 evaluated)

Positive ΔMSE favors biomarkers added to age and sex.

| Model | MIDJA2 all ΔMSE | New-participant ΔMSE | Repeated-participant ΔMSE |
|---|---:|---:|---:|
| covariates | +0.0000 | +0.0000 | +0.0000 |
| HbA1c | -1.5374 | -2.4928 | -1.2062 |
| hematology_raw | -0.4810 | -0.7353 | -0.3928 |
| raw_X4 | -1.6991 | -2.7146 | -1.3470 |
| PC1 | -0.3751 | -0.5876 | -0.3014 |
| PC1_2 | -1.0431 | -1.5612 | -0.8635 |
| PC1_2_3 | -1.8270 | -2.8670 | -1.4665 |
| PC1_2_3_4 | -1.5942 | -2.5853 | -1.2506 |
| A_PC1_2_plus_G | -1.8306 | -2.8719 | -1.4696 |
| A_PC1_2_3_plus_G | -1.5987 | -2.5927 | -1.2541 |

## Paired transfer comparisons

Bootstrap intervals resample participants within each evaluated group; not survey-adjusted or multiplicity-adjusted.

| Group | Comparator → candidate | ΔMSE | Descriptive 95% interval | Interpretation |
|---|---|---:|---|---|
| MIDJA2_all | covariates → PC1_2 | -1.0431 | [-1.7646, -0.3840] | favors_reference |
| MIDJA2_all | HbA1c → PC1_2 | +0.4943 | [-0.1116, +1.1900] | inconclusive |
| MIDJA2_all | covariates → raw_X4 | -1.6991 | [-2.6200, -0.8629] | favors_reference |
| MIDJA2_all | hematology_raw → A_PC1_2_plus_G | -1.3497 | [-2.0346, -0.7893] | favors_reference |
| MIDJA2_no_wave1_participation | covariates → PC1_2 | -1.5612 | [-2.6763, -0.4678] | favors_reference |
| MIDJA2_no_wave1_participation | HbA1c → PC1_2 | +0.9316 | [-0.6680, +2.5915] | inconclusive |
| MIDJA2_no_wave1_participation | covariates → raw_X4 | -2.7146 | [-4.7783, -1.0141] | favors_reference |
| MIDJA2_no_wave1_participation | hematology_raw → A_PC1_2_plus_G | -2.1366 | [-4.0739, -0.5052] | favors_reference |
| MIDJA2_wave1_participants | covariates → PC1_2 | -0.8635 | [-1.5980, -0.0872] | favors_reference |
| MIDJA2_wave1_participants | HbA1c → PC1_2 | +0.3427 | [-0.2694, +1.0266] | inconclusive |
| MIDJA2_wave1_participants | covariates → raw_X4 | -1.3470 | [-2.1963, -0.4700] | favors_reference |
| MIDJA2_wave1_participants | hematology_raw → A_PC1_2_plus_G | -1.0768 | [-1.6833, -0.4817] | favors_reference |

## CES-D → Z → physiological reconstruction

Predicting physiology from CES-D is an uncertain inverse problem, not an exact inverse.

| MIDJA2 group | Model | Average PC MSE | Gain vs age/sex | Average nominal 90% marginal coverage |
|---|---|---:|---:|---:|
| MIDJA2_all | covariates | 0.8733 | +0.0000 | 0.881 |
| MIDJA2_all | CESD_total | 0.8849 | -0.0116 | 0.883 |
| MIDJA2_no_wave1_participation | covariates | 1.1909 | +0.0000 | 0.869 |
| MIDJA2_no_wave1_participation | CESD_total | 1.2081 | -0.0172 | 0.869 |
| MIDJA2_wave1_participants | covariates | 0.7632 | +0.0000 | 0.886 |
| MIDJA2_wave1_participants | CESD_total | 0.7729 | -0.0097 | 0.888 |

## Within-person chronological audit

- Matched complete cases: 211; verified later follow-up: 211.
- Status: exploratory_observational.

| Baseline → later CES-D model | Participant-level five-fold CV ΔMSE vs prior CES-D + age/sex |
|---|---:|
| prior_CESD | +0.0000 |
| prior_CESD_HbA1c | +1.0339 |
| prior_CESD_raw_X4 | +1.4985 |
| prior_CESD_PC1_2 | +0.2485 |
| prior_CESD_PC1_2_3_4 | +1.4165 |
- This is person-held-out *cross-validation on matched participants*, not external prospective validation. Baseline CES-D is included in every model.

## Interpretation limits

- Components are statistical axes; physiological interpretation needs independent testing.
- A full-rank four-PC representation is a rotation of four standardized biomarkers, not an information gain.
- HbA1c is the only glycemic measure here; full hematology–glycemia shared/private identification is limited.
- RBC values have source-specific numerical scale; scaling and laboratory units must be verified from source documentation.
- HbA1c adjusted versus raw is a sensitivity audit, not evidence that the assays are interchangeable.
- CES-D scoring and invariance, laboratory protocols, sampling weights and attrition are unaudited.
- The repeated-subject temporal analysis does not establish causality or eliminate unmeasured confounding.
- Exact numerical reconstruction at full rank checks algebra, not physiological validity.
- Interpret changes and cross-wave associations as exploratory and not proof of causality.
