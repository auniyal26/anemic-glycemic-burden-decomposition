# Script 10 — Independent MIDJA A/G decomposition and CES-D audits

## Study design

- Panel: Hb, RBC, MCV and MIDUS-adjusted HbA1c.
- MIDJA1 complete N=353; MIDJA2 complete N=303.
- MIDJA2 with no MIDJA1 biomarker record: 78; repeated MIDJA1 participants: 225.
- Components are learned without NHANES/MIDUS loadings or CES-D supervision.
- MIDJA waves contain overlapping participants; wave-to-wave structure is not independent-person replication.
- Biomarker and CES-D measurement comparability, attrition, and assay harmonization are not established.

## Statistical structure

| PCs | MIDJA1 cumulative variance | MIDJA2 cumulative variance | Subspace cosine | MIDJA2 frozen reconstruction energy |
|---|---:|---:|---:|---:|
| 1 | 0.4545 | 0.4742 | 0.9707 | 0.3155 |
| 2 | 0.7486 | 0.7645 | 0.9656 | 0.5622 |
| 3 | 0.9920 | 0.9884 | 0.9993 | 0.9912 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## CES-D forward transfer (MIDJA1 fitted, MIDJA2 evaluated)

Positive ΔMSE favors biomarkers added to age and sex.

| Model | MIDJA2 all ΔMSE | New-participant ΔMSE | Repeated-participant ΔMSE |
|---|---:|---:|---:|
| covariates | +0.0000 | +0.0000 | +0.0000 |
| HbA1c | -2.1227 | -3.6654 | -1.5879 |
| hematology_raw | -0.4810 | -0.7353 | -0.3928 |
| raw_X4 | -2.2038 | -3.7124 | -1.6808 |
| PC1 | -0.3906 | -0.6219 | -0.3104 |
| PC1_2 | -1.1533 | -1.7827 | -0.9351 |
| PC1_2_3 | -2.3445 | -3.8848 | -1.8106 |
| PC1_2_3_4 | -2.0881 | -3.5659 | -1.5758 |
| A_PC1_2_plus_G | -2.3477 | -3.8889 | -1.8134 |
| A_PC1_2_3_plus_G | -2.0939 | -3.5760 | -1.5801 |

## Paired transfer comparisons

Bootstrap intervals resample participants within each evaluated group; not survey-adjusted or multiplicity-adjusted.

| Group | Comparator → candidate | ΔMSE | Descriptive 95% interval | Interpretation |
|---|---|---:|---|---|
| MIDJA2_all | covariates → PC1_2 | -1.1533 | [-1.9100, -0.4430] | favors_reference |
| MIDJA2_all | HbA1c → PC1_2 | +0.9694 | [+0.1814, +1.8851] | favors_candidate |
| MIDJA2_all | covariates → raw_X4 | -2.2038 | [-3.3688, -1.1299] | favors_reference |
| MIDJA2_all | hematology_raw → A_PC1_2_plus_G | -1.8667 | [-2.8947, -1.0950] | favors_reference |
| MIDJA2_no_wave1_participation | covariates → PC1_2 | -1.7827 | [-3.0389, -0.6105] | favors_reference |
| MIDJA2_no_wave1_participation | HbA1c → PC1_2 | +1.8826 | [-0.3474, +4.3209] | inconclusive |
| MIDJA2_no_wave1_participation | covariates → raw_X4 | -3.7124 | [-6.6697, -1.3626] | favors_reference |
| MIDJA2_no_wave1_participation | hematology_raw → A_PC1_2_plus_G | -3.1536 | [-5.9497, -0.8137] | favors_reference |
| MIDJA2_wave1_participants | covariates → PC1_2 | -0.9351 | [-1.7117, -0.1391] | favors_reference |
| MIDJA2_wave1_participants | HbA1c → PC1_2 | +0.6528 | [-0.0566, +1.4650] | inconclusive |
| MIDJA2_wave1_participants | covariates → raw_X4 | -1.6808 | [-2.7739, -0.6021] | favors_reference |
| MIDJA2_wave1_participants | hematology_raw → A_PC1_2_plus_G | -1.4206 | [-2.2694, -0.6006] | favors_reference |

## CES-D → Z → physiological reconstruction

Predicting physiology from CES-D is an uncertain inverse problem, not an exact inverse.

| MIDJA2 group | Model | Average PC MSE | Gain vs age/sex | Average nominal 90% marginal coverage |
|---|---|---:|---:|---:|
| MIDJA2_all | covariates | 1.2049 | +0.0000 | 0.845 |
| MIDJA2_all | CESD_total | 1.2187 | -0.0138 | 0.847 |
| MIDJA2_no_wave1_participation | covariates | 1.8371 | +0.0000 | 0.837 |
| MIDJA2_no_wave1_participation | CESD_total | 1.8588 | -0.0217 | 0.837 |
| MIDJA2_wave1_participants | covariates | 0.9858 | +0.0000 | 0.848 |
| MIDJA2_wave1_participants | CESD_total | 0.9968 | -0.0110 | 0.851 |

## Within-person chronological audit

- Matched complete cases: 211; verified later follow-up: 211.
- Status: exploratory_observational.

| Baseline → later CES-D model | Participant-level five-fold CV ΔMSE vs prior CES-D + age/sex |
|---|---:|
| prior_CESD | +0.0000 |
| prior_CESD_HbA1c | +1.1268 |
| prior_CESD_raw_X4 | +1.5828 |
| prior_CESD_PC1_2 | +0.2324 |
| prior_CESD_PC1_2_3_4 | +1.5041 |
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
