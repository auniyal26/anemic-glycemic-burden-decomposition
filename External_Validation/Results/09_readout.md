# Script 09 — Independent MIDUS physiological discovery and CES-D validation

## Independent discovery

- MIDUS Core M2: 1219 complete participants; MIDUS Refresher MR1: 846 complete participants.
- Three glycemic markers were selected from MIDUS measurements without using NHANES coefficients or CES-D outcomes.
- Core-only outcome-independent PCA; reconstruction; separately rediscovered Refresher PCA; frozen forward and reverse models.
- This is a glycemic-only experiment, not the NHANES seven-variable hematology/glycemia decomposition.

## Structure and rediscovery

| PCs | Core explained variance | Refresher explained variance | Independent subspace cosine | Refresher frozen energy retained |
|---|---:|---:|---:|---:|
| 1 | 0.6226 | 0.6448 | 0.9917 | 0.5659 |
| 2 | 0.9219 | 0.9014 | 0.9945 | 0.9018 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## CES-D forward inference

Positive ΔMSE favors the augmented specification. Refresher values are frozen Core-model transfer.

| Outcome | Refresher covariate MSE | HbA1c ΔMSE | Raw G3 ΔMSE | PC1–2 ΔMSE | PC1–3 ΔMSE |
|---|---:|---:|---:|---:|---:|
| CESD_total | 59.1297 | +0.2114 | +0.5019 | +0.4876 | +0.4998 |
| CESD_somatic | 9.5076 | +0.0140 | +0.0405 | +0.0415 | +0.0396 |
| CESD_affect | 8.9392 | +0.0095 | -0.0607 | -0.0576 | -0.0610 |

## Paired frozen-transfer tests

Positive ΔMSE favors the candidate. Intervals are descriptive participant-paired case bootstrap percentiles, not design-based confidence intervals.

| Outcome | Comparator → candidate | ΔMSE | Descriptive 95% interval | Result |
|---|---|---:|---|---|
| CESD_total | covariates → PC1_2 | +0.4876 | [-0.5944, +1.6077] | inconclusive |
| CESD_total | HbA1c → PC1_2 | +0.2763 | [-0.7428, +1.3481] | inconclusive |
| CESD_total | covariates → raw_G3 | +0.5019 | [-0.6682, +1.4877] | inconclusive |
| CESD_somatic | covariates → PC1_2 | +0.0415 | [-0.1289, +0.2083] | inconclusive |
| CESD_somatic | HbA1c → PC1_2 | +0.0276 | [-0.1201, +0.1755] | inconclusive |
| CESD_somatic | covariates → raw_G3 | +0.0405 | [-0.1436, +0.2203] | inconclusive |
| CESD_affect | covariates → PC1_2 | -0.0576 | [-0.2103, +0.0931] | inconclusive |
| CESD_affect | HbA1c → PC1_2 | -0.0671 | [-0.2193, +0.0865] | inconclusive |
| CESD_affect | covariates → raw_G3 | -0.0607 | [-0.2137, +0.0989] | inconclusive |

## Reverse inference (CES-D → glycemic components)

Positive ΔMSE indicates an improvement over the age/sex-only reverse model.

| Model | Mean component MSE | Mean ΔMSE versus covariates | Mean 90% marginal interval coverage |
|---|---:|---:|---:|
| covariates | 1.0013 | +0.0000 | 0.847 |
| CESD_total | 0.9950 | +0.0063 | 0.850 |
| CESD_subscales | 1.0006 | +0.0006 | 0.853 |

## Interpretation boundaries

- The independently rediscovered axes are statistical patterns, not identified biological mechanisms.
- Equal-dimensional G3 and 3PC regression retain the same linear information; rotation alone cannot add information.
- Refresher is an independent cohort, but clinical assay harmonization, depression-scale invariance and attrition remain unaudited.
- The reverse-model 90% widths come from discovery cross-validation residuals; coverage is descriptive, not finite-sample guaranteed.
- Results are exploratory after testing several related outcomes and specifications; paired case-bootstraps do not adjust for sampling, selection or multiple tests.
- These are concurrent biomarker–CES-D associations, not prospective or causal effects.
- Outcome changes, instrument equivalence and the unavailable hematology block require separate experiments.
