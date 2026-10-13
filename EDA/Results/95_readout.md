# Experiment 95 — Exploratory physiological configuration atlas

**Dataset:** real NHANES Script-73; 2005–06 train, later cycles held out.
**Physiology:** seven frozen Z coordinates. **Clustering:** K=4, weighted KMeans on train only.
**Discovery cluster silhouette:** 0.1089; mean bootstrap ARI: 0.4617.

## Held-out group identification (weighted log loss; lower better)

| Period | Model | Log loss | Δ information vs prior (nats) | Top-2 group hit |
|---|---|---:|---:|---:|
| 2007-2008 | prior_only | 1.1759 | +0.0000 | 0.6807 |
| 2007-2008 | PHQ_total | 1.1720 | +0.0040 | 0.6980 |
| 2007-2008 | PHQ_items | 1.1852 | -0.0093 | 0.6824 |
| 2007-2008 | covariates_only | 0.9488 | +0.2271 | 0.8469 |
| 2007-2008 | PHQ_total_plus_covariates | 0.9488 | +0.2272 | 0.8469 |
| 2007-2008 | PHQ_items_plus_covariates | 0.9530 | +0.2230 | 0.8466 |
| 2009-2018 | prior_only | 1.2103 | +0.0000 | 0.5582 |
| 2009-2018 | PHQ_total | 1.2082 | +0.0022 | 0.6502 |
| 2009-2018 | PHQ_items | 1.2192 | -0.0088 | 0.6534 |
| 2009-2018 | covariates_only | 1.0222 | +0.1882 | 0.8307 |
| 2009-2018 | PHQ_total_plus_covariates | 1.0215 | +0.1888 | 0.8302 |
| 2009-2018 | PHQ_items_plus_covariates | 1.0307 | +0.1797 | 0.8305 |
| 2021-2023 | prior_only | 1.2264 | +0.0000 | 0.5604 |
| 2021-2023 | PHQ_total | 1.2175 | +0.0090 | 0.6635 |
| 2021-2023 | PHQ_items | 1.2260 | +0.0005 | 0.6706 |
| 2021-2023 | covariates_only | 1.0074 | +0.2190 | 0.8428 |
| 2021-2023 | PHQ_total_plus_covariates | 1.0010 | +0.2254 | 0.8485 |
| 2021-2023 | PHQ_items_plus_covariates | 1.0068 | +0.2197 | 0.8430 |

## Interpretation guardrails
- All groups are exploratory clusters, not discovered diseases or etiologies.
- A positive held-out information gain over the prior supports distinguishability; negative means worse than the prior.
- PHQ-vs-covariate comparisons test whether symptom information adds beyond available demographics.
- These metrics classify Z-derived groups, not individual 7D states; no proof of recovering patients' true physiology.
- No causal claim, no person-level longitudinal evidence, no external-dataset validation.
- Frozen mapping unchanged, no Git/Notion changes. Full complex survey-design inference not implemented.
