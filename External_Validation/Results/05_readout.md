# Script 05 — Frozen NHANES G3 representation and prospective MIDUS transfer

**Status:** Research analysis; source-codebook verification, survey selection and score invariance remain open.
**Scope:** A NEW NHANES-2005–08 outcome-blind **G3-only** PCA benchmark, NOT original NHANES 7D shared/discordant decomposition.
**Privacy:** No participant-level data, IDs, fitted individual predictions or per-person scores exported. Cells 0–10 suppressed.

## Reference physiology and lock

- Training data: `EDA\Results\73_frozen_phenotype_map_input.csv`; discovery 2005–08 eligible NHANES only; N = 4518.
- Variables, in order: HbA1c (%), fasting glucose (mg/dL), natural log fasting insulin (uIU/mL).
- Sample weights used ONLY to learn NHANES discovery means/SDs/covariance/eigenvectors.
- No MIDUS physiology or phenotype used to fit the NHANES reference; no NHANES outcome used.
- PC2/PC3 are compressions with possible reconstruction loss; full 3-PC invertibility is algebraic, not discovery of extra information.
- NHANES original frozen 7D transforms, outcomes and files: unchanged.

## Cohort selection

| Cohort | linked bio+earlier/later cognition | ordered, lab and basic covariates |
|---|---:|---:|
| CORE | 972 | 846 |
| REFRESHER | 649 | 576 |

## Reconstruction and physiology stability

Reconstruction rMSE is in *NHANES-frozen standardized G3 units*. "Energy retained" uses NHANES origin, not MIDUS-centered variance.

| Cohort | Retained PCs | Standardized rMSE | NHANES-origin energy retained |
|---|---:|---:|---:|
| CORE | 1 | 0.6192 | 0.5549 |
| CORE | 2 | 0.3781 | 0.8340 |
| CORE | 3 | 0.0000 | 1.0000 |
| REFRESHER | 1 | 0.6470 | 0.5720 |
| REFRESHER | 2 | 0.3572 | 0.8696 |
| REFRESHER | 3 | 0.0000 | 1.0000 |

| Cohort | Abs. cosine: NHANES vs local first PC |
|---|---:|
| CORE | 0.9749 |
| REFRESHER | 0.9488 |

## Locked cognitive mapping: summary

Five-fold shuffled *Core-only* CV; same participants per domain/model, fixed Ridge alpha=10 (after fold-training standardization).
All Refresher predictions use the **unchanged Core-trained models** and Core-trained feature scalers.
Each outcome is a follow-up BTACT reference-standardized score; baseline BTACT, age at blood, sex, earlier education and both visit gaps are baseline features.
Refresher and Core BTACT use different national reference populations: absolute transport metrics are **comparability-conditional**, not proof of phenotype invariance.

| Domain | Model | Core CV R2 | Refresher frozen R2 | Refresher ΔMSE vs baseline | 95% bootstrap CI |
|---|---|---:|---:|---:|---|
| comp | covariates | 0.6251 | 0.6337 | NA | NA |
| comp | HbA1c | 0.6271 | 0.6339 | 0.0001 | [-0.0029, 0.0033] |
| comp | HOMA_IR | 0.6253 | 0.6334 | -0.0001 | [-0.0023, 0.0018] |
| comp | raw_G3 | 0.6270 | 0.6326 | -0.0005 | [-0.0035, 0.0025] |
| comp | NHANES_PC1 | 0.6281 | 0.6349 | 0.0005 | [-0.0024, 0.0036] |
| comp | NHANES_PC2 | 0.6283 | 0.6327 | -0.0005 | [-0.0038, 0.0033] |
| comp | NHANES_PC3 | 0.6269 | 0.6326 | -0.0005 | [-0.0034, 0.0026] |
| memory | covariates | 0.3246 | 0.2889 | NA | NA |
| memory | HbA1c | 0.3238 | 0.2879 | -0.0009 | [-0.0044, 0.0027] |
| memory | HOMA_IR | 0.3248 | 0.2923 | 0.0031 | [-0.0018, 0.0074] |
| memory | raw_G3 | 0.3255 | 0.2885 | -0.0004 | [-0.0066, 0.0055] |
| memory | NHANES_PC1 | 0.3255 | 0.2905 | 0.0015 | [-0.0033, 0.0073] |
| memory | NHANES_PC2 | 0.3259 | 0.2848 | -0.0038 | [-0.0100, 0.0020] |
| memory | NHANES_PC3 | 0.3254 | 0.2886 | -0.0003 | [-0.0068, 0.0058] |
| executive | covariates | 0.6118 | 0.6103 | NA | NA |
| executive | HbA1c | 0.6137 | 0.6217 | 0.0055 | [0.0021, 0.0087] |
| executive | HOMA_IR | 0.6113 | 0.6102 | -0.0001 | [-0.0012, 0.0012] |
| executive | raw_G3 | 0.6074 | 0.6194 | 0.0043 | [0.0012, 0.0078] |
| executive | NHANES_PC1 | 0.6129 | 0.6182 | 0.0038 | [0.0006, 0.0070] |
| executive | NHANES_PC2 | 0.6133 | 0.6237 | 0.0064 | [0.0030, 0.0102] |
| executive | NHANES_PC3 | 0.6071 | 0.6192 | 0.0043 | [0.0011, 0.0076] |

## Caveats and stopping rules

1. The *only* physiological transfer tested is a **G3-only** representation: joint A/G, shared/discordant axes and phenotype→physiology reconstruction are NOT tested.
2. HbA1c, glucose and insulin have source-label units, but assay, fasting compliance, clinical exclusions and cross-study harmonization remain unverified.
3. HOMA-IR is `glucose(mg/dL) × insulin(uIU/mL) / 405`, an unvalidated clinical-index benchmark in this sample.
4. Standardization and predictor fits are trained on NHANES discovery (physiology) or Core (outcomes) only; Refresher is untouched until external evaluation.
5. Equal-dimension raw-G3 and 3-PC OLS models contain equivalent linear information. Ridge penalties may differ under scaling/rotation; no information gain should be claimed.
6. BTACT score comparability, practice effects, survivor/biomarker-selection bias, survey weighting and other confounders require a separate validation. These are unweighted sample metrics.
7. Baseline cognition occurred before the biomarker draw; prospective is not causal, and no model can use blood collected after the outcome.
8. Analyses are hypothesis-generating for cognition and not an exact replication of NHANES depression-PHQ phenotype associations.
9. Check metadata codebooks and category crosswalks, especially education; unfamiliar codes trigger an explicit error instead of being silently mapped.

## Output files

- `05_readout.md` (this interpretation with explicit limitations)
- `05_structure_transfer.csv` (aggregate reconstruction)
- `05_distribution_shift.csv` (aggregate means, SD, first-PC alignment)
- `05_cognitive_performance.csv` (Core CV and frozen Refresher model metrics)
- `05_external_mse_improvement.csv` (paired-bootstrap CI)
- `05_manifest.json` (input provenance, frozen basis and configuration; aggregate parameters only)

**Decision gate:** Review result directions and source manuals before claiming stable physiological or cognitive transport.
