# Research Log — 20 August 2026

## Project

**Hematological and Glycemic Decomposition of Depression**

## Goal for Today

Test whether the proposed decomposition framework is empirically viable using NHANES 2005–06 and 2007–08 before moving further into the PhD proposal and presentation.

## Conceptual Framework

Working mechanism:

1. Preserve the original mapping/reference.
2. Work on copies rather than overwrite the original representation.
3. Replace one large function with smaller components.
4. Quantify reconstruction/information loss after transformations.
5. Use domain knowledge to preserve clinically meaningful structure.
6. Decompose residual/latent structure where appropriate.
7. Test whether recovered components are stable and reproducible.

For this problem:

- $A$ = hematological burden
- $G$ = glycemic burden
- $P$ = depressive phenotype
- $X$ = demographic/confounding variables

Initial model:

$$
P=f(A,G,A\times G,X)
$$

---

## Data

NHANES 2005–06 and 2007–08.

Core dataset:

- CBC
- demographics
- HbA1c
- PHQ-9

Matched sample:

- Raw matched: **11,329**
- Complete A/G/P: **9,743**
- PHQ-9 ≥ 10: **776**

A PHQ encoding issue was identified and corrected: values representing zero were stored as approximately `5.397605e-79`.

---

## 01–04 — Initial Representation Tests

### Naive representation

$$
A=-z(Hb)
$$

$$
G=z(HbA1c)
$$

Results:

- $A$: $\beta=0.0033,\ p=.961$
- $G$: $\beta=0.3113,\ p\approx5\times10^{-6}$
- $A\times G$: $\beta=0.0124,\ p=.869$
- $R^2\approx.023$
- reconstruction error = **0**

The algebra reconstructed perfectly, but the naive anemia representation showed essentially no useful signal.

### Domain-informed representation

Anemia burden:

$$
A^*=\max(Hb_{\text{threshold(sex)}}-Hb,0)
$$

Thresholds:

- male: 13 g/dL
- female: 12 g/dL

Glycemic burden:

$$
G^*=\max(HbA1c-5.7,0)
$$

Interaction:

$$
A^*G^*=A^*\times G^*
$$

Model results:

- $A$: $\beta=0.640,\ p=.000107$
- $G$: $\beta=0.320,\ p=.000050$
- $A\times G$: $\beta=-0.268,\ p=.047$

The anemia signal reappeared after changing the representation.

**Key result:** representation choice can erase or recover phenotype-relevant information.

---

## 05 — Cross-Cycle Replication and Nonlinearity

The main $A$ and $G$ effects were more stable across cycles than the interaction.

Nonlinear pooled models slightly improved training fit but generalized poorly across cycles.

Cross-cycle nonlinear prediction produced negative test $R^2$.

Current interpretation:

$$
P\approx f_A(A)+f_G(G)+\text{residual}
$$

rather than requiring a large reproducible $A\times G$ interaction.

---

## 06–09 — Depression Phenotype Decomposition

The first residual ICA design incorrectly removed $A$ and $G$ before decomposition, forcing their later associations to zero. The design was corrected by removing only confounders before ICA.

Four-component ICA retained approximately **70.5%** of standardized residual information.

Independent decomposition in each cycle produced component similarities of approximately:

- IC1: **0.859**
- IC2: **0.875**
- IC3: **0.956**
- IC4: **0.561**

ICA repeated across 100 random seeds showed very high stability for IC1–IC3.

IC4 was less stable, especially in 2005–06.

### Current position

The strongest evidence supports approximately **three reproducible latent phenotype dimensions**, rather than assuming all four ICA components are equally reliable.

---

## Current Scientific Interpretation

The framework is empirically viable as a proof-of-concept.

What survived:

1. Original data/reference can be preserved.
2. Reconstruction can be explicitly checked.
3. Representation choice materially affects recovered signal.
4. $A$ and $G$ are more stable than a large nonlinear interaction.
5. Depression is not behaving as one homogeneous variable.
6. Reproducible latent phenotype structure exists across cycles.
7. The main latent structures are highly stable to ICA initialization.

### Main limitations

- Cross-sectional data cannot establish causality.
- Physiological effect sizes are small.
- ICA initialization stability is not sampling stability.
- $A\times G$ is not reliably replicated.
- External validation is still required.
- Full NHANES survey implementation still needs strengthening.

## Next Steps

1. Bootstrap sampling stability.
2. Determine the appropriate number of phenotype components.
3. Implement full NHANES survey design.
4. Formalize phenotype interpretation.
5. Add richer hematological features.
6. Test latent hematological and glycemic representations.
7. Compare alternative statistical frameworks.
8. Move toward external/Indian validation.

---

# Research Log — 21 August 2026

## Project

**Disentangling Hematological and Glycemic Contributions to Depressive Symptoms**

## Objective

Strengthen the initial A/G findings using proper confounding control, complex-survey analysis, phenotype validation, alternative glycemic definitions and internal physiological decomposition.

---

## Data Integrity and Analysis Freeze

A full integrity audit confirmed:

- required raw NHANES files
- unique SEQN identifiers
- PHQ-9 encoding
- complete A/G/P sample: **N = 9,743**
- raw-file hashes and transformation ledger
- reproducible latent-component structure

The exploratory analysis was frozen with:

- raw hashes
- transformation parameters
- ICA/scaler settings
- participant mappings
- seeds
- thresholds

---

## Confounding and Survey Design

Staged adjustment sets were defined:

- X0 = age + sex + race/ethnicity
- X1 = X0 + income + education + smoking
- X2 = X1 + BMI
- X3 = X2 + kidney function

Additional decisions:

- age ≥20
- known pregnancy excluded
- kidney function treated cautiously
- HbA1c/RBC measurement coupling explicitly acknowledged
- NHANES weights, strata and PSU retained

Matched survey sample:

**n = 7,957**

---

## Somatic Depression Signal

Somatic score defined a priori from:

- sleep
- fatigue
- appetite

Across survey-weighted models:

- $A$ survives adjustment
- $G$ survives adjustment
- pooled $A\times G$ remains sub-additive rather than synergistic

Independent-cycle replication showed similar negative interaction direction and magnitude, but confidence intervals crossed zero.

Therefore:

> Direction replicated better than strict interaction significance.

---

## Glycemia Sensitivity

Using fasting glucose instead of HbA1c:

- $A$ remained
- fasting-glucose $G$ remained
- $A\times G$ remained directionally similar but uncertain

This reduced concern that the glycemic signal was simply an HbA1c artifact caused by RBC turnover.

Clinical diabetes categories were less informative than continuous glycemic physiology.

---

## Formal Phenotype Validation

### Ordinal CFA

A two-factor PHQ structure outperformed a one-factor model in both cycles.

Somatic factor:

- sleep
- fatigue
- appetite

The measurement structure was approximately invariant across cycles.

### MIMIC / SEM

Model fit:

- **CFI = 0.996**
- **RMSEA = 0.011**
- **SRMR = 0.039**

Somatic factor:

- $A$: **replicates**
- $G$: **replicates**
- $A\times G$: **partial**

Cognitive-affective factor showed weaker physiological associations.

### DIF audit

No FDR-stable direct item effect was found for:

- $A$
- $G$
- $A\times G$

This suggests the signal operates at the broader somatic-factor level rather than being driven by one PHQ item.

---

## Internal Composition of A and G

### Hematological block

Exploratory variables:

- Hb
- hematocrit
- RBC count
- MCV
- MCH
- MCHC
- RDW

Complete sample:

**n = 8,686**

Three PCA dimensions were strongly reproducible across cycles.

Cross-cycle loading similarities:

- PC1: **0.999**
- PC2: **0.996**
- PC3: **0.995**

The hematological state is therefore clearly **multidimensional**.

The original Hb-deficit scalar captures only part of this structure.

### Glycemic block

Variables:

- HbA1c
- fasting glucose

Complete fasting sample:

**n = 4,204**

PC1 explained approximately **91%** of variance in both cycles and correlated strongly with the original scalar $G$.

Therefore the two-marker glycemic state appeared much closer to one dominant dimension.

---

## Current Scientific Position

1. A reproducible somatic depressive phenotype exists.
2. Hematological burden is associated with it.
3. Glycemic burden contributes independently.
4. Survey-weighted and latent-variable analyses converge.
5. The signal is not driven by one PHQ item.
6. Hematological state is substantially multidimensional.
7. The simple two-marker glycemic state is largely dominated by one axis.
8. $A\times G$ remains secondary because it is less robust than the main effects.

### Main interpretation

The original proxies were:

$$
A\approx\text{hemoglobin deficit}
$$

$$
G\approx\text{HbA1c excess}
$$

They are not equally informative.

Scalar $G$ approximates the available glycemic state reasonably well.

Scalar $A$ does not adequately represent hematological state.

## Next Step

Use stable physiological components to determine:

- which $A_i$ predict the somatic phenotype
- which $G_j$ predict it
- whether specific $A_i\times G_j$ interactions reproduce

Do not yet treat exploratory PCA components as final physiological representations.

---

# Research Log — 14 September 2026

## Project

**Computational Physiological Decomposition — Hematological and Glycemic Testbed**

## Objective

Move from exploratory A/G models toward finalized, outcome-independent physiological representations and test:

1. structural stability
2. reconstruction
3. phenotype information beyond scalar burdens
4. frozen temporal transfer

---

## Final Reduced Hematological Representation

Final primitive basis:

$$
A_{\text{raw}}=
\{Hb,\ RBC,\ MCV,\ RDW\}
$$

Three components were retained.

Discovery sample:

**n = 8,686**

Cross-cycle loading similarities:

- PC1: **0.993**
- PC2: **0.993**
- PC3: **0.999**

Reconstruction energy:

- 2005–06 → 2007–08: **0.989**
- pooled 2005–08: **0.992**

The reduced hematological state therefore preserves >99% of standardized information using three stable dimensions.

---

## Frozen Hematological Transfer

The 2005–08 transform was projected forward without refitting.

Reconstruction:

- 2009–18: **0.993**
- 2021–23: **0.996**

The hematological representation remains structurally stable through 2021–23.

This means later failure of the simple Hb-deficit scalar cannot automatically be interpreted as failure of the underlying hematological state.

---

## Glycemic Representation

### Minimal G baseline

HbA1c + fasting glucose remained dominated by one common axis and is now treated as a **minimal glycemic measurement baseline**, not proof that glycemic physiology is inherently one-dimensional.

### Deep G3 representation

$$
G3=
\{HbA1c,\ fasting\ glucose,\ \log(fasting\ insulin)\}
$$

Discovery sample:

**n = 4,145**

Approximate variance:

- PC1: **63%**
- PC2: **31%**
- PC3: **6%**

Cross-cycle loading similarity was approximately **0.9997–0.9999**.

Correlation with log-HOMA-IR:

- G3-PC1: **0.556**
- G3-PC2: **0.828**
- G3-PC3: **0.037**

Descriptive interpretation:

- PC1 = broad/common glycemic state
- PC2 = insulin-related dimension
- PC3 = smaller glucose-measurement discordance dimension

These remain descriptive rather than mechanistic labels.

---

## Extended G4 Representation

$$
G4=
\{HbA1c,\ fasting\ glucose,\ \log(fasting\ insulin),\ 2h\ OGTT\}
$$

Adding OGTT revealed additional challenge-response structure.

Cross-cycle loading similarities remained approximately **0.985–0.997**.

Important limitation:

> OGTT is unavailable after 2015–16, so G4 cannot support the full temporal ladder through 2021–23.

---

## Combined A + G3 Discovery Model

Joint fasting phenotype sample:

**n = 3,815**

Somatic phenotype:

- A PCs given G PCs + X: **p = 0.001546**
- G PCs given A PCs + X: **p = 0.027443**
- all six PCs beyond scalar A + G: **p = 0.014596**
- scalar A + G beyond all six PCs: **p = 0.955583**

Incremental information:

- scalar A + G beyond X: $\Delta R^2=0.004638$
- all PCs beyond X: $\Delta R^2=0.015235$
- PCs beyond scalar model: $\Delta R^2=0.010642$
- scalars beyond PCs: $\Delta R^2=0.000046$

### Main discovery conclusion

The multidimensional A + G3 representation contains substantially more somatic-phenotype information than conventional scalar A + G.

The information loss is asymmetric:

- scalar anemia loses substantial information
- scalar glycemia preserves more of the relevant G3 signal

---

## A/G Separability

Cross-block component correlations were generally weak.

Maximum absolute correlation:

**≈ 0.20**

This supports **statistical nonredundancy**, not biological independence.

---

## Frozen Temporal Phenotype Transfer

The discovery representations were projected forward unchanged.

No:

- representation refitting
- temporal re-standardization
- component reordering

### 2009–18

Somatic phenotype:

- A PCs given G PCs + X3: **p = 0.000070**
- G PCs given A PCs + X3: **p = 0.005769**
- all PCs beyond scalar A + G: **p = 0.006570**
- scalar pair beyond all PCs: **p = 0.176**

Incremental information:

- scalar model beyond X: $\Delta R^2=0.003869$
- all PCs beyond X: $\Delta R^2=0.007014$
- PCs beyond scalars: $\Delta R^2=0.003643$
- scalars beyond PCs: $\Delta R^2=0.000497$

The discovery decomposition advantage therefore formally replicated in 2009–18.

### Component-level replication

Major directions also reproduced:

- A-PC1: negative
- A-PC3: positive
- G-PC1: positive

This indicates that transfer is not limited to a global model statistic.

---

## 2021–23 Modern Holdout

The frozen representation remained structurally stable.

The full phenotype model became design-limited because the single modern cycle provides insufficient survey denominator degrees of freedom for the full block tests.

Effect-size pattern:

- scalar model beyond X: $\Delta R^2=0.002373$
- all PCs beyond X: $\Delta R^2=0.010668$
- PCs beyond scalar model: $\Delta R^2=0.011688$
- scalars beyond PCs: $\Delta R^2=0.003392$

This remains suggestive but is **not treated as confirmatory inference**.

---

## Current Scientific Position

The first physiological testbed now supports:

$$
\text{raw biomarkers}
\rightarrow
\text{outcome-independent decomposition}
\rightarrow
\text{reconstruction}
\rightarrow
\text{separability}
\rightarrow
\text{phenotype mapping}
\rightarrow
\text{scalar information-loss test}
\rightarrow
\text{frozen temporal transfer}
$$

Current evidence supports:

1. Hematological state is multidimensional and structurally stable.
2. Glycemic state becomes multidimensional once insulin and OGTT are included.
3. Scalar anemia loses phenotype-relevant information.
4. G3 independently contributes alongside A.
5. The combined A + G3 representation contains more information than scalar A + G.
6. The decomposition advantage replicates in 2009–18.
7. The representations remain structurally valid through 2021–23.
8. The 2021–23 phenotype analysis remains design-limited.
9. Scalar association failure can occur while multidimensional physiological structure remains preserved.

## Important Boundaries

Do not currently claim:

- causality
- clinical prediction or diagnostic utility
- biological independence of A and G
- stable A×G interaction
- final biological mechanisms for the PCs
- confirmatory 2021–23 significance
- that scalar-A failure has a known biological cause

---

# Research Log — 16 September 2026

## Objective

Test whether G4 should replace G3, and stress-test:

- broad A×G interactions
- age modification
- sex modification
- BMI modification
- eGFR modification

---

## G4 Structural Test

Extended glycemic representation:

$$
G4=
\{HbA1c,\ FPG,\ \log(fasting\ insulin),\ 2h\ OGTT\}
$$

Outcome-independent G4 discovery fit:

**n = 3,308**

K3 reconstruction:

- 2005–08: **92.83%**
- frozen 2009–16: **93.12%**

G4 therefore transfers structurally well.

---

## Same-Sample G3 vs G4

Exact OGTT analytic samples:

- 2005–08: **n = 2,972**
- 2009–16: **n = 6,248**

Weighted $R^2$:

| Period | A + G3 | A + G4 |
|---|---:|---:|
| 2005–08 | 0.08263 | 0.08278 |
| 2009–16 | 0.08174 | 0.08269 |

2h OGTT beyond full G3:

- 2005–08: **p = 0.416**
- 2009–16: **p = 0.062**

G4 adds physiological challenge-response information, but there is currently **no replicated evidence that OGTT adds substantial depressive-phenotype information beyond G3**.

Therefore:

$$
G3=\text{primary transferable glycemic representation}
$$

$$
G4=\text{extended physiological representation}
$$

---

## Hematology Under G4 Adjustment

A components remained informative after adjustment for G4:

- 2005–08: **p = 0.00738**
- 2009–16: **p = 0.000332**

G4 given A:

- 2005–08: **p = 0.0226**
- 2009–16: **p = 0.150**

The hematological representation therefore remains particularly robust across periods.

---

## A × G4 Interaction

No reproducible multiplicative phenotype interaction was detected.

2009–16:

- PC1 × PC1: **p = 0.392**
- all 9 A×G4 terms: **p = 0.250**

Discovery interaction inference remained survey-df limited.

Current evidence therefore favors:

$$
P\approx A+G+X
$$

over requiring:

$$
P\approx A+G+A\times G+X
$$

This does **not** imply that hematology and glycemia are biologically independent.

The stronger literature-supported A–G coupling remains erythrocyte-state effects on HbA1c interpretation.

---

## X Effect Modification

Prespecified modifiers:

- age
- sex
- BMI
- eGFR

No modifier survived multiplicity correction.

The closest nominal result was the combined age-interaction block in 2009–16:

- **p = 0.063**
- BH-adjusted **q = 0.646**

No robust X-dependent heterogeneity is currently supported.

---

## Current Working Hierarchy

### Core

- A basis: Hb + RBC + MCV + RDW
- G3: HbA1c + FPG + insulin
- additive phenotype model with X adjustment

### Useful Extensions

- G4 with 2h OGTT
- temporal drift analysis
- targeted mechanistic A–G measurement coupling

### Secondary / Sensitivity

- broad A×G interactions
- age / sex / BMI / eGFR effect modification

---

## Next Session

1. Fix the Script 49 development-vs-replication temporal-shift contrast.
2. Review auxiliary physiological markers.
3. Classify candidates as:
   - core stable
   - useful extension
   - sensitivity only
4. Avoid indiscriminate biomarker expansion.
5. Preserve A + G3 as the primary framework unless new evidence justifies changing it.

# Research Log Backfill — Work Completed After Script 24

## Project
**Computational Physiological Decomposition — Haematological × Glycaemic Testbed**

## Reason for Backfill

The previous experiment log ended after Script 24.

A substantial set of validation, transfer, falsification, simulation, and interpretability experiments was subsequently completed but not added to the running log.

This section reconstructs that work from the repository artifacts and generated results.

---

# Phase 1 — Formalizing Separability, Information Gain, and Transfer

## Scripts 25–27 — Validity, Separability, and Incremental Information

### Script 25
`25_pre_presentation_validity_audit.py`

A pre-presentation validity audit was run to challenge the preliminary conclusions before treating them as stable evidence.

The focus shifted from:

`A and G associate with depression`

toward:

`Do A and G contain distinguishable physiological information, and does a richer representation retain useful information that scalar proxies lose?`

### Script 26
`26_formal_separability_test.py`

Formalized the separability question.

The important distinction established was:

- low statistical correlation does **not** imply biological independence;
- biological coupling, measurement coupling, and observed statistical dependence are separate questions.

The original scalar burdens showed weak weighted correlation:

`rho_w ≈ 0.045`

This supported **statistical non-redundancy**, not biological independence.

### Script 27
`27_incremental_information_test.py`

Tested whether richer physiological information contributes beyond the clinical scalar burdens.

This introduced the central asymmetry test:

`rich representation beyond scalar`

versus

`scalar beyond rich representation`

This became one of the core methodological criteria used throughout the later work.

---

# Phase 2 — Frozen Temporal Transfer

## Scripts 28–33 — Temporal Validation Pipeline

### Script 28
`28_freeze_temporal_transfer_protocol.py`

A formal frozen-transfer protocol was established.

Rules:

- discovery transformations are frozen;
- no future-cycle refitting;
- no silent future re-standardization;
- future data are projected into the discovery coordinate system;
- later failure is treated as evidence rather than corrected post hoc.

### Scripts 29–31
- `29_download_temporal_validation_data.py`
- `30_build_frozen_transfer_cohort.py`
- `31_amend_transfer_weight_and_finalize_cohort.py`

Later NHANES cycles were harmonized and the survey-weighted temporal cohorts were constructed.

This extended the testbed beyond NHANES 2005–08 into:

- NHANES 2009–18
- NHANES 2021–23

### Script 32
`32_run_frozen_temporal_transfer_test.py`

The simple first-pass A/G representations were transferred forward.

Results:

### 2009–18
Both first-pass signals reproduced:

- A: beta ≈ **0.158**, p ≈ **0.001**
- G: beta ≈ **0.107**, p ≈ **0.00013**

### 2021–23
The simple haematological scalar no longer reproduced:

- A X0: beta ≈ **0.055**, p ≈ **0.73**

The glycaemic scalar remained positive:

- G X0: beta ≈ **0.200**, p ≈ **0.006**

The fully adjusted X3 model was design-limited in the single modern NHANES cycle.

### Script 33
`33_audit_2021_2023_transfer_failure.py`

The modern A failure was audited rather than automatically attributed to a biological cause.

Potential explanations considered included:

- survey design;
- measurement context;
- changing population composition;
- missingness;
- assay context;
- broader distribution shift.

No causal explanation was assigned.

This failure became scientifically useful:

> A scalar representation can work in discovery, reproduce for a decade, and still fail under later distribution shift.

---

## Scripts 34–36 — Evidence Consolidation

- `34_build_presentation_evidence_pack.py`
- `35_cleanup_presentation_visuals.py`
- `36_build_final_presentation_candidates.py`

The accumulated discovery and temporal results were consolidated into an auditable evidence pack and presentation-ready outputs.

At this point the research question had shifted from simple association toward **representation quality, preservation, separability, and transfer**.

---

# Phase 3 — Component-Level Physiological Analysis

## Scripts 37–38 — Component-Level Phenotype Mapping

### Script 37
`37_component_level_phenotype_models.py`

The physiological PCA components were tested directly against the depressive phenotype rather than treating the entire A or G block as a single number.

A major result emerged:

- information beyond A-PC1 remained;
- A-PC2/3 jointly added phenotype information;
- scalar A did not recover the information contained in the multivariate A representation.

This suggested that haematological phenotype information was not confined to a simple haemoglobin-deficit direction.

### Script 38
`38_cross_cycle_component_replication.py`

The component-level relationships were challenged across independent discovery cycles.

Specific A-component directions showed reproducibility, motivating construction of a final reduced and frozen haematological representation.

---

# Phase 4 — Final Reduced Haematological Representation

## Script 39 — Reduced A Representation

`39_freeze_reduced_A_representation.py`

The earlier seven-variable CBC representation contained substantial algebraic redundancy.

The final reduced basis became:

- haemoglobin;
- RBC count;
- MCV;
- RDW.

Complete reduced-CBC sample:

**n = 8,686**

Three components were retained outcome-independently.

Approximate variance:

- PC1: **47–48%**
- PC2: **36–37%**
- PC3: **14–16%**
- PC4: only **~0.6–0.7%**

Cross-cycle loading similarity:

- PC1 ≈ **0.993**
- PC2 ≈ **0.993**
- PC3 ≈ **0.999**

Final pooled reconstruction:

**~99.2%**

Frozen 2005–06 → 2007–08 reconstruction:

**~98.9%**

Conclusion:

> The reduced haematological state can be compressed into three stable dimensions while preserving essentially all measured information.

No depression outcome was used to construct the representation.

---

## Script 40 — Frozen A Temporal Projection

`40_project_frozen_A_temporal.py`

The discovery A transform was projected unchanged into later NHANES periods.

Reconstruction:

- 2009–18: **~99.3%**
- 2021–23: **~99.6%**

Therefore the **haematological coordinate system itself remained structurally valid**, even when the crude Hb-deficit scalar weakened later.

This established an important distinction:

> representation stability ≠ phenotype-mapping stability.

---

# Phase 5 — Deep Glycaemic Representation

## Script 41 — Two-Marker Baseline

`41_freeze_and_transfer_G_representation.py`

Initial glycaemic representation:

- HbA1c
- fasting glucose

The space was dominated by one common dimension:

- PC1 ≈ **91%** variance in discovery;
- similar structure persisted temporally.

This was treated only as a minimal glycaemic measurement baseline.

---

## Script 42 — Deep G Structure and Assay Audit

`42_deep_G_structure_and_assay_audit.py`

Core transferable representation:

`G3 = {HbA1c, fasting glucose, log fasting insulin}`

Extended representation:

`G4 = {HbA1c, fasting glucose, log fasting insulin, 2h OGTT}`

### G3 discovery

n ≈ **4,145**

Variance:

- PC1 ≈ **63%**
- PC2 ≈ **31%**
- PC3 ≈ **5–6%**

Cross-cycle loading similarities were all approximately **0.999+**.

Physiological anchors using log-HOMA-IR:

- G-PC1: r ≈ **0.556**
- G-PC2: r ≈ **0.828**
- G-PC3: r ≈ **0.037**

Working interpretation:

- G-PC1 = broad/common glycaemic state;
- G-PC2 = insulin-related dimension;
- G-PC3 = smaller glucose-measurement discordance dimension.

These remained descriptive labels, not final mechanistic identities.

Insulin assay changes across NHANES periods were explicitly audited rather than silently calibrated away.

---

## Script 43 — Deep G Phenotype Mapping

`43_deep_G_phenotype_mapping.py`

G3 and G4 representations were tested against somatic depressive phenotype using survey-aware inference.

G3 contributed phenotype information alongside the haematological representation.

However, much of the transferable glycaemic signal remained concentrated in the common G-PC1 axis.

This differed from haematology, where meaningful information was distributed across multiple axes.

---

# Phase 6 — Final Combined A + G Representation

## Script 44 — Combined Discovery Model

`44_final_reduced_A_deep_G3_combined_discovery.py`

Final discovery representation:

- 3 frozen A components
- 3 G3 components

Joint fasting phenotype sample:

**n ≈ 3,815–3,830**, depending on outcome completeness.

Somatic phenotype results showed:

- A PCs given G PCs + covariates: significant;
- G PCs given A PCs + covariates: significant;
- A PCs beyond scalar A/G: significant;
- scalar burdens beyond PCs: essentially no added information.

Approximate discovery information gains:

- scalar A+G beyond X: ΔR² ≈ **0.0046**
- all six PCs beyond X: ΔR² ≈ **0.015**
- PCs beyond scalars: ΔR² ≈ **0.010–0.011**
- scalars beyond PCs: approximately **zero**

Cross-block A/G component correlations were generally weak.

Maximum |r| ≈ **0.20**

Conclusion:

> The A and G component spaces were statistically distinguishable and the multivariate representation retained considerably more phenotype-associated information than threshold scalars.

This did **not** establish biological independence.

---

# Phase 7 — Frozen Combined Temporal Transfer

## Script 45

`45_frozen_combined_AG_temporal_phenotype_transfer.py`

The entire discovery representation was frozen and projected forward.

No refitting.

No future re-standardization.

### 2009–18 replication

The multivariate advantage formally reproduced.

Approximate findings:

- A PCs given G PCs + X3: p ≈ **7 × 10^-5**
- G PCs given A PCs + X3: p ≈ **0.0058**
- A-PC2/3 beyond A-PC1: p ≈ **0.0019**
- G-PC2/3 beyond G-PC1: p ≈ **0.84**
- all PCs beyond scalar A+G: p ≈ **0.0066**
- scalars beyond all PCs: p ≈ **0.18**

Specific replicated mappings included:

- A-PC1: negative somatic association;
- A-PC3: positive somatic association;
- G-PC1: positive somatic association.

Thus transfer was not merely a global R² phenomenon.

Specific physiological component–phenotype relationships also persisted.

### 2021–23

Formal full-model survey inference was design-limited.

However, the information pattern remained:

- PCs contained substantially more phenotype information than the scalar model;
- the multidimensional A representation remained intact despite weakening of the simple scalar-A result.

This suggested that phenotype-relevant information can redistribute across physiological dimensions while the underlying representation remains structurally preserved.

---

# 15–16 September — Outcome-Independent Refreeze and Falsification

## Script 46 — Refreeze Outcome-Independent Representations

`46_refreeze_outcome_independent_representations.py`

The final A and G representations were rebuilt/frozen under an explicitly outcome-independent pipeline.

This removed any remaining ambiguity about outcome leakage.

The representation construction remained independent of PHQ/depression.

---

## Script 47 — Domain-Corrected Combined Inference

`47_domain_corrected_combined_AG_inference.py`

The complex-survey implementation was corrected to preserve the full survey design before applying analysis-domain restrictions.

This became the authoritative inferential scaffold.

Final primary discovery somatic sample:

**n = 3,830**

Final 2009–18 replication:

**n = 9,919**

The component-over-scalar result remained.

---

## Script 48 — Comparator and Sensitivity Audit

`48_comparator_and_sensitivity_audit.py`

Several alternative explanations were challenged.

### Raw physiology vs PCA

Raw physiology performed extremely similarly to the PCA representation.

Difference in weighted R²:

- discovery: approximately **0.0007**
- replication: approximately **0.00003**

Raw variables still added information beyond threshold scalars.

Interpretation:

> PCA is not manufacturing new information.

It is acting mainly as a compact, stable coordinate system for information already present in the raw physiology.

### Model sensitivity

The result survived:

- quasi-Poisson outcome modelling;
- removal of insulin;
- removal of HbA1c;
- alternative physiological representations.

This strengthened the interpretation that the result was not dependent on one marker or one exact regression family.

---

# 16 September — Interaction, Temporal Drift, and G4 Challenge

## Script 49 — Interaction and Temporal Drift Audit

`49_interaction_and_temporal_drift_audit.py`

Explicit temporal mapping tests were run.

Time was represented as ordered NHANES cycle.

Component × time terms tested whether phenotype mapping changed while keeping the physiological coordinates frozen.

Results:

### Haematology
Linear temporal drift:

**p ≈ 0.858**

No evidence that the A-component mapping changed systematically with time.

### Glycaemia
G block temporal drift:

**p ≈ 0.0189**

Evidence of block-level temporal change.

However:

- no individual G temporal interaction survived BH correction;
- minimum BH-adjusted p ≈ **0.061**.

Arbitrary cycle-specific slope heterogeneity:

- A: p ≈ **0.605**
- G: p ≈ **0.131**

Interpretation:

> There is some evidence that the overall glycaemic phenotype mapping changes through time, but not enough evidence to identify one specific glycaemic component as the stable source of that drift.

---

## Script 50 — G4 Extension and Modifier Audit

`50_g4_extension_and_modifier_audit.py`

2-hour OGTT was added as a fourth glycaemic marker.

G4 produced additional physiological structure but only tiny phenotype-model improvement.

OGTT beyond G3:

- discovery: p ≈ **0.42**
- replication: p ≈ **0.06**

Incremental R² gain from G4 was approximately:

- discovery: **0.00015**
- replication: **0.00095**

No G4 modifier survived multiplicity correction.

Therefore G3 remained the preferred transferable glycaemic representation.

G4 is physiologically richer, but its extra complexity was not justified for the core temporal model.

---

# 17 September — Synthetic Longitudinal Ground-Truth Experiment

A separate `Synthetic_Longitudinal` analysis branch was created.

Important:

The script numbering in this branch overlaps later NHANES script numbers; these are separate experiment namespaces.

---

## Simulation Script 51 — Calibration Cohort

`51_prepare_simulation_calibration.py`

A simulation calibration cohort was constructed using **only NHANES 2005–08 discovery information**.

Leakage guards explicitly prevented future NHANES cycles from informing the simulator calibration.

The frozen outcome-independent A/G representations from Script 46 were used as the real-data reference.

Purpose:

> Build synthetic longitudinal experiments without pretending that repeated cross-sectional NHANES provides true individual trajectories.

---

## Simulation Script 52 — Synthetic Baseline Humans

`52_generate_synthetic_humans.py`

A synthetic baseline population was generated from the discovery-period physiological and demographic structure.

These participants were synthetic individuals rather than copied longitudinal NHANES subjects.

---

## Simulation Script 53 — Longitudinal Lives V1

`53_simulate_longitudinal_lives.py`

Synthetic individuals were evolved for approximately 15 years.

Hidden physiological truth was deliberately defined separately from the PCA representation.

Four prespecified worlds were generated:

1. `stable_additive`
2. `mapping_drift`
3. `ag_interaction`
4. `measurement_shift`

This gave known ground truth for questions that NHANES itself cannot answer.

### Stable additive
A and G contribute without a true interaction or mapping drift.

### Mapping drift
A/G → phenotype coefficients change with time.

### A×G interaction
A true interaction is deliberately introduced.

### Measurement shift
Observed HbA1c changes because of an assay-like shift despite the underlying glycaemic state remaining unchanged.

---

## Simulation Script 54 — Frozen Dynamic Framework Test

`54_test_dynamic_framework.py`

The framework was tested **without access to hidden truth**.

Models included:

- persistence;
- scalar additive;
- component additive;
- component + time;
- component + time + A×G interactions.

Ridge regression was used as a stable initial comparison model.

Training:

- years 0–4

Validation:

- years 5–9

Future test:

- years 10–14

Additionally:

**20% of synthetic people were never used for model fitting.**

This created both:

- future-time testing;
- unseen-person testing.

The frozen Script 46 physiological representation was projected into all synthetic years without refitting.

---

## Simulation Script 55 — Hidden-Truth Validation

`55_validate_against_hidden_truth.py`

Only **after predictions had been frozen** was the hidden simulator truth opened.

Hash checks ensured predictions could not be changed after seeing the answer.

This allowed three separate questions:

1. Do components outperform scalars when the true physiological state is known?
2. Do explicit time terms help specifically when the mapping truly drifts?
3. Do A×G terms help when an interaction truly exists?

This was the first controlled ground-truth validation of the framework logic.

---

## Simulation Script 56 — Reality Check Against Later NHANES

`56_compare_synthetic_to_future_nhanes.py`

Synthetic trajectories were compared against actual later repeated-cross-sectional NHANES cycles.

Comparison cycles:

- 2007–08
- 2009–10
- 2011–12
- 2013–14
- 2015–16
- 2017–18

2021–23 was deliberately left untouched.

Age/sex post-stratification was used before comparing synthetic and actual distributions.

The first simulator showed non-trivial discrepancies from later real NHANES, with some maximum standardized differences around or above 1 SD.

This was treated as simulator misspecification rather than ignored.

---

# Synthetic Longitudinal V2

## Script 57 — Corrected Longitudinal Lives

`57_simulate_longitudinal_lives_v2.py`

The first simulator had an important phenotype-dynamics problem:

the autoregressive formulation could pull individual depressive phenotype trajectories toward zero rather than preserving an individual's baseline symptom set-point.

V2 corrected this.

Each synthetic person was assigned an individual baseline latent set-point.

Temporal persistence then occurred **around that set-point** rather than shrinking everyone toward zero.

Physiological transition assumptions were otherwise kept unchanged.

This was an important failure-and-correction step.

---

## Scripts 58–60 — V2 Framework, Truth Audit, Reality Check

- `58_test_dynamic_framework_v2.py`
- `59_validate_against_hidden_truth_v2.py`
- `60_compare_synthetic_v2_to_future_nhanes.py`

The full frozen prediction → hidden-truth validation → later-NHANES reality-check pipeline was rerun using the corrected simulator.

The design preserved:

- chronological train/validation/test splitting;
- unseen-person holdout;
- frozen A/G transforms;
- hidden-truth isolation;
- hash-verified prediction freezing.

The synthetic experiments established the conceptual distinction:

> **representation stability asks whether the physiological coordinate system survives; mapping stability asks whether its relationship with the phenotype survives.**

They also demonstrated why time-dependent terms should only be rewarded when the underlying mapping actually changes.

The simulator was treated as a **methodological sandbox**, not as evidence that NHANES contains true individual longitudinal trajectories.

---

# 21 September — Formal Lock-In / Falsification Suite

## NHANES Script 51 — Lock-In Falsification Suite

`51_lock_in_falsification_suite.py`

A formal attempt was made to break the core result before moving forward.

Final status:

**53 PASS**
**2 WARN**
**0 FAIL**
**3 INFO**
**7 UNTESTABLE**

The core result survived:

- survey-design checks;
- adjustment-ladder tests;
- raw-variable comparisons;
- outcome sensitivities;
- marker-removal sensitivities;
- multiplicity correction;
- temporal heterogeneity checks;
- complexity controls;
- VIF checks;
- selection audits;
- reproducibility checks.

Adjustment ladder:

### Discovery
PCs beyond scalars:

- X0: p ≈ **4.3 × 10^-5**
- X1: p ≈ **0.00043**
- X2: p ≈ **0.010**
- X3: p ≈ **0.014**

### 2009–18
PCs beyond scalars remained significant through X3:

- X3: p ≈ **0.0061**

No adjustment set showed a supported reverse advantage for the threshold scalars.

Locked claim:

> **In NHANES, a frozen outcome-independent multidimensional physiological representation retains phenotype-associated information beyond simple threshold scalars, and that advantage reproduces under frozen 2009–18 temporal transfer.**

Explicit boundary:

> **The result remains associative, not causal.**

Irreducible limitations included:

- unmeasured confounding;
- reverse causation;
- collider / overadjustment ambiguity;
- MNAR missingness;
- external transportability;
- true longitudinal change.

---

# 24 September — Mechanistic / Interpretability Audit

The work then moved from:

`Does the representation work?`

to:

`Why does it work?`

---

## EDA Script 58 — Cohort and PHQ Structure

`EDA/58_cohort_phq_eda.py`

The locked analysis scaffold was used to reconstruct raw PHQ items and physiological variables without touching locked outputs.

Primary somatic X3 samples:

- 2005–08: **n = 3,830**
- 2009–18: **n = 9,919**
- 2021–23: **n = 2,261**

Full PHQ-9 samples:

- 2005–08: **n = 3,815**
- 2009–18: **n = 9,893**
- 2021–23: **n = 2,247**

Weighted PHQ-9 zero prevalence:

- 2005–08: **33.1%**
- 2009–18: **32.4%**
- 2021–23: **25.1%**

Thus the modern cohort did not contain fewer depressive symptoms.

Descriptively, depressive symptom burden was greater.

---

## EDA Script 59 — Clinical Overlap

`EDA/59_clinical_overlap_eda.py`

Threshold groups were explicitly counted.

Weighted prevalence:

### 2005–08
- anaemia: **4.94%**
- diabetes-range HbA1c: **6.39%**
- PHQ-9 ≥ 10: **6.26%**
- all three: **0.038%**

### 2009–18
- anaemia: **6.33%**
- diabetes-range HbA1c: **8.95%**
- PHQ-9 ≥ 10: **7.67%**
- all three: **0.086%**

### 2021–23
- anaemia: **7.30%**
- diabetes-range HbA1c: **9.78%**
- PHQ-9 ≥ 10: **12.02%**
- all three: **0.097%**

The triple-threshold subgroup is therefore extremely sparse.

The main multivariate effect cannot simply be interpreted as a special "anaemia + diabetes + depression" disease subgroup.

---

## EDA Script 60 — Non-Anaemic Signal Audit

`EDA/60_nonanaemic_signal_audit.py`

The critical diagnostic question became:

> Does the haematological representation still contain phenotype information among people for whom the anaemia scalar is exactly zero?

Weighted proportion with `A = 0`:

- 2005–08: **95.0%**
- 2009–18: **93.7%**
- 2021–23: **92.6%**

Thus the scalar collapses the overwhelming majority of the population to exactly the same value.

Yet among these non-anaemic participants:

### Somatic phenotype

A-PC block:

- 2005–08: p ≈ **0.0075**
- 2009–18: p ≈ **0.00066**

RBC + MCV + RDW beyond continuous Hb:

- 2005–08: p ≈ **0.021**
- 2009–18: p ≈ **0.00094**

Hb beyond RBC + MCV + RDW:

- 2005–08: p ≈ **0.37**
- 2009–18: p ≈ **0.97**

This provided the first direct explanation of the scalar information-loss result:

> The PCA advantage is not simply caused by retaining continuous Hb above the anaemia threshold.

Other erythrocyte characteristics contain phenotype-associated information.

---

## EDA Script 61 — Component Decoding

`EDA/61_component_decoding.py`

The existing frozen PCA was decoded without refitting.

### A-PC1
Predominantly an **Hb / RBC abundance axis**.

### A-PC2
Predominantly an **MCV versus RBC-count contrast**.

### A-PC3
Predominantly an **RDW / red-cell heterogeneity axis**, with additional MCV contribution.

Among non-anaemic participants, PC3–RDW correlation:

- 2005–08: **~0.58**
- 2009–18: **~0.67**
- 2021–23: **~0.68**

PC3 somatic association:

- 2005–08: beta ≈ **+0.172**, BH p ≈ **0.032**
- 2009–18: beta ≈ **+0.126**, BH p ≈ **0.0014**

Raw RDW independently reproduced the same direction:

- 2005–08: beta ≈ **+0.230**, p ≈ **0.0064**
- 2009–18: beta ≈ **+0.143**, p ≈ **0.00015**

Thus the working explanation became:

`Hb threshold`
→ loses most non-anaemic variation

whereas

`Hb + RBC + MCV + RDW`
→ preserves continuous red-cell state

and one particularly reproducible phenotype-relevant direction is:

`RDW / erythrocyte heterogeneity-like state`.

This remains an associative physiological interpretation, not a causal mechanism.

---

# New Research Strategy

The project has now entered a **differential-diagnosis phase**.

Instead of treating PCA as a black box, each observed model behaviour will be treated as a symptom requiring competing explanations.

Questions now include:

1. Why exactly does A-PCA outperform scalar A?
2. Why does G-PCA gain less over scalar G?
3. What physiological state does each A-PC and G-PC represent?
4. Why is the A×G interaction weak / unstable?
5. Are there specific A_i × G_j interactions hidden by the coarse scalar product?
6. Which covariates attenuate which component effects?
7. Is the signal concentrated in a small number of dimensions or distributed?
8. Would raw variables, sparse PCA, ICA, factor analysis, Ridge, LASSO, or Elastic Net yield the same physiological conclusion?
9. Which interpretations remain stable under temporal transfer?

Future sparse / regularized modelling may also be used as a mechanism-discovery tool for larger candidate libraries containing:

- component main effects;
- nonlinear terms;
- A_i × G_j terms;
- component × time terms;
- eventual dynamical terms.

The objective is no longer simply to show that PCA works.

The objective is to determine:

> **why it works, what physiological information it preserves, why simple clinical scalars lose that information, and under what conditions the representation, mapping, and interactions remain valid.**

# 25 Sep 2026 — Research Log

## Representation mechanics
- Extended the frozen PCA baseline with an ICA/source-space rotation.
- Tested bidirectional physiology ↔ latent mapping:
  X → Z → X → Z → X
- Haematology remained highly reconstructable despite 4D → 3D compression.
- Glycaemia was effectively lossless in the current 3D → 3D setup.
- Frozen mappings remained stable across 2009–18 and 2021–23.
- Independent future ICA solutions strongly rediscovered the discovery source structure.

## Jacobian / sensitivity experiment
- Computed dZ/dX and dX/dZ for the learned linear maps.
- No measured biomarker was completely ignored.
- Haematology showed one null direction, as expected from 4D → 3D compression.
- Glycaemia had no null direction.
- Component sensitivities matched the earlier physiological interpretations.

## Phenotype bidirectional experiment
Tested the actual phenotype as Y:

X → Y → X → Y → X

- Physiology → somatic phenotype explained ~4.9% of holdout variance.
- Phenotype → physiology was a poor inverse.
- Repeating the phenotype cycle did not recover the original physiology.
- Therefore the reversible structure belongs to the physiological representation, not to the phenotype itself.

## Architecture clarification
From now on:

X = raw physiology  
Z = physiological decomposition / latent representation  
Y = phenotype

Working structure:

X ↔ Z → Y

Z → X is the reconstruction/decoder route.
Y → X is not treated as a unique inverse because many physiological states may produce the same phenotype.

## Later experiments
- Phenotype decomposition.
- Derivatives of the actual phenotype mapping.
- Latent-space simulation using Z → X.
- VAE / β-VAE benchmark against the current linear system.

## Stop point
The physiology/latent architecture is now much clearer, but phenotype decomposition and the full conceptual architecture needs to be revisited fresh rather than pushed further today.