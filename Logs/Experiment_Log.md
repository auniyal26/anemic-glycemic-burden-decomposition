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