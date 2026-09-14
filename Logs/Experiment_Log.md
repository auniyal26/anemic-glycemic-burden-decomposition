# Research Log — 20 August 2026

## Project
**Hematological and Glycemic Decomposition of Depression**

## Goal for Today
Test whether the proposed decomposition framework is empirically viable using NHANES 2005–06 and 2007–08 data before moving further into the PhD proposal and presentation.

## Conceptual Framework
The working mechanism is:

1. Preserve the original mapping/reference:
   - `x -> y`
   - store `(x, y)` before further transformation.
2. Work on copies rather than overwriting the original representation.
3. Replace one large monolithic function with smaller components.
4. Quantify information/reconstruction loss after every transformation.
5. Use domain knowledge to preserve clinically meaningful structure.
6. Decompose residual/latent structure where appropriate.
7. Validate whether the recovered components are stable and reproducible.

For this problem:

- `A` = hematological burden
- `G` = glycemic burden
- `P` = depressive phenotype

Initial model:

`P = f(A, G, A×G, X)`

where `X` contains demographic/confounding variables.

---

## Data Used

### NHANES 2005–06
- CBC / hemoglobin
- demographics
- PHQ-9
- HbA1c
- fasting glucose
- retinal grading

### NHANES 2007–08
Same modalities as above.

### Initial overlap
All six files matched:
- 2005–06: **1,393**
- 2007–08: **1,906**

For the core A/G/P experiment, retinal and fasting-glucose files were not forced into the merge.

### Core matched dataset
CBC + HbA1c + PHQ-9 + demographics:

- Raw matched: **11,329**
- Complete Hb: **10,640**
- Complete HbA1c: **10,632**
- Complete PHQ-9: **10,214**
- Complete A/G/P: **9,743**
- PHQ-9 >= 10: **776**

---

## Scripts Created / Experiments Run

### 01 — Dataset Inspection
`01_inspect_nhanes.py`

Purpose:
- inspect all XPT files
- list variables
- measure `SEQN` overlap

Important issue found:
- the first downloaded `DEMO_D.xpt` had accidentally duplicated CBC content
- corrected file produced the expected 43 demographic variables

---

### 02 — Core Dataset Construction
`02_build_core.py`

Purpose:
- merge CBC + demographics + HbA1c + PHQ-9
- preserve a compressed reference dataset
- create a working copy

Important bug found:
NHANES encoded PHQ zero values as approximately:

`5.397605e-79`

instead of literal `0`.

This initially produced only 97 complete PHQ cases.

After correcting near-zero values to zero:

- complete A/G/P increased to **9,743**
- mean PHQ-9 = **3.03**
- PHQ-9 >= 10 = **776**

This confirmed the dataset was valid.

---

### 03 — Naive Linear Decomposition
`03_test_decomposition.py`

Naive anemia representation:

`A = -z(Hb)`

Glycemia:

`G = z(HbA1c)`

Results:

- anemia signal: not significant
- glycemia signal: significant
- A×G interaction: not significant
- reconstruction error: **0**

Final model:
- `A beta = 0.0033, p = .961`
- `G beta = 0.3113, p ≈ 5e-6`
- `AG beta = 0.0124, p = .869`
- `R² ≈ .023`

Interpretation:
The algebraic decomposition reconstructed perfectly, but the naive anemia representation was clinically poor.

---

### 04 — Domain-Informed Decomposition
`04_domain_informed_decomposition.py`

Anemia burden changed to a clinically meaningful deficit:

`A* = max(Hb_threshold(sex) - Hb, 0)`

Thresholds:
- male: 13 g/dL
- female: 12 g/dL

Glycemic burden:

`G* = max(HbA1c - 5.7, 0)`

Interaction:

`AG = A* × G*`

Clinical-group results:

| Group | N | Mean PHQ-9 | Depression prevalence |
|---|---:|---:|---:|
| Neither | 6193 | 2.74 | 6.10% |
| Dysglycemia only | 2714 | 3.04 | 8.00% |
| Anemia only | 485 | 3.34 | 7.89% |
| Both | 351 | **4.05** | **10.00%** |

Domain-informed model:

- `A beta = 0.640, p = .000107`
- `G beta = 0.320, p = .000050`
- `AG beta = -0.268, p = .047`

Key observation:
The anemia signal that disappeared under the naive representation reappeared strongly after applying domain knowledge.

This became the first empirical support for the idea that representation choice can erase useful information and that domain-informed transformations can preserve/recover meaningful structure.

---

### 05 — Cross-Cycle Replication + Nonlinear Interaction
`05_replication_nonlinear.py`

Linear replication:

#### 2005–06
- A: borderline/additive; significant with interaction model
- G: significant
- AG interaction: significant

#### 2007–08
- A: significant
- G: significant
- AG interaction: not significant

Nonlinear pooled models improved training fit slightly, but cross-cycle generalization became worse.

Cross-cycle nonlinear interaction:
- 0506 -> 0708: negative test R²
- 0708 -> 0506: negative test R²

Interpretation:
The interaction term was not stable.

Current evidence favors:

`P ≈ f_A(A) + f_G(G) + residual`

rather than a large reproducible interaction function.

This also supported the idea of smaller functions instead of a single large model.

---

### 06 — First Residual ICA Attempt
`06_residual_ica.py`

Initial design:
- regress PHQ items on A, G and confounders
- apply ICA to the residuals

Result:
All ICA component associations with A/G became exactly zero.

Reason:
A and G had already been explicitly regressed out before ICA.

This was not a biological result; it exposed a flaw in the experiment design.

---

### 07 — Corrected Latent PHQ Decomposition
`07_latent_phq_decomposition.py`

Corrected design:
- remove confounders only
- preserve A and G information
- perform ICA on the 9-dimensional PHQ residual phenotype
- test latent components against A/G afterward

Four-component ICA retained approximately **70.5%** of the standardized residual information.

Interpretable components emerged:

- IC1: affective / anhedonia / depressed mood
- IC2: cognitive / psychomotor
- IC3: severe-affective / suicidal ideation
- IC4: somatic — sleep, fatigue, appetite

Important associations:
Anemia showed reproducible relationships especially with IC3 and IC4 in pooled analyses.

This suggested that the total PHQ score may hide physiologically distinct depressive symptom structures.

---

### 08 — Independent Cross-Cycle Latent Replication
`08_cross_cycle_latent_replications.py`

ICA was run independently in each NHANES cycle and components were aligned only afterward.

Component similarities:

- Component 1: **0.859**
- Component 2: **0.875**
- Component 3: **0.956**
- Component 4: **0.561**

Interpretation:
Three latent structures reproduced strongly across completely independent decompositions.

The fourth component was much less stable.

This was considerably stronger evidence than the pooled ICA result.

---

### 09 — ICA Initialization Stability
`09_ica_stability.py`

ICA repeated across 100 random seeds.

#### 2005–06
- IC1 mean similarity: **0.997**
- IC2: **0.9997**
- IC3: **0.9999**
- IC4: **0.980**, but minimum dropped to 0.488

#### 2007–08
All four components were approximately **0.9999+** stable.

Interpretation:
IC1–IC3 are extremely stable to ICA initialization.

IC4 is less reliable in 2005–06, consistent with its weaker cross-cycle replication.

Current evidence therefore points toward approximately **three strongly reproducible latent components**, rather than assuming four components are inherently correct.

---

## Current Scientific Interpretation

The work today does **not prove the final theory**, but it provides a credible proof-of-concept.

What survived:

1. Original data/reference can be preserved while transformations are performed on working representations.
2. Decomposition and reconstruction can be explicitly checked.
3. Domain-informed representations recovered signal that a naive representation lost.
4. Smaller physiological components A and G behaved more robustly than a large nonlinear interaction model.
5. Depression did not behave as a single homogeneous variable.
6. ICA recovered latent PHQ structures.
7. Three of those structures independently replicated across NHANES cycles.
8. Those components were highly stable to ICA initialization.

Current working conclusion:

**Core mechanism is empirically viable, but not yet proven.**

---

## Important Limitations

- Current regression approach uses weighted/cluster-robust models but is not yet a complete NHANES complex-survey implementation.
- Cross-sectional NHANES data cannot establish causality.
- Current physiological effect sizes / R² values are small.
- ICA stability to initialization is not the same as stability to sampling.
- The fourth latent component is unstable across cycles.
- Interaction A×G is not reliably replicated.
- External population validation is still required.
- The framework has currently been demonstrated only in this problem/domain.

---

## Next Steps

### Immediate
1. **Bootstrap / sampling stability**
   - resample participants
   - rerun the full ICA decomposition
   - align components
   - quantify component stability under sampling

2. **Determine the correct number of components**
   - compare 2, 3, 4, 5-component solutions
   - balance reconstruction loss vs stability
   - avoid forcing unnecessary latent dimensions

3. **Proper NHANES survey analysis**
   - strata
   - PSU
   - correct combined-cycle weights
   - sensitivity checks

4. **Latent-component interpretation**
   - formalize item loading patterns
   - compare with known PHQ factor structures
   - test somatic vs cognitive-affective interpretations

### After that
5. Replicate with fasting glucose as an alternative glycemic variable.
6. Add hematological features beyond Hb:
   - Hct
   - MCV
   - MCH
   - RDW
   - RBC count
7. Test latent hematological burden instead of a single Hb threshold.
8. Compare linear decomposition, GAMs, SEM/CFA and latent-variable models.
9. Begin external/Indian validation.
10. Use these results in the PhD proposal presentation as preliminary feasibility evidence.

---

## Status at End of Day

**Good start.**

The proposed mechanism did not collapse under the first empirical tests.

The strongest result today is not the interaction term. It is the sequence:

`naive representation -> signal loss -> domain-informed representation -> signal recovery -> latent decomposition -> independent replication -> algorithmic stability`

This is currently sufficient to justify continuing the PhD direction and designing the next validation experiments.


# Research Log — 2026-08-21

## Project
**Disentangling Hematological and Glycemic Contributions to Depressive Symptoms**

## Objective
Continue testing whether hematological burden (A), glycemic burden (G), and their interaction (A×G) explain depressive phenotype, particularly somatic depressive symptoms, while preserving the original data and validating results across NHANES 2005–06 and 2007–08.

---

## 1. Data / Method Integrity

### Script 13 — Full Integrity Audit
- Verified all required raw NHANES files.
- Verified unique SEQN identifiers.
- Verified PHQ-9 encoding.
- Independently rebuilt the core dataset.
- Confirmed final complete A/G/P sample: **N = 9,743**.
- Verified raw-file hashes and transformation ledger.
- Confirmed K=3 latent component replication.
- Corrected earlier interpretation of ICA component 3:
  - dominant items = **sleep + fatigue**
  - therefore component is **somatic**, not severe-affective.

### Script 14 — Freeze Exploratory V1
Frozen:
`Audit/Exploratory_Analysis_V1`

Preserved:
- raw hashes
- transformation parameters
- ICA parameters
- scaler parameters
- residual models
- participant mappings
- figures
- seeds
- thresholds

This substantially implements the reference-preserving framework proposed after discussion with Prof. Saeed.

---

## 2. Confounding / DAG / Survey Design

### Script 15 — X + DAG Audit
Created staged adjustment sets:

- X0 = age + sex + race/ethnicity
- X1 = X0 + income + education + smoking
- X2 = X1 + BMI
- X3 = X2 + kidney function

Additional decisions:
- age ≥20
- known pregnancy excluded
- kidney function treated cautiously as potentially pathway-dependent
- HbA1c measurement coupling with RBC/anemia explicitly acknowledged
- NHANES weights, strata and PSU retained

### Script 16 — Staged Survey Models
Matched survey sample:

**n = 7,957**

Results:

#### Total PHQ-9
- A survives
- G survives
- A×G: no clear support

#### Somatic score
Defined a priori as:
- sleep
- fatigue
- appetite

Results:
- A survives
- G survives
- A×G survives pooled analysis

The interaction was **negative/sub-additive**, not synergistic.

---

## 3. Independent Cycle Replication

### Script 17 — Cross-Cycle Somatic Replication
Samples:
- 2005–06: n = 3,582
- 2007–08: n = 4,375

Somatic A:
- stronger in 2007–08
- partial support in 2005–06

Somatic G:
- same positive direction in both cycles
- individually uncertain

Somatic A×G:
- 2005–06 X3 ≈ −0.145
- 2007–08 X3 ≈ −0.128

Direction and magnitude were extremely similar, but final confidence intervals narrowly crossed zero.

Conclusion:

**Direction and magnitude replicate; strict independent-cycle statistical significance does not.**

---

## 4. Alternative Glycemia Definition

### Script 18 — Fasting Glucose Sensitivity
Matched fasting subset:

**n = 3,838**

Used:
- fasting glucose
- correct fasting-subsample NHANES weights

Results:

#### HbA1c-based G
- A survives
- G same direction but uncertain
- A×G survives

#### Fasting-glucose G
- A survives
- G survives
- A×G same negative direction but uncertain

Important implication:

The glycemic association is **not simply an artifact of HbA1c being influenced by anemia/RBC turnover**.

A×G remains more representation-sensitive than the main A and G effects.

---

## 5. Clinical Diabetes-State Sensitivity

### Script 19 — Clinical Diabetes Status
Matched sample:

**n = 3,836**

States:
- normal
- prediabetes
- undiagnosed diabetes
- diagnosed untreated diabetes
- diagnosed treated diabetes

Results:
- anemia association became uncertain after clinical-state adjustment
- diagnosed treated diabetes showed the clearest somatic elevation
- no stable anemia × clinical-diabetes-state interaction

Conclusion:

Continuous glycemic physiology contains information that coarse diagnostic categories do not fully preserve.

---

## 6. Formal Depression Phenotype Validation

### Script 20 — Ordinal CFA
Used WLSMV for ordinal PHQ-9 items.

Results:
- two-factor model outperformed one-factor depression in both cycles
- somatic factor = sleep + fatigue + appetite
- all three items loaded strongly
- loadings approximately invariant across cycles
- ordinal thresholds approximately invariant across cycles

Therefore the somatic phenotype is no longer merely an ICA-derived exploratory pattern.

It is supported by formal ordinal factor analysis.

---

## 7. Latent Physiological Effects

### Script 21 — MIMIC / SEM
Model fit:

- CFI = 0.996
- RMSEA = 0.011
- SRMR = 0.039

Measurement structure constrained invariant across cycles.

### Somatic latent factor
- A: **REPLICATES**
- G: **REPLICATES**
- A×G: **PARTIAL**

### Cognitive-affective latent factor
- A: partial
- G: same direction, uncertain
- A×G: same direction, uncertain

Conclusion:

The strongest physiological signal is **somatic-specific**.

---

## 8. Method Convergence

### Script 22 — Convergence Audit

Results:

- A: **STRONG CONVERGENCE**
- G: **STRONG CONVERGENCE**
- A×G: **CONVERGENT, NOT FULLY REPLICATED**

Two distinct analysis routes agree:

1. complex-survey observed somatic score
2. ordinal latent somatic factor

Current core empirical result:

**Hematological and glycemic burden independently associate with a reproducible somatic depressive phenotype.**

A×G remains secondary/exploratory.

---

## 9. Differential Item Functioning

### Script 23 — MIMIC DIF
Ran:
- base invariant SEM
- 27 exposure × PHQ-item DIF models
- global Benjamini-Hochberg correction

Results:

- A: no FDR-stable direct item effect
- G: no FDR-stable direct item effect
- A×G: no FDR-stable direct item effect

Interpretation:

The physiological associations are not being driven by one rogue PHQ item.

A and G appear to act at the **latent somatic-factor level**, rather than only through fatigue, sleep, appetite, or another individual item.

---

## 10. Internal Composition of A and G

### Script 24 — Internal A/G Burden Decomposition

### Hematological block A
Variables:
- Hb
- hematocrit
- RBC count
- MCV
- MCH
- MCHC
- RDW

Complete sample:

**n = 8,686**

Independent-cycle PCA + alignment + 200 bootstrap checks.

Results:

#### PC1
- variance: 45.3% / 47.5%
- cross-cycle similarity: 0.999
- closest to current A proxy
- correlation with current A:
  - 2005–06: r = 0.560
  - 2007–08: r = 0.586

#### PC2
- variance: 34.4% / 32.4%
- similarity: 0.996

#### PC3
- variance: 11.7% / 10.8%
- similarity: 0.995

Therefore hematological state is strongly reproducible but **multidimensional**.

The current Hb-deficit A proxy captures only part of this structure.

### Glycemic block G
Variables:
- HbA1c
- fasting glucose

Complete fasting sample:

**n = 4,204**

#### PC1
- variance: 91.2% / 90.9%
- cross-cycle similarity: 1.000
- correlation with current HbA1c G proxy:
  - 2005–06: r = 0.936
  - 2007–08: r = 0.942

#### PC2
- variance: 8.8% / 9.1%
- similarity: 1.000
- represents HbA1c / fasting-glucose discordance

Therefore glycemic burden is largely one-dimensional in the current biomarker space.

---

# Current Scientific Position

The evidence now supports:

1. A reproducible somatic depressive phenotype exists.
2. Hematological burden associates with that phenotype.
3. Glycemic burden independently associates with that phenotype.
4. Both A and G replicate across independent NHANES cycles in latent models.
5. Results converge between survey-weighted observed-score and ordinal latent-variable approaches.
6. The signal is factor-level rather than being caused by one PHQ item.
7. Hematological burden is substantially multidimensional.
8. Glycemic burden is largely one-dimensional with the available glucose biomarkers.
9. A×G repeatedly shows a similar sub-additive direction, but is not yet sufficiently robust to treat as a primary finding.

# Current Interpretation

The original proxies were:

A ≈ hemoglobin deficit

G ≈ HbA1c excess

These are not equivalent representations.

G appears to approximate the underlying glycemic axis reasonably well.

A does not.

Therefore instability in A×G may partly arise because the original interaction multiplies a relatively incomplete hematological proxy by a much better glycemic proxy.

# Next Step

Use the stable internal hematological and glycemic components as candidate physiological burden axes and test:

- which A components predict the validated somatic latent phenotype
- which G components predict it
- whether specific A_i × G_j interactions reproduce across cycles

Do not yet label the PCA components as the final A or G.

The next goal is to identify physiologically interpretable, reproducible burden components before constructing the final decomposition model.

---

# Research Log — 2026-09-14

## Project
**Computational Physiological Decomposition — Hematological and Glycemic Testbed**

## Objective

Move from the earlier exploratory A/G decomposition toward a finalized, outcome-independent physiological representation and test whether:

1. hematological and glycemic state can be represented as stable multidimensional components,
2. those representations preserve information across time,
3. the decomposed representation contains depressive-phenotype information beyond the original scalar A and G burdens,
4. the discovery result transfers forward without refitting.

---

## 1. Final Reduced Hematological Representation

### Script 39 — Freeze Reduced A Representation
`39_freeze_reduced_A_representation.py`

Final hematological basis:

- hemoglobin
- RBC count
- MCV
- RDW

This replaced the earlier seven-variable exploratory CBC representation, which contained substantial algebraic redundancy.

### Discovery structure

Complete reduced-CBC sample:

**n = 8,686**

Cycle-specific explained variance:

#### PC1
- 2005–06: 47.4%
- 2007–08: 48.2%
- loading similarity: **0.993**

#### PC2
- 2005–06: 36.2%
- 2007–08: 37.3%
- loading similarity: **0.993**

#### PC3
- 2005–06: 15.7%
- 2007–08: 13.9%
- loading similarity: **0.999**

#### PC4
- approximately 0.6–0.7% variance

Outcome-independent retention criterion selected:

**K = 3**

Frozen 2005–06 -> 2007–08 reconstruction energy:

**0.989**

Final pooled 2005–08 reconstruction energy:

**0.992**

Conclusion:

The reduced hematological state can be represented by a stable three-dimensional structure while preserving >99% of the standardized information.

The final transform was frozen without using depression outcomes.

---

## 2. Frozen Hematological Temporal Projection

### Script 40 — Temporal A Projection
`40_project_frozen_A_temporal.py`

The 2005–08 hematological transform was projected into later NHANES cycles without refitting:

- means
- scales
- loadings
- component order
- component signs
- retained dimensionality

### Reconstruction

#### 2009–18
- reconstruction energy: **0.993**

#### 2021–23
- reconstruction energy: **0.996**

Every later cycle passed the structural transfer criterion.

Conclusion:

The hematological representation itself remains structurally stable through 2021–23.

Therefore the earlier failure of the simple scalar Hb-deficit A in 2021–23 cannot automatically be interpreted as failure of the underlying hematological state.

---

## 3. Glycemic Representation Revisited

### Script 41 — Two-Marker G Baseline
`41_freeze_and_transfer_G_representation.py`

Initial glycemic representation:

- HbA1c
- fasting glucose

The two-marker space remained dominated by one common axis:

- discovery PC1 variance: ~91%
- 2009–18: ~91%
- 2021–23: ~93%

This representation is now treated only as a:

**minimal glycemic measurement baseline**

and not as evidence that diabetes/glycemic physiology is inherently one-dimensional.

---

## 4. Deep Glycemic Representation

### Script 42 — Deep G Structure + Assay Audit
`42_deep_G_structure_and_assay_audit.py`

Core transferable glycemic representation:

`G3 = {HbA1c, fasting glucose, log fasting insulin}`

Extended representation:

`G4 = {HbA1c, fasting glucose, log fasting insulin, 2-hour OGTT glucose}`

### G3 discovery results

Sample:

**n = 4,145**

Variance:

#### 2005–06
- PC1 = 63.2%
- PC2 = 31.4%
- PC3 = 5.5%

#### 2007–08
- PC1 = 62.6%
- PC2 = 31.2%
- PC3 = 6.1%

Cross-cycle loading similarity:

- PC1 = **0.999883**
- PC2 = **0.999722**
- PC3 = **0.999837**

Therefore adding insulin reveals a clearly multidimensional glycemic state.

### Physiological anchors

Correlation with log-HOMA-IR:

- G3-PC1: **0.556**
- G3-PC2: **0.828**
- G3-PC3: **0.037**

This suggests:

- PC1 = broad/common glycemic state
- PC2 = strongly insulin-related dimension
- PC3 = smaller glucose-measurement discordance dimension

These remain descriptive labels rather than final mechanistic claims.

---

## 5. Extended G4 Representation

Adding 2-hour OGTT produced a richer structure:

- PC1: ~58–62%
- PC2: ~21%
- PC3: ~10–13%
- PC4: ~6–9%

Cross-cycle loading similarities remained high:

**~0.985–0.997**

OGTT therefore introduces additional metabolic/challenge-response information not contained in the simple HbA1c + fasting glucose representation.

Important limitation:

OGTT is unavailable after 2015–16, so G4 cannot serve as the full temporal representation through 2021–23.

---

## 6. Insulin Assay Audit

Insulin assay methods changed across NHANES eras.

Therefore:

- frozen projections were retained,
- correlation/loading structure was audited,
- no silent temporal calibration was performed,
- absolute insulin-containing score shifts are not automatically interpreted as biological.

Despite assay changes, the component geometry remained highly similar across later cycles.

This supports structural stability while preserving the assay caveat.

---

## 7. Deep G Phenotype Mapping

### Script 43 — G3/G4 Phenotype Mapping
`43_deep_G_phenotype_mapping.py`

Primary phenotype:

**somatic PHQ symptom score**

Primary adjustment:

**X3**

Formal inference:

`survey::svyglm + survey::regTermTest`

### G3 results

Sample:

**n = 3,815**

Somatic phenotype:

- G3 PCs jointly given A + X:
  - p = **0.0617**
- PC2–3 beyond PC1:
  - p = 0.233
- G3 PCs beyond scalar G:
  - p = 0.210
- scalar G beyond G3 PCs:
  - p = 0.803

G3-PC1:

- beta = **0.114**
- p = **0.0285**
- BH-adjusted p = 0.0855

Interpretation:

G3 is physiologically multidimensional, but the depression-relevant signal is concentrated mainly in the common glycemic dimension.

The deeper G3 representation does not independently demonstrate a strong information advantage over scalar G.

---

## 8. G4 Phenotype Mapping

Extended G4 produced stronger evidence.

Somatic phenotype:

- joint G4 block:
  - p = **0.0279**
- G4 PCs beyond scalar G:
  - p = 0.0959

For total PHQ-9, the richer G4 block showed clearer evidence that multidimensional metabolic/challenge physiology contains information beyond the HbA1c-based scalar.

Interpretation:

The simple scalar G appears to capture much of the transferable common glycemic signal, but richer metabolic measurements can reveal additional phenotype-relevant information.

---

## 9. Final Reduced A Phenotype Validation

### Script 44 — Final Reduced A + Deep G3
`44_final_reduced_A_deep_G3_combined_discovery.py`

This replaced the earlier phenotype result based on the legacy seven-variable CBC PCA.

Final reduced-A MEC sample:

**n = 7,957**

Somatic phenotype:

- A PCs jointly given scalar G + X:
  - p = **0.000130**
- A-PC2/3 beyond A-PC1:
  - p = **0.001040**
- A PCs beyond scalar A + scalar G:
  - p = **0.000724**
- scalar A beyond A PCs + scalar G:
  - p = 0.276

Conclusion:

The main hematological information-loss result survives using the finalized reduced and frozen representation.

The scalar Hb-deficit A does not preserve all phenotype-relevant hematological information.

---

## 10. Final Combined A + G3 Discovery Model

Joint fasting phenotype sample:

**n = 3,815**

Somatic phenotype:

- A PCs given G PCs + X:
  - p = **0.001546**
- G PCs given A PCs + X:
  - p = **0.027443**
- A PCs beyond scalar A + scalar G + G PCs:
  - p = **0.005464**
- G PCs beyond scalar G + scalar A + A PCs:
  - p = 0.148223
- all six PCs beyond scalar A + scalar G:
  - p = **0.014596**
- both scalar burdens beyond all six PCs:
  - p = **0.955583**

### Incremental information

Scalar A + G beyond X:

`ΔR² = 0.004638`

All six PCs beyond X:

`ΔR² = 0.015235`

All PCs beyond scalar A + G:

`ΔR² = 0.010642`

Both scalars beyond all PCs:

`ΔR² = 0.000046`

Conclusion:

The multidimensional physiological representation contains substantially more somatic-phenotype information than the conventional scalar A + G model.

The reverse is not true: the scalar burdens add essentially no information once the decomposed representation is present.

---

## 11. A/G Separability

Cross-block A/G component correlations were generally weak.

Maximum absolute correlation was approximately:

**0.20**

Therefore the recovered hematological and glycemic spaces remain statistically distinguishable.

This supports statistical nonredundancy, not biological independence.

---

## 12. Frozen Temporal Phenotype Transfer

### Script 45 — Frozen Combined A + G3 Temporal Transfer
`45_frozen_combined_AG_temporal_phenotype_transfer.py`

No representation refit.

No temporal re-standardization.

Discovery coordinates were projected forward unchanged.

Temporal physiology samples:

- 2009–18: **12,230**
- 2021–23: **3,010**

Phenotype-complete samples:

- 2009–18 X3: **9,893**
- 2021–23 X3: **2,247**
- 2021–23 X0 sensitivity: **2,590**

---

## 13. Formal 2009–18 Replication

Somatic phenotype:

- A PCs given G PCs + X3:
  - p = **0.000070**
- G PCs given A PCs + X3:
  - p = **0.005769**
- A-PC2/3 beyond A-PC1:
  - p = **0.001851**
- G-PC2/3 beyond G-PC1:
  - p = 0.840
- A PCs beyond scalar A + G + G PCs:
  - p = **0.000932**
- all six PCs beyond scalar A + scalar G:
  - p = **0.006570**
- both scalars beyond all PCs:
  - p = 0.176

### Incremental information

Scalar model beyond X:

`ΔR² = 0.003869`

All PCs beyond X:

`ΔR² = 0.007014`

All PCs beyond scalar model:

`ΔR² = 0.003643`

Scalars beyond all PCs:

`ΔR² = 0.000497`

Conclusion:

The discovery-stage decomposition advantage **formally replicates in pooled 2009–18** under the frozen X3 complex-survey model.

---

## 14. Component-Level Temporal Replication

2009–18 all-PC somatic model:

### A-PC1
- beta = **-0.129**
- BH-adjusted p = **0.00430**

### A-PC3
- beta = **+0.110**
- BH-adjusted p = **0.00139**

### G-PC1
- beta = **+0.097**
- BH-adjusted p = **0.00139**

These reproduce the major discovery directions.

Therefore temporal transfer is not limited to a global R² result; specific component-level phenotype mappings also persist.

---

## 15. 2021–23 Modern Holdout

The full X3 model became design-limited because the single NHANES cycle does not provide enough denominator survey degrees of freedom for the full parameterized block test.

Therefore confirmatory X3 p-values are not available.

X0 sensitivity remains design-limited as well and is not treated as confirmatory evidence.

However, the information pattern remains informative.

### X3 effect sizes

Scalar model beyond X:

`ΔR² = 0.002373`

All PCs beyond X:

`ΔR² = 0.010668`

All PCs beyond scalar model:

`ΔR² = 0.011688`

Scalars beyond all PCs:

`ΔR² = 0.003392`

Thus the decomposed representation continues to carry more somatic-phenotype information than the scalar model in the modern holdout, although formal complex-survey confirmation is not possible with the available design degrees of freedom.

---

## 16. Interpretation of the Earlier Scalar-A Failure

A notable pattern emerged across time.

Approximate A-PC1 somatic coefficients:

- discovery: **-0.222**
- 2009–18: **-0.129**
- 2021–23: **-0.071**

A-PC3:

- discovery: **+0.136**
- 2009–18: **+0.110**
- 2021–23: **+0.130**

Therefore the earlier failure of scalar Hb-deficit A in 2021–23 may be consistent with a redistribution of phenotype-relevant hematological information across dimensions.

The multidimensional hematological representation remains intact even when the Hb-deficit-associated dimension weakens.

This is a computational interpretation of the observed pattern and is **not** yet a causal biological explanation.

---

# Current Scientific Position

The first physiological testbed now supports the following chain:

`raw biomarkers -> outcome-independent decomposition -> reconstruction -> separability -> phenotype mapping -> scalar information-loss test -> frozen temporal transfer`

Current evidence supports:

1. Hematological state is multidimensional and structurally stable.
2. Glycemic state becomes clearly multidimensional once insulin and OGTT are included.
3. Final reduced A components contain somatic-phenotype information that scalar anemia does not preserve.
4. G3 independently contributes to somatic phenotype when modeled alongside A.
5. Much of the transferable G3 signal remains concentrated in the common glycemic axis.
6. The combined A + G3 representation contains more somatic-phenotype information than scalar A + G.
7. The combined decomposition advantage formally replicates in 2009–18.
8. The underlying representations remain structurally valid through 2021–23.
9. The 2021–23 phenotype result remains suggestive but design-limited for formal complex-survey inference.
10. Scalar representation failure can occur even when multidimensional physiological information remains preserved.

---

## Important Boundaries

Do **not** currently claim:

- causality
- clinical prediction or diagnostic utility
- final biological mechanisms for the PCs
- biological independence of A and G
- a stable A×G interaction
- that G3 fully captures diabetes physiology
- confirmatory 2021–23 X3 significance
- that the scalar-A failure has a known biological cause

---

## Stop Point

Work stops here for today.

Do not add further exploratory models before reviewing the current evidence as a whole.

### Next time

1. Build a concise discovery -> 2009–18 replication -> 2021–23 holdout synthesis.
2. Decide whether one targeted heterogeneity/sensitivity analysis is still required.
3. Freeze the resulting evidence pack.
4. Begin structuring Paper 1 around:
   - representation choice,
   - information preservation,
   - scalar information loss,
   - physiological separability,
   - temporal transfer.

---

## Status at End of Day

The main decomposition result now survives:

`discovery -> frozen representation -> temporal replication`

The strongest result is no longer merely that A and G associate with depression.

It is that a stable multidimensional physiological representation preserves phenotype-relevant information that conventional scalar burden measures can lose, and that this advantage transfers forward in time.

**Valid stopping point.**