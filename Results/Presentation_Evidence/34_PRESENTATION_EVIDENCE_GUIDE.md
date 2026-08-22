# Presentation Evidence Guide — Script 34

## 1. Scope lock

This evidence pack supports a **preliminary feasibility / proposal presentation**.

The defensible question is:

> Can hematological and glycemic physiological information be represented and statistically separated with respect to depressive symptom dimensions, and which parts remain stable under temporal transfer?

Do **not** present this as:
- causal proof that anemia or dysglycemia causes depression;
- a completed biological mechanism;
- a finished decomposition/reconstruction framework;
- proof that the model generalizes to all populations;
- proof that COVID caused the 2021–2023 change.

---

## 2. People and data flow

The script rebuilt the discovery core directly from the raw 2005–06 and 2007–08 XPT files.

| stage                               |     n |   retained_vs_previous_pct |   retained_vs_first_pct |
|:------------------------------------|------:|---------------------------:|------------------------:|
| Same-person raw core match, age 18+ | 11329 |                   100      |                100      |
| Complete hemoglobin                 | 10640 |                    93.9183 |                 93.9183 |
| Complete HbA1c                      | 10632 |                    99.9248 |                 93.8476 |
| Complete PHQ-9                      | 10214 |                    96.0685 |                 90.158  |
| Complete A / G / PHQ                |  9743 |                    95.3887 |                 86.0005 |
| Final survey X3 analysis cohort     |  7957 |                    81.6689 |                 70.2357 |

The important distinction is:

- **9,743** (or the value above if your raw files reproduce differently) = complete first-pass A/G/PHQ discovery core.
- **X3 survey cohort** = stricter complete-case cohort used for the fully adjusted complex-survey models.
- Fasting, clinical-state, CFA/MIMIC and internal-PCA analyses have their own required-variable subsets.

Never present all analyses as if they used an identical N.

---

## 3. What x and y actually were

### Preserved x

For participant i, the preserved source state includes the measured inputs:

`x_i = [Hb, CBC markers, HbA1c, PHQ item responses, demographics/covariates, ...]`

The raw XPT files remain authoritative.

### First operational transformation

`A = max(Hb_threshold(sex) - Hb, 0)`

- male threshold = 13 g/dL
- female threshold = 12 g/dL

`G = max(HbA1c - 5.7, 0)`

`AG = A * G`

The first operational working representation can therefore be thought of as:

`y_i = [A, G, AG, preserved PHQ item state / derived phenotype, X]`

This is **not yet the final mathematical y** promised by the thesis.

The current y summary is in `Tables/34_operational_y_summary.csv`.

### Why the domain transformation mattered

The earlier naive `A = -z(Hb)` representation did not carry a useful A association.
The clinically informed deficit representation did. That is evidence that **representation choice can erase or expose statistical structure**, not proof that the thresholded scalar A is the final biological representation.

---

## 4. Did we repeatedly modify the same y?

### What the code architecture supports

The project audit shows the intended architecture is **reference preserving and branched**:

`raw XPT -> preserved working reference -> fresh analysis branches`

The main downstream scripts reload `nhanes_core_working.parquet` and/or a defined derivative cohort, then build new DataFrames with `.copy()`, merges, residual matrices, standardized matrices, latent-factor inputs, or sensitivity subsets.

Script 34 also performs a static source scan and writes:
- `Tables/34_reference_preservation_source_audit.csv`
- `Tables/34_reference_preservation_summary.csv`

Core hash unchanged during this evidence build: **True**

Downstream explicit core-overwrite scan: **none detected**

### Precise wording for the presentation

> We preserved the raw and core working references. Individual analyses branched from those references or explicitly defined derivative cohorts; they were not intended as a destructive serial chain in which each experiment overwrote the previous experiment's representation.

Do **not** say “every script rebuilt from raw.” Many correctly reload a preserved processed reference instead.

### Remaining reconstruction gap

We have **not** yet completed the final:

`final y -> decomposed components -> y_hat`

with a final prespecified reconstruction loss `L_y`.

Earlier exploratory algebraic/PCA/ICA reconstruction work exists, but the thesis-level final reconstruction test is still open.

That is a future-work box, not a result box.

---

## 5. Primary statistical result

### Discovery: 2005–2008, strict additive somatic model

A:
- beta = 0.3315
- 95% CI = [0.1374, 0.5257]
- p = 0.00242

G:
- beta = 0.0974
- 95% CI = [-0.0068, 0.2016]
- p = 0.06476

Interpretation:

- A shows clear adjusted association in the strict additive pooled survey model.
- G is positive but is weaker/borderline in this exact strict additive discovery specification.
- G is supported by other prespecified/convergent analyses, especially fasting glucose and latent MIMIC.
- A×G is secondary because it is small and representation/replication sensitive.

---

## 6. What beta means in plain language

For the strict additive model:

`Somatic = beta0 + beta_A*A + beta_G*G + beta_X*X + error`

### A beta

A is measured in **g/dL below the sex-specific Hb threshold**.

So, if beta_A were 0.33:

> among otherwise model-comparable participants, a 1 g/dL larger Hb deficit is associated with about 0.33 higher points on the 0–9 somatic score, conditional on G and the modeled X variables.

### G beta

G is measured in **HbA1c percentage points above 5.7**.

So, if beta_G were 0.10:

> a 1 percentage-point larger HbA1c excess is associated with about 0.10 higher somatic-score points, conditional on A and X.

These are **conditional associations**, not individual-level causal predictions.

---

## 7. Why the survey machinery was necessary

NHANES is a complex probability sample.

Therefore the main regressions used:
- examination weights;
- sampling strata;
- primary sampling units (PSUs);
- cycle-aware pooling.

Raw N is not the same thing as inferential degrees of freedom.

This becomes crucial in 2021–2023: the modern holdout still has thousands of participants, but the fully adjusted X3 model consumes the residual survey-design degrees of freedom. That is why the X3 beta can be estimated while its usual t-based CI/p becomes non-estimable.

See:
- `Figures/34_25_2123_residual_design_df.png`
- `Tables/34_2123_design_degrees_of_freedom.csv`

---

## 8. Why X0 -> X3 instead of one giant adjustment set

The ladder was designed to make assumptions visible:

- **X0:** age, sex, race
- **X1:** + socioeconomic status, education, smoking
- **X2:** + BMI
- **X3:** + kidney function

BMI and kidney function can sit on or near plausible physiological pathways, so forcing them into the only model would hide an important modeling assumption.

The staged ladder shows whether A/G are robust or whether an association disappears only after adding a pathway-sensitive variable.

---

## 9. PHQ phenotype: why CFA / MIMIC

A PHQ-9 total score collapses multiple symptoms.

We therefore tested whether the item structure supports:
- a **somatic** factor: sleep, fatigue, appetite
- a **cognitive-affective** factor: remaining affective/cognitive items

Ordinal WLSMV CFA was used because PHQ items are ordered categories rather than continuous Gaussian measurements.

MIMIC then asks whether A/G relate to those validated latent factors.

Current weighted MIMIC fit:
- CFI = 0.996
- RMSEA = 0.011
- SRMR = 0.039

Use fit indices as model diagnostics, not as proof that the factor model is “true.”

---

## 10. Why fasting glucose mattered

Anemia / altered RBC turnover can distort measured HbA1c.

Therefore the glycemia result had to survive an alternative measurement that does not rely on the same erythrocyte mechanism.

The fasting-glucose sensitivity is shown in:
`Figures/34_19_fasting_glucose_sensitivity.png`

If the fasting G coefficient remains positive with a valid interval excluding zero, it reduces the likelihood that the entire glycemia association is merely an HbA1c/RBC-turnover artifact.

It does not eliminate all measurement bias or confounding.

---

## 11. Why internal A/G decomposition mattered

The scalar A and G were always intended as first-pass representations.

Reduced hematology PCA asks:

> Is the CBC state really describable by one scalar deficit, or is stable multivariate information being discarded?

The reduced CBC basis uses:
- Hb
- RBC count
- MCV
- RDW

This avoids the most obvious algebraic redundancy in the larger CBC set.

The PCA axes are **stable measurement axes**, not yet named biological mechanisms.

For G, the current PCA uses only HbA1c and fasting glucose. If PC1 captures about 91% of their standardized variance, the correct claim is:

> the two-marker glycemic measurement space is dominated by one common axis.

Not:

> glycemic physiology is one-dimensional.

---

## 12. Temporal validation: what happened

2009–2018 frozen X3:

A:
- beta = 0.1579
- CI = [0.0674, 0.2483]
- p = 0.00103

G:
- beta = 0.1066
- CI = [0.0553, 0.1578]
- p = 0.00013

This is strong temporal replication of the simple A/G association pattern.

2021–2023 diagnostic X0:

A:
- beta = 0.0546
- p = 0.73013

G:
- beta = 0.2004
- p = 0.00615

A is already near zero at the simplest valid adjustment stage, while G remains positive.

Therefore the modern A failure cannot be dismissed as *only* an X3 degree-of-freedom problem.

Correct statement:

> The simple Hb-deficit A representation transferred through 2009–2018 but did not reproduce in the 2021–2023 holdout; G was more temporally stable. The reason for the A non-transfer is not yet known.

---

## 13. What the important numbers do and do not prove

### Strong enough to show

1. The discovery dataset is large enough for a serious feasibility analysis.
2. A has a clear adjusted somatic association in pooled discovery.
3. G contributes smaller/convergent information; the strict discovery additive p-value alone is borderline, but fasting and latent analyses support the direction.
4. The somatic PHQ structure is reproducible using ordinal CFA.
5. A/G are not highly redundant proxies in the discovery model.
6. 2009–2018 validates both simple A and G prospectively in time, under frozen definitions.
7. 2021–2023 exposes a real boundary: simple A does not transfer, while G remains more stable.
8. The physiological measurement space contains structure beyond the first scalar proxies.

### Not strong enough to claim

1. Causality.
2. Clinical diagnostic utility.
3. A final biological mechanism.
4. Stable A×G interaction.
5. A final A_i/G_j decomposition.
6. Final y reconstruction.
7. India transfer.
8. COVID as explanation for 2021–2023.

---

## 14. Recommended presentation figure sequence

Use the pack as a menu; do not put every figure in the main deck.

**Main deck**
1. `34_11_x_to_y_operational_pipeline.png`
2. `34_01_discovery_cohort_flow.png`
3. `34_08_hb_to_A_transform.png` + `34_09_hba1c_to_G_transform.png` (can be placed side-by-side later)
4. `34_13_ag_group_mean_phq9.png`
5. `34_14_discovery_primary_effects.png`
6. `34_16_cfa_model_comparison.png`
7. `34_17_somatic_factor_loadings.png`
8. `34_18_mimic_somatic_paths.png`
9. `34_19_fasting_glucose_sensitivity.png`
10. `34_20_reduced_A_pca_variance.png`
11. `34_22_temporal_transfer_frozen_X3.png`
12. `34_23_2123_A_xladder.png`
13. `34_30_reference_preserving_branch_architecture.png`
14. final scope / next-step diagram later in the slide-building stage.

**Appendix**
- detailed distributions;
- G PCA;
- 2021–23 design df;
- temporal raw distribution comparisons;
- reference-preservation source audit;
- full statistic glossary.

---

## 15. One-sentence presentation claim

> In NHANES, hematological and glycemic burden show separable but unequal associations with a reproducible somatic depressive phenotype; the pattern transfers through 2009–2018, while the simple hematological representation fails in the 2021–2023 holdout, motivating a more principled and information-preserving physiological decomposition.

That is a proposal-level result. It is neither “we solved depression” nor “we found nothing.”
