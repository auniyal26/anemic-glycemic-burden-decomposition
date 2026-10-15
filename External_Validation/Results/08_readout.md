# Script 08 — Causal-identification stress test: hematology and HbA1c

**Identification verdict: NOT ESTABLISHED.** AIPW is a causal *estimation method* only when its causal assumptions hold; these cross-sectional data cannot verify them.

## Explicit causal question

Would changing the red-cell/iron-deficiency process, without changing average glucose exposure, change the measured HbA1c? **Not directly identifiable here**. Low hemoglobin is a composite clinical state with multiple etiologies, not a specific treatment.

## What the data can estimate

Cross-fitted, survey-weighted AIPW contrasts of sex-threshold low Hb vs other Hb in measured HbA1c; adjustment includes fasting glucose/log-insulin, age, sex, BMI, eGFR and optionally recorded socioeconomic/behavioral fields. Matching uses the same complete cohort within each period. All models are fitted separately by period.

| NHANES period | Specification | N | Low-Hb N | Adjusted contrast | Propensity clipped N |
|---|---|---:|---:|---:|---:|
| discovery_2005_08 | glucose_adjusted_basic | 4099 | 304 | +0.2592 | 2418 |
| discovery_2005_08 | glucose_proxy_specificity | 4099 | 304 | -0.7798 | 2393 |
| discovery_2005_08 | glucose_adjusted_social | 4099 | 304 | +0.1881 | 2236 |
| holdout_2009_18 | glucose_adjusted_basic | 10888 | 1030 | +0.1474 | 4923 |
| holdout_2009_18 | glucose_proxy_specificity | 10888 | 1030 | -2.1947 | 4741 |
| holdout_2009_18 | glucose_adjusted_social | 10888 | 1030 | +0.1250 | 4489 |
| holdout_2021_23 | glucose_adjusted_basic | 2575 | 239 | +0.1007 | 672 |
| holdout_2021_23 | glucose_proxy_specificity | 2575 | 239 | +8.0482 | 482 |
| holdout_2021_23 | glucose_adjusted_social | 2575 | 239 | -0.0156 | 870 |

Contrasts are in HbA1c **percentage points** except the `glucose_proxy_specificity` outcome, in glucose **mg/dL**. A null glucose contrast is suggestive of specificity at most: glucose is NOT a validated negative-control outcome.

## Causal assumptions and threats

1. **Well-defined intervention / consistency:** low Hb can reflect iron deficiency, renal disease, inflammation, blood loss, hemoglobin variants, etc.; changing Hb alone is NOT defined by this comparison.
2. **Conditional exchangeability:** unavailable treatment history, iron/ferritin status, RBC lifespan, hemoglobin variants, medication, and long-term glycemia create plausible unmeasured confounding.
3. **Temporal ordering:** Hb, HbA1c, fasting glucose and insulin are measured concurrently; baseline covariates cannot be proven pre-exposure. Conditioning on glucose/insulin could also condition on mediators or colliders.
4. **Reference measurement:** a single fasting glucose cannot establish true integrated multi-month glucose and therefore cannot isolate *HbA1c assay/biology distortion*.
5. **Positivity / overlap:** inspect propensity distributions and clipped counts in the CSV; cross-fitting cannot repair missing support.
6. **Sample/design:** the source is an analytical extract; weights were used for point summaries, but NO complete NHANES PSU/stratum-aware variance estimator was implemented. No publication-grade CI or causal p-value is reported.
7. **Double robustness:** protects against one of the two nuisance models being misspecified *conditional on causal assumptions*; does not remove unmeasured confounding.

## Sensitivity analyses

- The same-row basic vs expanded covariate set tests measured-confounder sensitivity, not identification.
- `08_omitted_confounder_scenarios.csv` applies purely algebraic bias scenarios Δ_prevalence(U) × effect(U) to the contrast. These are NOT formal bounds or inferred strengths of an actual confounder.
- `glucose_proxy_specificity` is a companion association. Fasting glucose is physiologically related to Hb status and does not qualify as a negative control.

## What would establish a stronger causal claim?

An intervention or defensible natural experiment changing iron/red-cell biology with **repeated, appropriately timed HbA1c and CGM or repeated glucose**; account for glucose-affecting effects of treatment, lab assay, drugs and disease progression. A randomized iron-repletion trial among iron-deficient adults (with ethics/clinical supervision) is one concrete route. A rigorous instrument would require a justified exclusion restriction; none is identified here.

**Scientific decision:** Treat these results as an identification and falsification audit, NOT as proof of cause, an HbA1c correction rule, or a diagnostic product.
