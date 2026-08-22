# Full methodological audit — current status

## Bottom line

The project is methodologically coherent as an **exploratory proof-of-concept**, but it is not yet publication-grade or a proof of a causal theory.

### What passes now
- Raw NHANES XPT files remain available as ground truth.
- Working copies and reference Parquet files preserve participant-level linkage via `SEQN`.
- The domain-informed anemia/glycemia transformations are legitimate working representations as long as raw values are retained.
- Independent ICA by NHANES cycle, component alignment, seed stability, and bootstrap sampling stability are mathematically legitimate exploratory tests.
- The current latent PHQ structure resembles established PHQ-9 factor structures, which is external validation of the decomposition rather than novelty by itself.

### What requires correction/confirmation
1. **Complex survey inference:** current WLS + clustered SE analyses are not the full NHANES survey design. Final inference must use weights + strata + PSU together.
2. **Ordinal PHQ items:** ICA on 0–3 items is exploratory. Confirm with ordinal CFA/WLSMV.
3. **Three vs four components:** choosing 3 because they were more stable was data-driven. Existing literature supports a reproducible four-factor PHQ structure. Compare 1-, 2-, 3-, and 4-factor solutions formally rather than declaring K=3 final.
4. **IC3 interpretation:** the old IC3 figure came from the 4-component solution while the final validation used 3 components. Regenerate 3-component loadings before interpretation.
5. **Rare-item ICA behavior:** suicidal ideation is highly sparse/non-Gaussian and may dominate an ICA component. Test whether IC3 adds information beyond DPQ090 alone and beyond the established Internalizing factor (worth/guilt + suicidality).
6. **HbA1c measurement coupling:** anemia/erythrocyte abnormalities can alter HbA1c independently of glucose. Replicate G using fasting glucose and a broader diabetes definition.
7. **Confounders vs mediators:** kidney disease and BMI are not automatically “confounders.” A causal DAG/estimand must decide which variables belong in each adjustment set.
8. **Interaction:** failure of `A×G` in the current linear PHQ scale does not prove no biological interaction. Interaction is scale-dependent. Test additive and multiplicative/risk-scale interaction separately.
9. **Multiple testing:** many models/components were explored. Current p-values are exploratory. Use a discovery/confirmation split, FDR where appropriate, and pre-specify the confirmatory model.
10. **Nonlinear cross-cycle leakage:** the prior spline basis was constructed using pooled data before train/test separation. Refit preprocessing and spline bases on training data only.
11. **Missingness:** current complete-case analysis may induce selection bias. Audit missingness patterns and compare included vs excluded participants.
12. **Pregnancy/anemia definition:** pregnancy must be excluded or handled with pregnancy-specific Hb thresholds. Smoking and other factors that alter Hb need sensitivity analysis.

## Novelty audit

Already established in the literature:
- anemia is associated with depression in NHANES;
- diabetes/glycemia is associated with depression;
- PHQ-9 has reproducible subfactor structure;
- a four-factor PHQ-9 structure has been reported: Affective, Somatic, Internalizing, Sensorimotor;
- diabetes-specific depressive symptom profiles have been studied.

Still plausibly novel after targeted searching:
- jointly modeling hematological and glycemic burden against **specific depressive symptom dimensions**, while preserving raw/reference states and validating the decomposition across independent NHANES cycles and resampling;
- a reproducible anemia association with a specific aligned latent PHQ phenotype, if it survives survey-corrected, confounder-aware, ordinal-factor confirmation and is not reducible to one PHQ item.

Do **not** claim “first ever” until a formal systematic novelty review is completed.

## Prof. Saeed's reference/compression idea

Current implementation is partially correct:
- raw XPT = immutable ground truth;
- Parquet + ZSTD = lossless storage compression of tabular working/reference data;
- transformed variables and ICA spaces are lossy representations;
- original raw values are retained.

What must be added:
- SHA256 hashes of every raw file;
- transformation ledger;
- exact transformation parameters;
- scaler means/SDs;
- ICA mixing/unmixing matrices;
- component count and random seed;
- script hash/version;
- explicit mapping from `SEQN` and cycle to each derived representation.

The raw XPT bytes, not the Parquet conversion, should remain the authoritative source.

## Recommended order after audit

1. Run the integrity audit script.
2. Fix final 3-component/4-component provenance and test IC3 single-item dominance.
3. Build a causal DAG and define total-effect vs direct-effect adjustment sets.
4. Add BMI, kidney function, smoking, SES/education without blindly calling them all confounders.
5. Redo main inference with proper NHANES survey design.
6. Replicate glycemia using fasting glucose and diabetes status/medications.
7. Confirm PHQ structure with ordinal CFA/WLSMV.
8. Only then freeze the preliminary result and build the proposal presentation.
