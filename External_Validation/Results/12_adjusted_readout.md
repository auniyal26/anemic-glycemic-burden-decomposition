# Script 12 — MIDJA observed-person challenge of NHANES longitudinal simulations

## Matched longitudinal design

- HbA1c panel: adjusted; 211 strictly chronological MIDJA1→MIDJA2 complete participants.
- Median follow-up: 53.0 months (range 44.0–65.0).
- Same people, two genuine measurement times; this is not independent-person replication.
- Four markers: Hb, RBC, MCV, HbA1c. No glucose, insulin or RDW; NHANES seven-marker loadings cannot be reused.
- All physiological PCA transforms learned within proper-training participants only and applied unchanged to their calibration and held-out participants.
- 10 repeated 5-fold person-held-out splits with a 25% calibration partition within each training fold.
- Current CES-D and its change are measured at follow-up. Models using them are *same-time physiological inference*, not forecasts made at baseline.

## Empirical test of Script 84 — recover current physiology

| Model | Standardized mean PC MSE | Gain vs previous Z only | Nominal 90% coverage | Mean half-width |
|---|---:|---:|---:|---:|
| age_sex | 1.4159 | -0.5857 | 0.912 | 1.986 |
| CESD2_only | 1.4127 | -0.5825 | 0.913 | 1.991 |
| Z1_only | 0.8302 | +0.0000 | 0.908 | 1.419 |
| Z1_plus_CESD2 | 0.8369 | -0.0066 | 0.909 | 1.426 |
| Z1_plus_CESD1 | 0.8267 | +0.0035 | 0.907 | 1.417 |
| Z1_plus_CESD1_plus_deltaCESD | 0.8308 | -0.0005 | 0.907 | 1.422 |
| identity_persistence | 1.1379 | -0.3076 | 0.912 | 1.700 |

## Empirical test of Script 85 — recover physiological change

| Model | Standardized mean ΔPC MSE | Gain vs previous Z only | Nominal 90% coverage | Mean half-width |
|---|---:|---:|---:|---:|
| age_sex | 0.9787 | -0.1468 | 0.911 | 1.597 |
| deltaCESD_only | 0.9629 | -0.1309 | 0.911 | 1.595 |
| Z1_only | 0.8319 | +0.0000 | 0.912 | 1.423 |
| Z1_plus_deltaCESD | 0.8296 | +0.0023 | 0.910 | 1.424 |
| Z1_plus_CESD1 | 0.8280 | +0.0040 | 0.911 | 1.418 |
| Z1_plus_CESD1_plus_deltaCESD | 0.8317 | +0.0002 | 0.909 | 1.424 |

## Paired uncertainty (first CV repeat)

Positive MSE gain favors the candidate. Participant bootstrap intervals describe fixed held-out squared-error differences; they do not refit models.

| Scenario | Reference → candidate | ΔMSE | Descriptive 95% interval |
|---|---|---:|---|
| 84 | Z1_only → CESD2_only | -0.5579 | [-0.7545, -0.3885] |
| 84 | Z1_only → Z1_plus_CESD2 | -0.0038 | [-0.0086, +0.0005] |
| 84 | Z1_plus_CESD1 → Z1_plus_CESD1_plus_deltaCESD | -0.0027 | [-0.0099, +0.0040] |
| 84 | identity_persistence → Z1_only | +0.3184 | [+0.1969, +0.4358] |
| 85 | Z1_only → deltaCESD_only | -0.1520 | [-0.2434, -0.0586] |
| 85 | Z1_only → Z1_plus_deltaCESD | +0.0024 | [-0.0109, +0.0160] |
| 85 | Z1_plus_CESD1 → Z1_plus_CESD1_plus_deltaCESD | -0.0031 | [-0.0104, +0.0042] |

## Physiological transition versus NHANES Script 84 simulation

These five rho values are the original simulation grid, **not estimates derived from NHANES follow-up**. Synthetic transitions assume no population drift and are generated in the MIDJA baseline component space.

| Component | Observed lag correlation | Observed ΔZ mean | Observed ΔZ SD | Best grid rho for lag correlation |
|---|---:|---:|---:|---:|
| PC1 | +0.793 | -0.042 | 0.638 | 0.75 |
| PC2 | +0.615 | -0.235 | 0.916 | 0.50 |
| PC3 | +0.821 | +0.669 | 0.878 | 0.75 |
| PC4 | +0.299 | -0.361 | 1.286 | 0.25 |

## Limits

- Similar persistence does not imply identical physiological mechanisms or validate a 15-year simulator.
- With two visits per participant, the four Script 53 synthetic worlds (stable, mapping drift, A×G interaction, measurement drift) cannot be individually identified or distinguished reliably.
- Only the Script 84–85 **hypotheses and rho grid** are challenged here, not a replay of the original seven-feature NHANES simulation or its fitted phenotype mapping.
- Underlying Script 84–85 NHANES simulation outcomes differ (PHQ somatic/COGAFF vs MIDJA CES-D); do not compare their raw error magnitudes.
- Selection/attrition, lab comparability, RBC units, CES-D scoring, survey weights and within-person confounding remain unverified.
- Predicted uncertainty is split-calibrated within each outer training fold; exchangeability and population coverage are not guaranteed under selection/shift.
- No causal conclusion follows from predictive gains or empirical lag correlations.
