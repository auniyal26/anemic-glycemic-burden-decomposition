# Script 12 — MIDJA observed-person challenge of NHANES longitudinal simulations

## Matched longitudinal design

- HbA1c panel: raw; 211 strictly chronological MIDJA1→MIDJA2 complete participants.
- Median follow-up: 53.0 months (range 44.0–65.0).
- Same people, two genuine measurement times; this is not independent-person replication.
- Four markers: Hb, RBC, MCV, HbA1c. No glucose, insulin or RDW; NHANES seven-marker loadings cannot be reused.
- All physiological PCA transforms learned within proper-training participants only and applied unchanged to their calibration and held-out participants.
- 10 repeated 5-fold person-held-out splits with a 25% calibration partition within each training fold.
- Current CES-D and its change are measured at follow-up. Models using them are *same-time physiological inference*, not forecasts made at baseline.

## Empirical test of Script 84 — recover current physiology

| Model | Standardized mean PC MSE | Gain vs previous Z only | Nominal 90% coverage | Mean half-width |
|---|---:|---:|---:|---:|
| age_sex | 1.0629 | -0.3853 | 0.910 | 1.761 |
| CESD2_only | 1.0642 | -0.3866 | 0.912 | 1.765 |
| Z1_only | 0.6776 | +0.0000 | 0.907 | 1.286 |
| Z1_plus_CESD2 | 0.6836 | -0.0060 | 0.907 | 1.290 |
| Z1_plus_CESD1 | 0.6730 | +0.0046 | 0.908 | 1.282 |
| Z1_plus_CESD1_plus_deltaCESD | 0.6765 | +0.0011 | 0.908 | 1.286 |
| identity_persistence | 0.9454 | -0.2678 | 0.909 | 1.520 |

## Empirical test of Script 85 — recover physiological change

| Model | Standardized mean ΔPC MSE | Gain vs previous Z only | Nominal 90% coverage | Mean half-width |
|---|---:|---:|---:|---:|
| age_sex | 0.8380 | -0.1584 | 0.908 | 1.452 |
| deltaCESD_only | 0.8236 | -0.1441 | 0.909 | 1.441 |
| Z1_only | 0.6796 | +0.0000 | 0.912 | 1.287 |
| Z1_plus_deltaCESD | 0.6764 | +0.0032 | 0.911 | 1.285 |
| Z1_plus_CESD1 | 0.6746 | +0.0050 | 0.911 | 1.280 |
| Z1_plus_CESD1_plus_deltaCESD | 0.6777 | +0.0018 | 0.910 | 1.283 |

## Paired uncertainty (first CV repeat)

Positive MSE gain favors the candidate. Participant bootstrap intervals describe fixed held-out squared-error differences; they do not refit models.

| Scenario | Reference → candidate | ΔMSE | Descriptive 95% interval |
|---|---|---:|---|
| 84 | Z1_only → CESD2_only | -0.3665 | [-0.4730, -0.2655] |
| 84 | Z1_only → Z1_plus_CESD2 | -0.0030 | [-0.0070, +0.0008] |
| 84 | Z1_plus_CESD1 → Z1_plus_CESD1_plus_deltaCESD | -0.0020 | [-0.0095, +0.0046] |
| 84 | identity_persistence → Z1_only | +0.2750 | [+0.1883, +0.3664] |
| 85 | Z1_only → deltaCESD_only | -0.1515 | [-0.2236, -0.0809] |
| 85 | Z1_only → Z1_plus_deltaCESD | +0.0028 | [-0.0104, +0.0163] |
| 85 | Z1_plus_CESD1 → Z1_plus_CESD1_plus_deltaCESD | -0.0024 | [-0.0094, +0.0051] |

## Physiological transition versus NHANES Script 84 simulation

These five rho values are the original simulation grid, **not estimates derived from NHANES follow-up**. Synthetic transitions assume no population drift and are generated in the MIDJA baseline component space.

| Component | Observed lag correlation | Observed ΔZ mean | Observed ΔZ SD | Best grid rho for lag correlation |
|---|---:|---:|---:|---:|
| PC1 | +0.793 | -0.052 | 0.638 | 0.75 |
| PC2 | +0.628 | -0.222 | 0.882 | 0.75 |
| PC3 | +0.833 | +0.532 | 0.571 | 0.90 |
| PC4 | +0.305 | -0.342 | 1.271 | 0.25 |

## Limits

- Similar persistence does not imply identical physiological mechanisms or validate a 15-year simulator.
- With two visits per participant, the four Script 53 synthetic worlds (stable, mapping drift, A×G interaction, measurement drift) cannot be individually identified or distinguished reliably.
- Only the Script 84–85 **hypotheses and rho grid** are challenged here, not a replay of the original seven-feature NHANES simulation or its fitted phenotype mapping.
- Underlying Script 84–85 NHANES simulation outcomes differ (PHQ somatic/COGAFF vs MIDJA CES-D); do not compare their raw error magnitudes.
- Selection/attrition, lab comparability, RBC units, CES-D scoring, survey weights and within-person confounding remain unverified.
- Predicted uncertainty is split-calibrated within each outer training fold; exchangeability and population coverage are not guaranteed under selection/shift.
- No causal conclusion follows from predictive gains or empirical lag correlations.
