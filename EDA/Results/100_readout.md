# Script 100 — Dynamic physiological regression: readout

## Experimental rule

- 2005–08: training and cycle-stratified 5-fold CV; entire gate/expert fitted inside training folds.
- 2009–18 and 2021–23: frozen predictions only; no tuning or refitting.
- **Historical warning:** both later periods have been repeatedly examined in Scripts 72–99, so results are exploratory replication rather than untouched confirmatory tests.
- NHANES cycles are different participants, not individual longitudinal trajectories.

## Selected models and transfer performance

| target    | role         | label                                         |   CV_MSE | period             |       R2 |    RMSE |     MAE |     bias |
|:----------|:-------------|:----------------------------------------------|---------:|:-------------------|---------:|--------:|--------:|---------:|
| PHQ_total | cv_winner    | full/smooth_experts/G_axis/alpha=100/tau=0.75 | 12.33095 | test_2009-2018     |  0.07947 | 3.87752 | 2.73296 | -0.05524 |
| PHQ_total | cv_winner    | full/smooth_experts/G_axis/alpha=100/tau=0.75 | 12.33095 | validate_2021-2023 | -0.00697 | 4.43138 | 3.06750 | -0.94853 |
| PHQ_total | best_dynamic | full/smooth_experts/G_axis/alpha=100/tau=0.75 | 12.33095 | test_2009-2018     |  0.07947 | 3.87752 | 2.73296 | -0.05524 |
| PHQ_total | best_dynamic | full/smooth_experts/G_axis/alpha=100/tau=0.75 | 12.33095 | validate_2021-2023 | -0.00697 | 4.43138 | 3.06750 | -0.94853 |
| PHQ_total | best_global  | full/global_z_pairwise/none/alpha=0.05/l1=0.8 | 12.33502 | test_2009-2018     |  0.07538 | 3.88613 | 2.74054 | -0.05709 |
| PHQ_total | best_global  | full/global_z_pairwise/none/alpha=0.05/l1=0.8 | 12.33502 | validate_2021-2023 |  0.00077 | 4.41431 | 3.08075 | -0.90269 |
| Somatic   | cv_winner    | full/smooth_experts/G_axis/alpha=100/tau=0.75 |  3.41604 | test_2009-2018     |  0.07813 | 1.93830 | 1.49645 |  0.01803 |
| Somatic   | cv_winner    | full/smooth_experts/G_axis/alpha=100/tau=0.75 |  3.41604 | validate_2021-2023 |  0.03450 | 2.04910 | 1.55044 | -0.30605 |
| Somatic   | best_dynamic | full/smooth_experts/G_axis/alpha=100/tau=0.75 |  3.41604 | test_2009-2018     |  0.07813 | 1.93830 | 1.49645 |  0.01803 |
| Somatic   | best_dynamic | full/smooth_experts/G_axis/alpha=100/tau=0.75 |  3.41604 | validate_2021-2023 |  0.03450 | 2.04910 | 1.55044 | -0.30605 |
| Somatic   | best_global  | full/global_z_pairwise/none/alpha=0.05/l1=0.5 |  3.41826 | test_2009-2018     |  0.07360 | 1.94305 | 1.50269 |  0.01660 |
| Somatic   | best_global  | full/global_z_pairwise/none/alpha=0.05/l1=0.5 |  3.41826 | validate_2021-2023 |  0.03844 | 2.04491 | 1.55891 | -0.28442 |
| CogAff    | cv_winner    | full/smooth_experts/G_axis/alpha=100/tau=1.5  |  4.44826 | test_2009-2018     |  0.05815 | 2.38023 | 1.55025 | -0.07301 |
| CogAff    | cv_winner    | full/smooth_experts/G_axis/alpha=100/tau=1.5  |  4.44826 | validate_2021-2023 | -0.03545 | 2.81634 | 1.83247 | -0.64197 |
| CogAff    | best_dynamic | full/smooth_experts/G_axis/alpha=100/tau=1.5  |  4.44826 | test_2009-2018     |  0.05815 | 2.38023 | 1.55025 | -0.07301 |
| CogAff    | best_dynamic | full/smooth_experts/G_axis/alpha=100/tau=1.5  |  4.44826 | validate_2021-2023 | -0.03545 | 2.81634 | 1.83247 | -0.64197 |
| CogAff    | best_global  | full/global_z_ridge/none/alpha=100            |  4.45128 | test_2009-2018     |  0.05922 | 2.37888 | 1.55020 | -0.07111 |
| CogAff    | best_global  | full/global_z_ridge/none/alpha=100            |  4.45128 | validate_2021-2023 | -0.03518 | 2.81596 | 1.83227 | -0.64037 |

## PHQ decomposition coherence

| period             |    n |   sum_somatic_cogaff_R2 |   direct_total_R2 |
|:-------------------|-----:|------------------------:|------------------:|
| train_2005-2008    | 3815 |                 0.10658 |           0.10713 |
| test_2009-2018     | 9893 |                 0.07978 |           0.07947 |
| validate_2021-2023 | 2247 |                -0.00649 |          -0.00697 |

## Interpretation

The model has a fitted numerical equation for every expert; gates choose a smooth convex mixture. A static model is valid if it beats dynamic alternatives in TRAIN CV.
Physiological block contrasts are model sensitivity measures, **not** causal/clinical attribution or additive percentage shares.
Global interaction models and smooth gating may overfit; low transfer R² does not invalidate the mathematical equation.
All metrics use sample weights only, not complex NHANES survey-design uncertainty.

## Next scientific decision

Evaluate whether gates beat the best global model on **training CV** first; inspect later-cycle degradation, calibration, and gate occupancy without using those results to select a new winner.
If dynamic models fail, retain the fitted global 7-Z Ridge equation and report the negative result.
