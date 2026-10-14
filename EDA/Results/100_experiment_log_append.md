## Script 100 — Dynamic physiological regression (14 October 2026)

**Objective.** Compare static seven-coordinate Ridge/ElasticNet phenotype equations with multiple physiology-gated Ridge equations, preserving the frozen X→Z representation and explicit intercept, coefficients and residual.

**Protocol.** Train and select exclusively using 2005–08 NHANES with 5-fold CV. Frozen transfer evaluation on 2009–18 and 2021–23. The later periods have already been inspected in earlier scripts; interpretations are exploratory.

**Model.** $Y=\sum_k w_k(Z)(\beta_{0k}+\beta_k^\top Z+\gamma_k^\top C)+\varepsilon$. Tested A-private, G-private, joint A/G and unsupervised two-center gating, with lean and full covariates.

**Training-selected results.**

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

**Decision.** Prefer the training-CV winner; dynamic complexity is not automatically better. Coefficients, gate parameters and block contrasts saved in `100_dynamic_equations.md`, `100_equation_bundle.json`, and `100_block_contrasts.csv`. This establishes statistical associations, not causal pathways or a physiological inverse.

**Integrity.** Prior models/decomposition unchanged; no validation-based retuning; no new clinical or within-subject longitudinal claims.
