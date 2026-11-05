# Script 110: Evidence audit (NOT a new full benchmark)

*Source:* repository Scripts 103, 105, 106 and 109. No subject-level rows were accessed or fitted here. Scripts 105 and 106 have distinct analytic samples; their estimates are **not pooled** with Script 103.

## Same-sample 5D comparisons, Script 103

- 2009–2018: FrozenZ5 R²=0.8129, difference vs PCA5=-0.1636.
- 2021–2023: FrozenZ5 R²=0.8013, difference vs PCA5=-0.1792.
- All six FrozenZ5 vs PCA5 outcome-period MSE differences favour PCA5 in descriptive participant bootstraps.
- JIVE5 and PCA5 are essentially tied at this rank; native JIVE/AJIVE out-of-period scores use **training-only affine adapters**, not native out-of-sample transforms.

## Specificity and uncertainty, supplemental tests

- Script 105: shared/discordant subspaces exceed age/sex-conditioned permutation nulls in tested eras; 2021–2023 glycaemic-isolated component is close to its conditional null and does not pass nominal p<0.05.
- Script 106: rescaled-PSU bootstrap gives approximate survey-aware structural uncertainty. These are **different sample definitions** from the Script 103 matched-rank benchmark.
- Script 109: no consistent improvement over raw biomarkers in screening discrimination and Brier score.

**Result:** Statistical reproducibility is plausible for specific subspaces; reconstruction or predictive superiority is not established. Full unified **survey-design-based, rank-matched, same-sample comparison remains unrun**.
