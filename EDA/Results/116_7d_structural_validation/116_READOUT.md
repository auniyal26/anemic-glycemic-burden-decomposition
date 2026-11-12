# Script 116: seven-dimensional structure validation

**Synthetic experiment only.** Fixed 7D budget for HG7, PCA7 and official JIVE7;
AJIVE's actual rank is reported and <7D runs are excluded from strict comparisons.

## Negative control

No-sharing: 1/50 replicates falsely declared any coupling via max-statistic permutation gate.
Observed rate = 0.0200; exact binomial 95% interval [0.0005062279831152354, 0.10646954571150097]; one-sided p vs 0.05 0.9230550247232867.

A non-significant calibration test is NOT proof of nominal error control.

## Evidence boundaries

- A full-rank 7D affine map retains all seven measured coordinates; identical all-score decoding is expected.
- Global-independence max-permutation calibration is NOT a formal sequential test of rank 2 vs rank 1.
- Group tests use train-only synthetic-truth decoders. They do not validate physiology in humans.
- Groupwise comparisons match dimensions but not the underlying semantic objective of each method.
- PCA first two scores are used only as a leading-variance comparator.
- An explicit discordant group is absent in the comparator native structures; do not substitute a labelled proxy.
- H/G group dimensions of JIVE/AJIVE are truncated using native score order, not fitted to test truth.
- Cross-sectional NHANES outcomes and patient-level validation are not evaluated here.
- These Monte Carlo intervals quantify sampling across simulations, not clinical/survey confidence coverage.
- If simulated results favor HG7, they establish only advantage under the stipulated data generator.

## Output inventory

- 116_metrics.csv: all-score and native-group held-out target R² plus group refit stability
- 116_summary.csv: replicate distribution (not CI) of metric values
- 116_paired_group_differences.csv: paired-bootstrap intervals of mean differences
- 116_rank_gate.csv / 116_gate_summary.json: conservative multiplicity-aware null gate
- 116_failures.csv: full tracebacks; no failures silently removed
