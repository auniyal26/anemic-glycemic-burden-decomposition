# Script 127 | Reviewer-directed validation

NHANES modules complete: True

External validation is gated; metadata alone is not validation.

## B_ferritin

- archived_118_reproduced: True
- archive_checked: True
- manuscript_rounded_gap_reproduced: True
- interpretation: CDC_chain_e601 is chained point-estimate assay conversion; no direct 2005-to-2023 calibration or coefficient-uncertainty propagation. E170_only is partial bridge sensitivity, not e601 harmonisation.
- complete: True

## A_HGI

- performance: 84
- contrasts: 84
- conditional_tests: 14
- failures: 0
- complete: True

## C_partial_R2

- tests: 44
- Holm_significant: 21
- complete: True
- note: Percentile intervals conditional on frozen 2005-08 map; outcome regressions refitted per replicate. Nonnegative partial R2 has boundary bias: use original design-adjusted Wald/Holm tests for significance, not CI exclusion of zero.

## D_external

- status: audit_only
- files_found: 15
- metadata_rows: 24618
- validated: False
- reason: No verified participant-level mapping; no external validation claimed
