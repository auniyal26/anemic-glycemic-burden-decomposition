# Script 128 | Final reviewer-directed validation

All checks complete: False

Completion does not mean HG7 won comparisons. Inspect all effect estimates and unfavourable results.

## C_survey

- complete: False
- design_pass: True
- raw_marker_pass: False
- source63_available: True
- cycles_audited: 8
- errors: 0
- indeterminate_pregnancy_complete7_n: 102
- note: Eligibility replicates Script 47: age >=20, exclude RIDEXPRG==1; code 3 indeterminate counted separately. Full survey design audited, not item-nonresponse reweighting or assay harmonisation. Raw 63 and 73 marker disagreements reported separately.

## A_CBC

- complete: True
- contrasts: 147
- primary_tests: 16
- primary_holm_significant: 2
- conditional_tests: 14
- archive_reproduced: True
- failures: 0

## B_DCCA

- complete: True
- native_available: True
- native_completed: True
- simulation_checks: 600
- failures: 0
- interpretation: Matched identity expected, not a superiority test; native released code unweighted with fixed full block ranks.

## D_impact_gate

- complete: False
- status: review_and_rebuild_required
- automatic_rerun_performed: False
- affected_if_weight_or_eligibility_diff: ['47 source survey frame', '62/63 physiological scaffold', '69 raw bank', '72 frozen representation', '73 projection', '101-127 affected analyses and figures']
- instruction: If the audit fails, inspect the aggregate mismatch tables; resolve source version, eligibility, raw marker or survey-design differences before rebuilding the full dependency chain in a separate run. Do not patch only downstream weights or overwrite existing results. If D-CCA absent rerun with --dcca path/to/dcca.py.
