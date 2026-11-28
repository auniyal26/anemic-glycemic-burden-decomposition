# Script 129 | Reference reconciliation

## A_reference

- complete: True
- literal_positive_reference_n: 4506
- excluded_by_original_weight_floor: 418
- original_reference_n: 4088
- raw_reference_n: 4088
- same_reference_participant_ids: True
- scoped_source_checks_pass: True
- all_73_memberships_match: True
- reference_score_error: 1.7763568394002505e-15
- all_73_score_error: 1.7763568394002505e-15
- max_rho_delta: 1.0269562977782698e-15
- min_refit_subspace_similarity: 1.0
- known_indeterminate_reference_n: 20
- known_indeterminate_all_complete_n: 94
- interpretation: Tiny positive weights are counted by literal >0 but excluded by original >1e-12 rule; only verified if counts and IDs match. Later 63 marker absence is out of scope for reference fitting.

## B_pregnancy

- complete: True
- holm_significant: {'archived': 21, 'exclude_indeterminate_all': 20, 'exclude_indeterminate_ref': 20, 'raw_refit': 21}
- max_partial_R2_change: {'archived': 0.0, 'exclude_indeterminate_all': 0.08522905953950252, 'exclude_indeterminate_ref': 0.00859653295469509, 'raw_refit': 6.078471059822732e-15}
- changed_significance_count: {'archived': 0, 'exclude_indeterminate_all': 1, 'exclude_indeterminate_ref': 1, 'raw_refit': 0}
- min_strict_subspace_similarity: 0.9992725891736334
- note: Sensitivity, not a prespecified equivalence test. Review all changes before submission.

## C_HGI

- complete: True
- rows: 84
- note: Descriptive sensitivity of frozen HGI comparisons; original 2005-06 reference uses official weights up to constant scaling.

## D_gate

- reference_rebuild_required: False
- status: reference_reconciled_review_sensitivity
- automatic_rebuild: False
- all_other_analyses_reexecuted: False
