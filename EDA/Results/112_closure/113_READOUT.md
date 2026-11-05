# Script work closure — 5 November 2026

## Decision

The scripts now support specific evidence-based comparisons. They do not require the H/G representation to win. Read each claim against its named endpoint, reference method, dimensional budget and era. All later eras have been inspected previously; the results remain exploratory.

## Corrected AJIVE comparator

The old `reconsider_joint_components=False` path retains all initial common-space vectors in downstream block projections while exposing only the requested common vectors in the reported scores. On these discovery data, the resulting relative projection errors are about 0.719 (H) and 0.635 (G), and one requested H-individual singular value is about 2e-13. This numerically null score is not a valid stable physiological axis.

Production Script112 uses the original package with its default joint-component check enabled and exact dense SVD as the numerical backend. No installed package or original research script was edited. The retained-projector identity, nonzero extracted singular values, score-adapter recovery and repeated-fit subspaces are explicitly checked. Old AJIVE performance and rediscovery comparisons using the faulty setting need replacement; do not interpret their weakness as a limitation of AJIVE in general. Existing RawX/PCA/H/G and primary survey comparisons do not use that setting.

## Dimensional budgets

5D compares the same fixed five-score budget. 7D compares explicitly specified full-budget maps. The 7D AJIVE/JIVE settings are not automatically selected denoised signal ranks and should be labelled full-budget sensitivity analyses. It is inappropriate to claim superiority for a seven-variable invertible map merely because it retains more than a five-score compression.

Future JIVE/AJIVE scores use discovery-trained affine adapters, not each package's native prediction routine. These corrected fits have exact discovery-score recovery. Reconstruction uses a discovery-trained free linear decoder over scores, rather than each package's own joint-plus-individual reconstructed matrix. The original frozen H/G map retains its historical discovery fit; identical matched participants apply to downstream fitting/evaluation, not to a new identical-row unsupervised refit of every representation.

The maximum difference between weighted OLS predictions from the five full-rank coordinate systems is 6.25e-13. This verifies equal linear information. Ridge differences can reflect coordinate scaling and penalty geometry rather than newly discovered information.

### Reconstruction

| method | period | n | dimensions | reconstruction_R2 |
| --- | --- | --- | --- | --- |
| PCA5 | test_2009-2018 | 9893 | 5 | 0.976477 |
| PCA5 | validate_2021-2023 | 2247 | 5 | 0.980477 |
| FrozenZ5 | test_2009-2018 | 9893 | 5 | 0.812894 |
| FrozenZ5 | validate_2021-2023 | 2247 | 5 | 0.80132 |
| JIVE5 | test_2009-2018 | 9893 | 5 | 0.976382 |
| JIVE5 | validate_2021-2023 | 2247 | 5 | 0.980406 |
| AJIVE5_rankchecked | test_2009-2018 | 9893 | 5 | 0.974739 |
| AJIVE5_rankchecked | validate_2021-2023 | 2247 | 5 | 0.979078 |
| RawX7 | test_2009-2018 | 9893 | 7 | 1 |
| RawX7 | validate_2021-2023 | 2247 | 7 | 1 |
| PCA7 | test_2009-2018 | 9893 | 7 | 1 |
| PCA7 | validate_2021-2023 | 2247 | 7 | 1 |
| FrozenZ7 | test_2009-2018 | 9893 | 7 | 1 |
| FrozenZ7 | validate_2021-2023 | 2247 | 7 | 1 |
| JIVE7 | test_2009-2018 | 9893 | 7 | 1 |
| JIVE7 | validate_2021-2023 | 2247 | 7 | 1 |
| AJIVE7_rankchecked | test_2009-2018 | 9893 | 7 | 1 |
| AJIVE7_rankchecked | validate_2021-2023 | 2247 | 7 | 1 |

### Rank-5 phenotype comparisons

Positive gain is lower MSE than PCA5. All outcomes and both later eras are shown.

| outcome | period | method | reference | paired_gain_positive_better | approx_PSU_low | approx_PSU_high | direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PHQ_total | test_2009-2018 | FrozenZ5 | PCA5 | -0.149681 | -0.261249 | -0.0327435 | favours_reference |
| PHQ_total | test_2009-2018 | JIVE5 | PCA5 | 0.000735182 | -0.00096292 | 0.00237559 | inconclusive |
| PHQ_total | test_2009-2018 | AJIVE5_rankchecked | PCA5 | 0.00219759 | -0.00884351 | 0.013018 | inconclusive |
| Somatic | test_2009-2018 | FrozenZ5 | PCA5 | -0.0369476 | -0.0665059 | -0.00512436 | favours_reference |
| Somatic | test_2009-2018 | JIVE5 | PCA5 | 0.000294535 | -0.000140097 | 0.000800866 | inconclusive |
| Somatic | test_2009-2018 | AJIVE5_rankchecked | PCA5 | 0.00111946 | -0.00159882 | 0.00428692 | inconclusive |
| CogAff | test_2009-2018 | FrozenZ5 | PCA5 | -0.0367715 | -0.0706615 | -0.00365899 | favours_reference |
| CogAff | test_2009-2018 | JIVE5 | PCA5 | 9.32524e-05 | -0.00036045 | 0.000521517 | inconclusive |
| CogAff | test_2009-2018 | AJIVE5_rankchecked | PCA5 | -0.000893131 | -0.0101931 | 0.0076084 | inconclusive |
| PHQ_total | validate_2021-2023 | FrozenZ5 | PCA5 | -0.792311 | -1.04525 | -0.497342 | favours_reference |
| PHQ_total | validate_2021-2023 | JIVE5 | PCA5 | -0.000329228 | -0.00443178 | 0.00365258 | inconclusive |
| PHQ_total | validate_2021-2023 | AJIVE5_rankchecked | PCA5 | -0.0218344 | -0.0459511 | 0.00229247 | inconclusive |
| Somatic | validate_2021-2023 | FrozenZ5 | PCA5 | -0.164724 | -0.218973 | -0.101519 | favours_reference |
| Somatic | validate_2021-2023 | JIVE5 | PCA5 | -0.000592284 | -0.00176454 | 0.000452122 | inconclusive |
| Somatic | validate_2021-2023 | AJIVE5_rankchecked | PCA5 | -0.00747214 | -0.0143594 | -0.000662899 | favours_reference |
| CogAff | validate_2021-2023 | FrozenZ5 | PCA5 | -0.221836 | -0.301203 | -0.130995 | favours_reference |
| CogAff | validate_2021-2023 | JIVE5 | PCA5 | 0.000339393 | -0.000720614 | 0.00145736 | inconclusive |
| CogAff | validate_2021-2023 | AJIVE5_rankchecked | PCA5 | -0.0361933 | -0.0524422 | -0.0189576 | favours_reference |

### Rank-7 phenotype comparisons

Positive gain is lower MSE than RawX7. These are regularized-pipeline comparisons with equivalent input information.

| outcome | period | method | reference | paired_gain_positive_better | approx_PSU_low | approx_PSU_high | direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PHQ_total | test_2009-2018 | PCA7 | RawX7 | -0.000953615 | -0.00864268 | 0.00707976 | inconclusive |
| PHQ_total | test_2009-2018 | FrozenZ7 | RawX7 | -0.000723761 | -0.00842811 | 0.00716991 | inconclusive |
| PHQ_total | test_2009-2018 | JIVE7 | RawX7 | -0.000668783 | -0.00846488 | 0.00746558 | inconclusive |
| PHQ_total | test_2009-2018 | AJIVE7_rankchecked | RawX7 | 0.0306037 | -0.0127903 | 0.0692751 | inconclusive |
| Somatic | test_2009-2018 | PCA7 | RawX7 | 0.000410817 | -0.00167358 | 0.00259372 | inconclusive |
| Somatic | test_2009-2018 | FrozenZ7 | RawX7 | 0.000477933 | -0.00165463 | 0.00270894 | inconclusive |
| Somatic | test_2009-2018 | JIVE7 | RawX7 | 0.000508283 | -0.00161187 | 0.00273721 | inconclusive |
| Somatic | test_2009-2018 | AJIVE7_rankchecked | RawX7 | 0.0102071 | 0.000199833 | 0.0204148 | favours_method |
| CogAff | test_2009-2018 | PCA7 | RawX7 | -0.000706115 | -0.00339935 | 0.00195823 | inconclusive |
| CogAff | test_2009-2018 | FrozenZ7 | RawX7 | -0.000655319 | -0.00326853 | 0.00193331 | inconclusive |
| CogAff | test_2009-2018 | JIVE7 | RawX7 | -0.000659591 | -0.00336671 | 0.00200499 | inconclusive |
| CogAff | test_2009-2018 | AJIVE7_rankchecked | RawX7 | 0.00433245 | -0.0111523 | 0.0189199 | inconclusive |
| PHQ_total | validate_2021-2023 | PCA7 | RawX7 | -0.00624958 | -0.0261764 | 0.0135698 | inconclusive |
| PHQ_total | validate_2021-2023 | FrozenZ7 | RawX7 | -0.00581646 | -0.0254511 | 0.0138345 | inconclusive |
| PHQ_total | validate_2021-2023 | JIVE7 | RawX7 | -0.00683455 | -0.0272049 | 0.0130623 | inconclusive |
| PHQ_total | validate_2021-2023 | AJIVE7_rankchecked | RawX7 | 0.0775613 | -0.0170556 | 0.175356 | inconclusive |
| Somatic | validate_2021-2023 | PCA7 | RawX7 | -0.00031097 | -0.00526139 | 0.00454372 | inconclusive |
| Somatic | validate_2021-2023 | FrozenZ7 | RawX7 | -0.000234044 | -0.00528234 | 0.00470608 | inconclusive |
| Somatic | validate_2021-2023 | JIVE7 | RawX7 | -0.000378146 | -0.00536613 | 0.00455246 | inconclusive |
| Somatic | validate_2021-2023 | AJIVE7_rankchecked | RawX7 | 0.0178784 | -0.00418452 | 0.0389642 | inconclusive |
| CogAff | validate_2021-2023 | PCA7 | RawX7 | -0.000837594 | -0.00688274 | 0.00499074 | inconclusive |
| CogAff | validate_2021-2023 | FrozenZ7 | RawX7 | -0.000700253 | -0.0066098 | 0.00512988 | inconclusive |
| CogAff | validate_2021-2023 | JIVE7 | RawX7 | -0.00104966 | -0.00712579 | 0.00487741 | inconclusive |
| CogAff | validate_2021-2023 | AJIVE7_rankchecked | RawX7 | 0.0220779 | -0.0141375 | 0.0585288 | inconclusive |

## A-only, G-only and A+G versus depressive phenotypes

H4 is the four-marker haematological block; G3 is the three-marker glycaemic block. Every model includes the same covariates and uses identical participants. PHQ total, somatic and cognitive-affective outcomes are symptom scores, not diagnoses or cognitive-test performance. Positive gains do not identify individual biological causes.

| outcome | period | method | reference | paired_gain_positive_better | approx_PSU_low | approx_PSU_high | direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PHQ_total | test_2009-2018 | H4 | Covariates | 0.0500935 | -0.0558036 | 0.16275 | inconclusive |
| PHQ_total | test_2009-2018 | G3 | Covariates | 0.0140766 | -0.0333709 | 0.0614312 | inconclusive |
| PHQ_total | test_2009-2018 | RawX7 | Covariates | 0.0647225 | -0.0443498 | 0.18057 | inconclusive |
| PHQ_total | test_2009-2018 | FrozenZ7 | Covariates | 0.0639987 | -0.0409827 | 0.177598 | inconclusive |
| PHQ_total | test_2009-2018 | Threshold2 | Covariates | 0.0329373 | -0.0161613 | 0.0828246 | inconclusive |
| PHQ_total | test_2009-2018 | RawX7 | H4 | 0.0146289 | -0.0420931 | 0.0706301 | inconclusive |
| PHQ_total | test_2009-2018 | RawX7 | G3 | 0.0506459 | -0.0578496 | 0.163037 | inconclusive |
| PHQ_total | test_2009-2018 | RawX7 | Threshold2 | 0.0317852 | -0.0726264 | 0.139445 | inconclusive |
| Somatic | test_2009-2018 | H4 | Covariates | 0.013862 | -0.0141723 | 0.0428991 | inconclusive |
| Somatic | test_2009-2018 | G3 | Covariates | 0.00575557 | -0.00653825 | 0.0182505 | inconclusive |
| Somatic | test_2009-2018 | RawX7 | Covariates | 0.0181471 | -0.0113051 | 0.0494384 | inconclusive |
| Somatic | test_2009-2018 | FrozenZ7 | Covariates | 0.018625 | -0.00968521 | 0.0486748 | inconclusive |
| Somatic | test_2009-2018 | Threshold2 | Covariates | 0.0173545 | 0.0036096 | 0.0314377 | favours_method |
| Somatic | test_2009-2018 | RawX7 | H4 | 0.00428508 | -0.0109312 | 0.0194064 | inconclusive |
| Somatic | test_2009-2018 | RawX7 | G3 | 0.0123915 | -0.0160489 | 0.042105 | inconclusive |
| Somatic | test_2009-2018 | RawX7 | Threshold2 | 0.000792559 | -0.0263339 | 0.0283986 | inconclusive |
| CogAff | test_2009-2018 | H4 | Covariates | 0.0115812 | -0.0174076 | 0.0402256 | inconclusive |
| CogAff | test_2009-2018 | G3 | Covariates | 0.00112718 | -0.0126501 | 0.0143436 | inconclusive |
| CogAff | test_2009-2018 | RawX7 | Covariates | 0.0142897 | -0.0179003 | 0.0440137 | inconclusive |
| CogAff | test_2009-2018 | FrozenZ7 | Covariates | 0.0136344 | -0.0181924 | 0.0426436 | inconclusive |
| CogAff | test_2009-2018 | Threshold2 | Covariates | 0.000157122 | -0.012154 | 0.0129319 | inconclusive |
| CogAff | test_2009-2018 | RawX7 | H4 | 0.00270853 | -0.0130863 | 0.0172793 | inconclusive |
| CogAff | test_2009-2018 | RawX7 | G3 | 0.0131625 | -0.0162456 | 0.0415651 | inconclusive |
| CogAff | test_2009-2018 | RawX7 | Threshold2 | 0.0141326 | -0.015567 | 0.0430871 | inconclusive |
| PHQ_total | validate_2021-2023 | H4 | Covariates | 0.514642 | 0.15416 | 0.834788 | favours_method |
| PHQ_total | validate_2021-2023 | G3 | Covariates | 0.0654099 | -0.0194793 | 0.158719 | inconclusive |
| PHQ_total | validate_2021-2023 | RawX7 | Covariates | 0.623006 | 0.274093 | 0.935423 | favours_method |
| PHQ_total | validate_2021-2023 | FrozenZ7 | Covariates | 0.617189 | 0.274129 | 0.922632 | favours_method |
| PHQ_total | validate_2021-2023 | Threshold2 | Covariates | -0.0765722 | -0.194271 | 0.0262513 | inconclusive |
| PHQ_total | validate_2021-2023 | RawX7 | H4 | 0.108364 | 0.000284523 | 0.22146 | favours_method |
| PHQ_total | validate_2021-2023 | RawX7 | G3 | 0.557596 | 0.188848 | 0.881595 | favours_method |
| PHQ_total | validate_2021-2023 | RawX7 | Threshold2 | 0.699578 | 0.409887 | 0.975457 | favours_method |
| Somatic | validate_2021-2023 | H4 | Covariates | 0.108333 | 0.0272037 | 0.174667 | favours_method |
| Somatic | validate_2021-2023 | G3 | Covariates | 0.0191661 | -0.00192969 | 0.0428822 | inconclusive |
| Somatic | validate_2021-2023 | RawX7 | Covariates | 0.136097 | 0.0653714 | 0.196551 | favours_method |
| Somatic | validate_2021-2023 | FrozenZ7 | Covariates | 0.135863 | 0.0688794 | 0.195135 | favours_method |
| Somatic | validate_2021-2023 | Threshold2 | Covariates | -0.0152679 | -0.0376845 | 0.00429362 | inconclusive |
| Somatic | validate_2021-2023 | RawX7 | H4 | 0.0277633 | 0.00247065 | 0.055359 | favours_method |
| Somatic | validate_2021-2023 | RawX7 | G3 | 0.116931 | 0.037485 | 0.182422 | favours_method |
| Somatic | validate_2021-2023 | RawX7 | Threshold2 | 0.151364 | 0.086141 | 0.209248 | favours_method |
| CogAff | validate_2021-2023 | H4 | Covariates | 0.141245 | 0.0227455 | 0.239183 | favours_method |
| CogAff | validate_2021-2023 | G3 | Covariates | 0.0133004 | -0.0126692 | 0.0395597 | inconclusive |
| CogAff | validate_2021-2023 | RawX7 | Covariates | 0.166275 | 0.037447 | 0.275041 | favours_method |
| CogAff | validate_2021-2023 | FrozenZ7 | Covariates | 0.165574 | 0.0386022 | 0.275127 | favours_method |
| CogAff | validate_2021-2023 | Threshold2 | Covariates | -0.0220196 | -0.0582993 | 0.0100894 | inconclusive |
| CogAff | validate_2021-2023 | RawX7 | H4 | 0.0250296 | -0.00444583 | 0.0556629 | inconclusive |
| CogAff | validate_2021-2023 | RawX7 | G3 | 0.152974 | 0.0295955 | 0.256149 | favours_method |
| CogAff | validate_2021-2023 | RawX7 | Threshold2 | 0.188294 | 0.0786989 | 0.280811 | favours_method |

## What counts as an advantage

- The previously reproduced survey result supports modest information beyond threshold scalars. Continuous raw markers can retain that information too; it is not proof of a uniquely superior decomposition.
- Full-rank fidelity and explicitly named shared/discordant/domain-specific coordinates are construction properties. `113_coordinate_domain_support.csv` tests their algebraic loading support, not biological validity. Raw markers themselves also have domain-specific support.
- A predictive advantage must survive the specific paired comparisons above. A mixture of wins, losses and inconclusive intervals does not establish general superiority.
- Previous independent-refit/ferritin analyses provide exploratory structural and biological-anchor evidence. They were not unified into this matched prediction benchmark, and no new external cohort is introduced here.

## Uncertainty and reproducibility

999 paired Rao-Wu-style rescaled PSU replicates use the complete original DEMO design frame within cycle and stratum. Cluster-aggregated bootstrap calculations were checked against direct participant-weight calculations. Intervals condition on fitted models; they omit training/model-selection uncertainty, do not correct multiplicity, and do not undo repeated inspection of later outcomes. They are approximate sensitivity intervals, not universal design-based coverage guarantees.

The frozen protocol, model-selection table, source hashes, aggregate affine transformation maps and downstream model parameters are saved. No participant-level final records are exported. Check source hashes before reusing cached native JIVE fits.

## Paper implications

Present a transparent physiological decomposition and validation study with a complete A–D, G–D and combined evidence map. Show where threshold summaries discard information, where the full-coordinate maps tie, which compressed versions lose information, and how outcome transfer changes. Treat any domain-interpretation benefit as a specific design objective until independent anchors validate it. Independent statistical review and a genuinely independent replication remain separate requirements for stronger claims.

## Aggregate scorecard

| family | method | reference | analysis | comparisons | favours_method | favours_reference | inconclusive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rank5 | AJIVE5_rankchecked | PCA5 | prediction | 6 | 0 | 2 | 4 |
| rank5 | AJIVE5_rankchecked | PCA5 | reconstruction | 2 | 0 | 2 | 0 |
| rank5 | FrozenZ5 | PCA5 | prediction | 6 | 0 | 6 | 0 |
| rank5 | FrozenZ5 | PCA5 | reconstruction | 2 | 0 | 2 | 0 |
| rank5 | JIVE5 | PCA5 | prediction | 6 | 0 | 0 | 6 |
| rank5 | JIVE5 | PCA5 | reconstruction | 2 | 0 | 1 | 1 |
| rank7 | AJIVE7_rankchecked | RawX7 | prediction | 6 | 1 | 0 | 5 |
| rank7 | AJIVE7_rankchecked | RawX7 | reconstruction | 2 | 0 | 0 | 2 |
| rank7 | FrozenZ7 | RawX7 | prediction | 6 | 0 | 0 | 6 |
| rank7 | FrozenZ7 | RawX7 | reconstruction | 2 | 0 | 0 | 2 |
| rank7 | JIVE7 | RawX7 | prediction | 6 | 0 | 0 | 6 |
| rank7 | JIVE7 | RawX7 | reconstruction | 2 | 0 | 0 | 2 |
| rank7 | PCA7 | RawX7 | prediction | 6 | 0 | 0 | 6 |
| rank7 | PCA7 | RawX7 | reconstruction | 2 | 0 | 0 | 2 |
| triad | Discordant2 | Covariates | prediction | 6 | 3 | 0 | 3 |
| triad | FrozenZ7 | Covariates | prediction | 6 | 3 | 0 | 3 |
| triad | G3 | Covariates | prediction | 6 | 0 | 0 | 6 |
| triad | G_private1 | Covariates | prediction | 6 | 0 | 0 | 6 |
| triad | H4 | Covariates | prediction | 6 | 3 | 0 | 3 |
| triad | H_private2 | Covariates | prediction | 6 | 3 | 0 | 3 |
| triad | RawX7 | Covariates | prediction | 6 | 3 | 0 | 3 |
| triad | RawX7 | G3 | prediction | 6 | 3 | 0 | 3 |
| triad | RawX7 | H4 | prediction | 6 | 2 | 0 | 4 |
| triad | RawX7 | Threshold2 | prediction | 6 | 3 | 0 | 3 |
| triad | Shared2 | Covariates | prediction | 6 | 4 | 0 | 2 |
| triad | Threshold2 | Covariates | prediction | 6 | 1 | 0 | 5 |
