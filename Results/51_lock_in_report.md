# Chapter 2 lock-in falsification report

**CORE LOCK STATUS: PASS**

This is a falsification/stability audit, not mathematical proof that all bias is absent. A PASS means the prespecified core claim survived the checks implemented here.

## Summary
- PASS: 53
- WARN: 2
- FAIL: 0
- INFO: 3
- UNTESTABLE: 7

## Findings

| Category | Test | Status | Critical | Evidence |
|---|---|---:|---:|---|
| representation | Outcome leakage static audit | **PASS** | yes | No operational depression/outcome tokens in Script 46. |
| representation | Script 46 audit manifest | **PASS** | yes | reads_PHQ_or_depression=False; A_retained_components=3 |
| representation | Cross-cycle loading similarity | **PASS** | yes | minimum aligned loading similarity = 0.994483 |
| representation | Frozen A reconstruction | **PASS** | yes | minimum reconstruction-energy fraction = 0.992364 |
| primary_replication | 2005-2008: components beyond scalars | **PASS** | yes | p=0.0140752 < 0.05 |
| primary_replication | 2005-2008: scalars beyond components | **PASS** | no | p=0.965949 >= 0.05 |
| primary_replication | 2005-2008: incremental weighted R2 asymmetry | **PASS** | yes | PCs beyond scalars ΔR²=0.010197; scalars beyond PCs ΔR²=0.000034 |
| survey_design | 2005-2008: design-before-domain df preservation | **PASS** | no | full design df=31; domain df=31 |
| primary_replication | 2009-2018: components beyond scalars | **PASS** | yes | p=0.00612027 < 0.05 |
| primary_replication | 2009-2018: scalars beyond components | **PASS** | no | p=0.149261 >= 0.05 |
| primary_replication | 2009-2018: incremental weighted R2 asymmetry | **PASS** | yes | PCs beyond scalars ΔR²=0.003686; scalars beyond PCs ΔR²=0.000543 |
| survey_design | 2009-2018: design-before-domain df preservation | **PASS** | no | full design df=78; domain df=78 |
| outcome_sensitivity | 2005-2008: PHQ9 total components beyond scalars | **PASS** | no | p=0.0104334 < 0.05 |
| outcome_sensitivity | 2009-2018: PHQ9 total components beyond scalars | **PASS** | no | p=0.0195974 < 0.05 |
| outcome_sensitivity | Cognitive-affective replication | **INFO** | no | 2009-2018 p=0.136869 |
| representation_sensitivity | 2005-2008: raw7 beyond threshold scalars | **PASS** | yes | p=0.0220242 < 0.05 |
| representation_sensitivity | 2005-2008: threshold scalars beyond raw7 | **PASS** | no | p=0.977979 >= 0.05 |
| representation_sensitivity | 2009-2018: raw7 beyond threshold scalars | **PASS** | yes | p=0.0103119 < 0.05 |
| representation_sensitivity | 2009-2018: threshold scalars beyond raw7 | **PASS** | no | p=0.147703 >= 0.05 |
| representation_sensitivity | 2005-2008: raw7 vs PCA R2 proximity | **PASS** | no | \|R²_raw7 - R²_PCA\| = 0.000723 |
| representation_sensitivity | 2009-2018: raw7 vs PCA R2 proximity | **PASS** | no | \|R²_raw7 - R²_PCA\| = 0.000030 |
| model_specification | 2005-2008: quasi-Poisson components beyond scalars | **PASS** | yes | p=0.00361275 < 0.05 |
| model_specification | 2009-2018: quasi-Poisson components beyond scalars | **PASS** | yes | p=0.00297345 < 0.05 |
| measurement_sensitivity | 2005-2008: no-insulin sensitivity | **PASS** | yes | p=0.0164035 < 0.05 |
| measurement_sensitivity | 2005-2008: no-HbA1c sensitivity | **PASS** | yes | p=0.000696468 < 0.05 |
| measurement_sensitivity | 2009-2018: no-insulin sensitivity | **PASS** | yes | p=0.00398515 < 0.05 |
| measurement_sensitivity | 2009-2018: no-HbA1c sensitivity | **PASS** | yes | p=0.00465671 < 0.05 |
| interaction_nonclaim | Replication: broad A×G interaction not supported | **PASS** | no | p=0.752591 >= 0.05 |
| temporal_mapping | A linear temporal drift | **PASS** | no | p=0.858115 >= 0.05 |
| temporal_mapping | G block linear temporal drift | **PASS** | no | p=0.0188571 < 0.05 |
| temporal_mapping | All-component linear drift | **INFO** | no | p=0.0631264 |
| multiplicity | Individual temporal interactions after BH | **PASS** | no | minimum BH-adjusted p = 0.0606428 |
| complexity_control | 2005-2008: 2h OGTT beyond G3 | **PASS** | no | p=0.416322 >= 0.05 |
| complexity_control | 2009-2016: 2h OGTT beyond G3 | **PASS** | no | p=0.0624043 >= 0.05 |
| complexity_control | 2005-2008: G4 same-sample R2 gain | **PASS** | no | R²(G4)-R²(G3)=0.000146 |
| complexity_control | 2009-2016: G4 same-sample R2 gain | **PASS** | no | R²(G4)-R²(G3)=0.000947 |
| multiplicity | G4 modifier family after BH | **PASS** | no | minimum modifier-family BH q=0.646496 |
| survey_design | 2021-23 full X3 inference | **PASS** | no | All primary 2021-23 X3 block tests are design-limited. |
| irreducible_limitations | Unmeasured confounding | **UNTESTABLE** | no | No observational adjustment set can prove that all common causes were measured. |
| irreducible_limitations | Reverse causation | **UNTESTABLE** | no | NHANES is cross-sectional within each cycle. |
| irreducible_limitations | Collider/overadjustment | **UNTESTABLE** | no | BMI/eGFR may have different causal roles under different biological DAGs. |
| irreducible_limitations | MNAR missingness | **UNTESTABLE** | no | Missingness mechanisms involving unobserved values are not identified. |
| irreducible_limitations | External transportability | **UNTESTABLE** | no | 2009-18 is temporal transfer within NHANES. |
| irreducible_limitations | True longitudinal change | **UNTESTABLE** | no | NHANES cycles are repeated cross-sections. |
| new_stress_tests | R stress-test execution | **PASS** | yes | Same-sample adjustment ladder and arbitrary cycle-heterogeneity tests completed. |
| adjustment_ladder | 2005-2008 X0: PCs beyond scalars | **PASS** | yes | p=4.27897e-05; ΔR²=0.024302; same X3-complete sample n=3830 |
| adjustment_ladder | 2005-2008 X1: PCs beyond scalars | **PASS** | yes | p=0.000431154; ΔR²=0.017754; same X3-complete sample n=3830 |
| adjustment_ladder | 2005-2008 X2: PCs beyond scalars | **PASS** | yes | p=0.0102126; ΔR²=0.010276; same X3-complete sample n=3830 |
| adjustment_ladder | 2005-2008 X3: PCs beyond scalars | **PASS** | yes | p=0.0140752; ΔR²=0.010197; same X3-complete sample n=3830 |
| adjustment_ladder | 2009-2018 X0: PCs beyond scalars | **PASS** | yes | p=6.88576e-10; ΔR²=0.016056; same X3-complete sample n=9919 |
| adjustment_ladder | 2009-2018 X1: PCs beyond scalars | **PASS** | yes | p=3.33064e-06; ΔR²=0.009364; same X3-complete sample n=9919 |
| adjustment_ladder | 2009-2018 X2: PCs beyond scalars | **PASS** | yes | p=0.00516247; ΔR²=0.003717; same X3-complete sample n=9919 |
| adjustment_ladder | 2009-2018 X3: PCs beyond scalars | **PASS** | yes | p=0.00612027; ΔR²=0.003686; same X3-complete sample n=9919 |
| adjustment_ladder | Reverse scalar advantage across adjustment ladder | **PASS** | no | No adjustment set shows a statistically supported reverse scalar advantage. |
| temporal_mapping | A: arbitrary cycle-specific slope heterogeneity | **PASS** | no | p=0.604635 |
| temporal_mapping | G3: arbitrary cycle-specific slope heterogeneity | **INFO** | no | p=0.13063 |
| selection | 2005-2008 analytic retention | **PASS** | no | eligible n=4,869; included n=3,830; unweighted retention=0.787; weighted retention=0.879 |
| selection | 2009-2018 analytic retention | **PASS** | no | eligible n=13,223; included n=9,919; unweighted retention=0.750; weighted retention=0.836 |
| selection | 2021-2023 analytic retention | **PASS** | no | eligible n=3,397; included n=2,261; unweighted retention=0.666; weighted retention=0.713 |
| selection | Maximum observed-variable selection SMD | **WARN** | no | max \|weighted SMD\| = 0.430 |
| selection | MNAR / unobserved selection | **UNTESTABLE** | no | Observed data cannot establish that missingness is independent of unobserved physiology/phenotype. |
| model_specification | 2005-2008 component-only VIF | **PASS** | no | max VIF among six frozen PCs = 1.069 |
| model_specification | 2009-2018 component-only VIF | **PASS** | no | max VIF among six frozen PCs = 1.062 |
| reproducibility | Git commit identified | **PASS** | no | branch=main; commit=b3749c2cb980d2fcad2a431b06281e7bcfadbbb1 |
| reproducibility | Working tree cleanliness | **WARN** | no | Working tree has uncommitted changes. |

## Locked claim if CORE LOCK STATUS = PASS

> In NHANES, a frozen outcome-independent multidimensional physiological representation retains phenotype-associated information beyond simple threshold scalars, and that advantage reproduces under frozen 2009-18 temporal transfer. The result is associative, not causal.

## Irreducible limitations

Unmeasured confounding, MNAR missingness/selection, reverse causation, causal roles of BMI/eGFR, and transportability outside NHANES cannot be eliminated by this script. They remain explicit limitations even when the lock passes.
