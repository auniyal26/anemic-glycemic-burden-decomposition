from pathlib import Path
import hashlib
import json
from datetime import datetime

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
SCRIPTS = ROOT / "Scripts"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"

AUDIT.mkdir(parents=True, exist_ok=True)

PROTOCOL = {
    "protocol_name": "Temporal transfer validation of hematological and glycemic associations with somatic depressive symptoms",
    "frozen_on": "2026-08-22",
    "discovery_data": ["NHANES 2005-2006", "NHANES 2007-2008"],
    "temporal_replication_data": [
        "NHANES 2009-2010",
        "NHANES 2011-2012",
        "NHANES 2013-2014",
        "NHANES 2015-2016",
        "NHANES 2017-2018"
    ],
    "final_modern_holdout": ["NHANES August 2021-August 2023"],
    "analysis_population": {
        "age": ">=20 years",
        "pregnancy": "exclude known pregnant participants when RIDEXPRG == 1",
        "complete_case_primary": True
    },
    "outcomes": {
        "primary": {
            "name": "SOMATIC_SCORE",
            "definition": "DPQ030 + DPQ040 + DPQ050",
            "items": ["sleep", "fatigue", "appetite"]
        },
        "secondary": {
            "name": "PHQ9_TOTAL",
            "definition": "sum DPQ010-DPQ090 when all 9 items are valid 0-3"
        }
    },
    "exposures": {
        "A": {
            "name": "hematological deficit proxy",
            "formula": "max(HB_THRESHOLD - LBXHGB, 0)",
            "HB_THRESHOLD": {"male_RIAGENDR_1": 13.0, "female_RIAGENDR_2": 12.0}
        },
        "G_HBA1C": {
            "name": "glycemic excess proxy",
            "formula": "max(LBXGH - 5.7, 0)"
        },
        "AG_HBA1C": {
            "formula": "A * G_HBA1C",
            "status": "exploratory interaction"
        }
    },
    "confounders": {
        "X0_demographic": ["RIDAGEYR", "RIAGENDR", "RIDRETH1"],
        "X1_primary": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "EDUC3", "SMOKING3"],
        "X2_bmi_sensitivity": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI"],
        "X3_kidney_direct_effect": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"]
    },
    "derived_covariates": {
        "EDUC3": {
            "DMDEDUC2_1_or_2": 1,
            "DMDEDUC2_3": 2,
            "DMDEDUC2_4_or_5": 3
        },
        "SMOKING3": {
            "never": "SMQ020 == 2 -> 0",
            "former": "SMQ020 == 1 and SMQ040 == 3 -> 1",
            "current": "SMQ020 == 1 and SMQ040 in [1,2] -> 2"
        },
        "EGFR_2021": "2021 CKD-EPI creatinine equation exactly as in script 15"
    },
    "primary_model": "SOMATIC_SCORE ~ A + G_HBA1C + X3",
    "interaction_model": "SOMATIC_SCORE ~ A + G_HBA1C + AG_HBA1C + X3",
    "secondary_model": "PHQ9_TOTAL ~ A + G_HBA1C + X3",
    "survey_design": {
        "2009_2018_pooled": "WTMEC2YR/5; cycle-prefixed strata and PSU",
        "2021_2023": "WTMEC2YR as released; native SDMVSTRA and SDMVPSU",
        "fasting_sensitivity_2009_2018": "WTSAF2YR/5",
        "fasting_sensitivity_2021_2023": "WTSAF2YR as released",
        "inference": "R survey::svyglm / design-based Wald inference"
    },
    "replication_rules": {
        "primary_A": {
            "minimum": "beta_A > 0 in 2009-2018 pooled and 2021-2023 holdout",
            "strong": "design-based 95% CI excludes 0 in pooled 2009-2018; modern holdout same direction"
        },
        "secondary_G": {
            "minimum": "beta_G > 0 and weighted incremental delta-R2 > 0 in both validation periods",
            "strong": "design-based 95% CI excludes 0 in pooled 2009-2018 and modern holdout"
        },
        "interaction_AG": {
            "rule": "exploratory only; report sign/effect/CI without promoting to primary claim unless independently stable"
        },
        "failure": [
            "A reverses direction in either major validation period",
            "G reverses direction in either major validation period",
            "apparent replication requires changing thresholds, outcome items, or confounder definitions after inspecting results"
        ]
    },
    "anti_leakage_rules": [
        "Do not change A, G, somatic items, X sets, or success criteria after viewing validation outcomes.",
        "Do not tune thresholds on 2009-2018 or 2021-2023.",
        "Do not use 2021-2023 to choose models.",
        "Any post-hoc model is labelled exploratory and kept separate from frozen validation."
    ],
    "planned_sensitivities": [
        "fasting glucose replaces HbA1c-based G using fasting-subsample weights",
        "clinical diabetes state",
        "X0 -> X1 -> X2 -> X3 adjustment ladder",
        "missingness / included-vs-excluded audit"
    ]
}

FILES_TO_HASH = [
    SCRIPTS / "15_build_x_audit_and_dag_FINAL.py",
    SCRIPTS / "16_staged_survey_models.py",
    SCRIPTS / "20_ordinal_cfa_validation.py",
    SCRIPTS / "21_mimic_sem_validation.py",
    SCRIPTS / "25_pre_presentation_validity_audit.py",
    SCRIPTS / "26_formal_separability_test.py",
    SCRIPTS / "27_incremental_information_test.py",
    RESULTS / "25_survey_estimator_parity.csv",
    RESULTS / "26_design_based_coefficients.csv",
    RESULTS / "27_incremental_information_tests.csv"
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = []
for path in FILES_TO_HASH:
    manifest.append({
        "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
        "exists": path.exists(),
        "sha256": sha256(path) if path.exists() else None
    })

payload = {
    "protocol": PROTOCOL,
    "discovery_artifact_manifest": manifest
}

json_path = AUDIT / "28_temporal_transfer_protocol_FROZEN.json"
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

md = f"""# Frozen Temporal Transfer Protocol

**Frozen:** 2026-08-22  
**Purpose:** test whether the 2005-2008 NHANES finding transfers forward in time without changing the model after seeing new outcomes.

## Discovery
- NHANES 2005-2006
- NHANES 2007-2008

## Validation
- Temporal replication: 2009-2018
- Final modern holdout: August 2021-August 2023

## Frozen primary outcome
`SOMATIC_SCORE = DPQ030 + DPQ040 + DPQ050`

## Frozen exposure definitions
- `A = max(HB_THRESHOLD - LBXHGB, 0)`
  - male threshold = 13.0 g/dL
  - female threshold = 12.0 g/dL
- `G_HBA1C = max(LBXGH - 5.7, 0)`
- `AG_HBA1C = A * G_HBA1C` — exploratory interaction

## Frozen X3 adjustment
`RIDAGEYR + RIAGENDR + RIDRETH1 + INDFMPIR + EDUC3 + SMOKING3 + BMXBMI + EGFR_2021`

## Frozen primary model
`SOMATIC_SCORE ~ A + G_HBA1C + X3`

## Frozen secondary models
- `PHQ9_TOTAL ~ A + G_HBA1C + X3`
- `SOMATIC_SCORE ~ A + G_HBA1C + AG_HBA1C + X3`

## Transfer criteria
A minimum replication requires:
- A remains positive in 2009-2018 and 2021-2023.
- G remains positive and adds non-zero weighted incremental information in both periods.
- No thresholds, PHQ items, or confounder definitions are changed after outcome inspection.

A strong replication additionally requires design-based confidence intervals excluding zero where the survey design has adequate precision.

A×G remains exploratory unless it independently stabilizes.

## Survey design
- 2009-2018 pooled full-sample weight: `WTMEC2YR / 5`
- 2021-2023: released `WTMEC2YR`
- fasting sensitivity: corresponding `WTSAF2YR` weights
- inference: R `survey` design-based models

## Rule
The validation data are tests of the frozen model, not a new model-selection dataset.

Full machine-readable protocol and discovery file hashes:
`Audit/28_temporal_transfer_protocol_FROZEN.json`
"""

md_path = AUDIT / "28_temporal_transfer_protocol_FROZEN.md"
md_path.write_text(md, encoding="utf-8")

print()
print("TEMPORAL TRANSFER PROTOCOL FREEZE")
print("=================================")
print(f"PASS  Frozen protocol: {json_path}")
print(f"PASS  Human-readable protocol: {md_path}")

missing = [x["path"] for x in manifest if not x["exists"]]
if missing:
    print("NOTE  Some expected artifacts were not found under these exact names:")
    for x in missing:
        print("     ", x)
    print("      This does not change the frozen definitions above.")
else:
    print("PASS  All listed discovery artifacts hashed.")

print()
print("DO NOT EDIT THIS PROTOCOL AFTER VIEWING VALIDATION RESULTS.")
