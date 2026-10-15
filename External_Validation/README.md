# External Validation — MIDUS / MIDJA intake

This folder is a separate scientific validation branch. The existing `EDA/` scripts and all previously frozen NHANES transformations are unchanged.

## Canonical files and numbering

- `00_dataset_inventory.py`: metadata-only inventory of MIDUS and MIDJA staged SPSS files; outputs `Results/00_inventory.md` and `Results/00_dataset_manifest.json`.
- `01_midus_schema_linkage_audit.py`: full SPSS variable-label search for Hb, RBC, MCV, RDW, HbA1c, glucose, insulin, depression, cognition, and covariates; on-device aggregate participant ID overlap; outputs `Results/01_readout.md`, `Results/01_candidate_variables.csv`, `Results/01_linkage_counts.csv`, `Results/01_variable_catalog.csv`, and `Results/01_audit.json`.

The previous `01_acquire_datasets.py` is obsolete and is **not** part of this sequence. The former `02_midus_schema_linkage_audit.py` was renumbered to `01`. KNHANES is deferred and is not scanned by either script.

## Directory layout

```text
<project root>/
├── data/
│   └── MIDUS/
│       ├── raw/            # user-acquired original .sav files, unchanged
│       └── docs/
├── EDA/                    # old NHANES experiments, unchanged
└── External_Validation/
    ├── 00_dataset_inventory.py
    ├── 01_midus_schema_linkage_audit.py
    ├── Results/
    └── README.md
```

## Running from Windows VS Code terminal

```powershell
cd "D:\Hematological and Glycemic Decomposition of Depression"
python -m pip install pyreadstat
python External_Validation\00_dataset_inventory.py
python External_Validation\01_midus_schema_linkage_audit.py
Get-Content External_Validation\Results\01_readout.md
```

For schema-only inspection without ID-join calculations, run script 01 with `--schema-only`.

## Restrictions and scientific interpretation

- Source datasets remain local; scripts emit metadata, provisional variable-name/label matches, and aggregate joins only. IDs and participant rows are never written into results.
- Numeric aggregate cells <=10 are suppressed in script 01 (per MIDUS disclosure policy).
- Candidate variables are **not** codebook-verified. Do not use them in modeling until units, measurement timing, fasting status, special missing values, cross-wave equivalence, and overlapping complete cases are established.
- Separate original MIDUS, Refresher, and MIDJA populations; never join them as one cohort using numeric identifiers.
- Phenotype CES-D/MASQ cannot be treated as equivalent to PHQ-9 without an explicit instrument adaptation.
- NHANES's seven-marker frozen transform must not be declared externally replicated when any original marker is absent or redefined.
