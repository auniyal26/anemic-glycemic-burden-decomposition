from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES_Transfer"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = ROOT / "Audit"
RESULTS = ROOT / "Results"

PROCESSED.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

PROTOCOL = AUDIT / "28_temporal_transfer_protocol_FROZEN.json"
if not PROTOCOL.exists():
    raise FileNotFoundError("Frozen transfer protocol not found.")

with open(PROTOCOL, "r", encoding="utf-8") as f:
    frozen = json.load(f)["protocol"]

CYCLES = {
    "0910": "F",
    "1112": "G",
    "1314": "H",
    "1516": "I",
    "1718": "J",
    "2123": "L",
}

FILES = ["DEMO", "DPQ", "CBC", "GHB", "BMX", "BIOPRO", "SMQ"]

REQUIRED_EXACT = {
    "DEMO": [
        "SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR",
        "DMDEDUC2", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"
    ],
    "DPQ": [
        "SEQN", "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050",
        "DPQ060", "DPQ070", "DPQ080", "DPQ090"
    ],
    "CBC": ["SEQN", "LBXHGB"],
    "GHB": ["SEQN", "LBXGH"],
    "BMX": ["SEQN", "BMXBMI"],
    "BIOPRO": ["SEQN", "LBXSCR"],
    "SMQ": ["SEQN", "SMQ020", "SMQ040"],
}

RACE_CANDIDATES = ["RIDRETH1", "RIDRETH3"]
PREG_CANDIDATES = ["RIDEXPRG"]

PHQ = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]

def read_component(cycle, suffix, component):
    path = DATA / cycle / f"{component}_{suffix}.XPT"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_sas(path, format="xport")

def clean_phq(x):
    x = x.copy()
    for c in PHQ:
        x.loc[x[c].abs() < 1e-10, c] = 0
        x.loc[~x[c].isin([0, 1, 2, 3]), c] = np.nan
    x["PHQ9_TOTAL"] = x[PHQ].sum(axis=1, min_count=9)
    x["SOMATIC_SCORE"] = x[SOMATIC].sum(axis=1, min_count=3)
    return x

def make_educ3(s):
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s.isin([1, 2])] = 1
    out[s == 3] = 2
    out[s.isin([4, 5])] = 3
    return out

def make_smoking3(df):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out[df["SMQ020"] == 2] = 0
    out[(df["SMQ020"] == 1) & (df["SMQ040"] == 3)] = 1
    out[(df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))] = 2
    return out

def egfr_2021(scr, age, sex):
    # CKD-EPI 2021 creatinine equation.
    female = (sex == 2).astype(float)
    k = np.where(female == 1, 0.7, 0.9)
    alpha = np.where(female == 1, -0.241, -0.302)
    ratio = scr / k
    return (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** (-1.200)
        * (0.9938 ** age)
        * np.where(female == 1, 1.012, 1.0)
    )

schema_rows = []
cycle_frames = []

print()
print("FROZEN TRANSFER HARMONIZATION AUDIT")
print("===================================")

for cycle, suffix in CYCLES.items():
    print(f"Checking {cycle}...")

    parts = {}
    for component in FILES:
        df = read_component(cycle, suffix, component)
        parts[component] = df

        for var in REQUIRED_EXACT[component]:
            schema_rows.append({
                "cycle": cycle,
                "component": component,
                "variable": var,
                "present": var in df.columns
            })

    demo = parts["DEMO"]

    race_var = next((c for c in RACE_CANDIDATES if c in demo.columns), None)
    preg_var = next((c for c in PREG_CANDIDATES if c in demo.columns), None)

    schema_rows.append({
        "cycle": cycle,
        "component": "DEMO",
        "variable": "RACE_VARIABLE_USED",
        "present": race_var is not None,
        "resolved_name": race_var
    })
    schema_rows.append({
        "cycle": cycle,
        "component": "DEMO",
        "variable": "PREG_VARIABLE_USED",
        "present": preg_var is not None,
        "resolved_name": preg_var
    })

    hard_missing = []
    for component, vars_ in REQUIRED_EXACT.items():
        for var in vars_:
            if var not in parts[component].columns:
                hard_missing.append(f"{component}:{var}")

    if race_var is None:
        hard_missing.append("DEMO:RIDRETH1/RIDRETH3")

    if hard_missing:
        print(f"FAIL  {cycle}: missing required fields")
        for x in hard_missing:
            print("     ", x)
        continue

    d = demo[
        ["SEQN", "RIDAGEYR", "RIAGENDR", race_var, "INDFMPIR",
         "DMDEDUC2", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
        + ([preg_var] if preg_var else [])
    ].copy()

    d = d.rename(columns={race_var: "RACE_RAW"})
    if preg_var:
        d = d.rename(columns={preg_var: "RIDEXPRG"})

    d = d.merge(
        clean_phq(parts["DPQ"][["SEQN"] + PHQ]),
        on="SEQN", how="inner", validate="one_to_one"
    )
    d = d.merge(
        parts["CBC"][["SEQN", "LBXHGB"]],
        on="SEQN", how="inner", validate="one_to_one"
    )
    d = d.merge(
        parts["GHB"][["SEQN", "LBXGH"]],
        on="SEQN", how="inner", validate="one_to_one"
    )
    d = d.merge(
        parts["BMX"][["SEQN", "BMXBMI"]],
        on="SEQN", how="left", validate="one_to_one"
    )
    d = d.merge(
        parts["BIOPRO"][["SEQN", "LBXSCR"]],
        on="SEQN", how="left", validate="one_to_one"
    )
    d = d.merge(
        parts["SMQ"][["SEQN", "SMQ020", "SMQ040"]],
        on="SEQN", how="left", validate="one_to_one"
    )

    d["CYCLE"] = cycle
    d["RACE_SOURCE"] = race_var

    # Frozen eligibility.
    d = d[d["RIDAGEYR"] >= 20].copy()
    if "RIDEXPRG" in d.columns:
        d = d[d["RIDEXPRG"] != 1].copy()

    # Frozen A.
    hb_threshold = np.where(
        d["RIAGENDR"] == 1, 13.0,
        np.where(d["RIAGENDR"] == 2, 12.0, np.nan)
    )
    d["A"] = np.maximum(hb_threshold - d["LBXHGB"], 0)

    # Frozen G.
    d["G_HBA1C"] = np.maximum(d["LBXGH"] - 5.7, 0)
    d["AG_HBA1C"] = d["A"] * d["G_HBA1C"]

    # Frozen covariates.
    d["EDUC3"] = make_educ3(d["DMDEDUC2"])
    d["SMOKING3"] = make_smoking3(d)
    d["EGFR_2021"] = egfr_2021(
        pd.to_numeric(d["LBXSCR"], errors="coerce"),
        pd.to_numeric(d["RIDAGEYR"], errors="coerce"),
        pd.to_numeric(d["RIAGENDR"], errors="coerce")
    )

    # Do NOT silently convert race coding systems.
    # We retain raw coding plus source variable and audit it before modeling.
    d["RIDRETH_FROZEN"] = d["RACE_RAW"]

    cycle_frames.append(d)

schema = pd.DataFrame(schema_rows)
schema.to_csv(RESULTS / "30_transfer_schema_audit.csv", index=False)

if len(cycle_frames) != len(CYCLES):
    print()
    print("STOP  Not all cycles passed schema audit.")
    print("No validation model has been fit.")
    raise SystemExit(2)

all_data = pd.concat(cycle_frames, ignore_index=True)

# Primary complete-case variables, except race requires a coding audit first.
X3 = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN",
    "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"
]
primary_required = [
    "SOMATIC_SCORE", "A", "G_HBA1C", "WTMEC2YR",
    "SDMVPSU", "SDMVSTRA"
] + X3

summary_rows = []

for cycle in CYCLES:
    q = all_data[all_data["CYCLE"] == cycle].copy()
    complete = q.dropna(subset=primary_required)

    summary_rows.append({
        "cycle": cycle,
        "eligible_20plus_nonpreg_n": len(q),
        "primary_complete_case_n": len(complete),
        "complete_case_rate": len(complete) / len(q) if len(q) else np.nan,
        "race_source": q["RACE_SOURCE"].iloc[0] if len(q) else None
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(RESULTS / "30_transfer_harmonization_summary.csv", index=False)

all_data.to_parquet(
    PROCESSED / "30_transfer_harmonized_PREMODEL.parquet",
    index=False
)

race_audit = (
    all_data.groupby(["CYCLE", "RACE_SOURCE", "RIDRETH_FROZEN"], dropna=False)
    .size()
    .reset_index(name="n")
)
race_audit.to_csv(RESULTS / "30_race_coding_audit.csv", index=False)

print()
print("PASS  All cycles harmonized without fitting validation models.")
print(f"PASS  Premodel cohort rows: {len(all_data):,}")
print()
print(summary.to_string(index=False))
print()
print("IMPORTANT")
print("=========")
print("Race coding is intentionally NOT harmonized silently.")
print("Inspect Results/30_race_coding_audit.csv before the first transfer model.")
print("If later cycles use RIDRETH3 instead of RIDRETH1, we will define the")
print("cross-cycle category mapping transparently before viewing any outcome coefficients.")
print()
print("Saved:")
print("  Results/30_transfer_schema_audit.csv")
print("  Results/30_transfer_harmonization_summary.csv")
print("  Results/30_race_coding_audit.csv")
print("  Data/Processed/30_transfer_harmonized_PREMODEL.parquet")
