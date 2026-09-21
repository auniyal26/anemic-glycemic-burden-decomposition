from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "Synthetic_Longitudinal" else Path.cwd()
DATA_DISC = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
SIM_DIR = ROOT / "Synthetic_Longitudinal"
SIM_RESULTS = SIM_DIR / "Results"
SIM_RESULTS.mkdir(parents=True, exist_ok=True)

A_FILE = PROCESSED / "46_outcome_independent_A_discovery_scores.parquet"
G_FILE = PROCESSED / "46_outcome_independent_G3_discovery_scores.parquet"

DISC = {"0506": "D", "0708": "E"}
PHQ = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]
A_RAW = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G_RAW = ["LBXGH", "LBXGLU", "LBXIN", "LOG_IN"]
A_PCS = [f"A_OI_PC{i}" for i in range(1, 4)]
G_PCS = [f"G3_OI_PC{i}" for i in range(1, 4)]
X3 = ["RIDAGEYR", "RIAGENDR", "RACE", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"]

OUT_PARQUET = SIM_DIR / "51_simulation_calibration_2005_2008.parquet"
OUT_MANIFEST = SIM_DIR / "51_simulation_calibration_manifest.json"
OUT_SUMMARY = SIM_RESULTS / "51_calibration_summary.csv"

print()
print("SYNTHETIC LONGITUDINAL — CALIBRATION COHORT")
print("===========================================")
print("Allowed source cycles: 2005-06 and 2007-08 ONLY.")
print("2009+ NHANES data are not read by this script.")
print("Frozen A/G representations come from outcome-independent Script 46 outputs.")
print()

for p in [A_FILE, G_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run NHANES Script 46 first.")


def xpt_path(base: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")


def clean_phq(dpq: pd.DataFrame) -> pd.DataFrame:
    d = dpq[["SEQN"] + PHQ].copy()
    for c in PHQ:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[d[c].abs() < 1e-10, c] = 0
        d.loc[~d[c].isin([0, 1, 2, 3]), c] = np.nan
    d["PHQ9_TOTAL"] = d[PHQ].sum(axis=1, min_count=9)
    d["SOMATIC_SCORE"] = d[SOMATIC].sum(axis=1, min_count=3)
    d["COGAFF_SUM"] = d["PHQ9_TOTAL"] - d["SOMATIC_SCORE"]
    return d


def educ3(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out[x.isin([1, 2])] = 1
    out[x.eq(3)] = 2
    out[x.isin([4, 5])] = 3
    return out


def smoking3(df: pd.DataFrame) -> pd.Series:
    s20 = pd.to_numeric(df["SMQ020"], errors="coerce")
    s40 = pd.to_numeric(df["SMQ040"], errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out[s20.eq(2)] = 0
    out[s20.eq(1) & s40.eq(3)] = 1
    out[s20.eq(1) & s40.isin([1, 2])] = 2
    return out


def egfr_2021(scr, age, sex):
    scr = pd.to_numeric(scr, errors="coerce").to_numpy(float)
    age = pd.to_numeric(age, errors="coerce").to_numpy(float)
    sex = pd.to_numeric(sex, errors="coerce").to_numpy(float)
    female = sex == 2
    k = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / k
    return (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** -1.200
        * 0.9938 ** age
        * np.where(female, 1.012, 1.0)
    )


def build_cycle(cycle: str, suffix: str) -> pd.DataFrame:
    if cycle not in DISC:
        raise RuntimeError(f"Leakage guard: forbidden cycle {cycle}")

    base = DATA_DISC / cycle
    demo = pd.read_sas(xpt_path(base, f"DEMO_{suffix}"), format="xport")
    glu = pd.read_sas(xpt_path(base, f"GLU_{suffix}"), format="xport")
    dpq = pd.read_sas(xpt_path(base, f"DPQ_{suffix}"), format="xport")
    bmx = pd.read_sas(xpt_path(base, f"BMX_{suffix}"), format="xport")
    bio = pd.read_sas(xpt_path(base, f"BIOPRO_{suffix}"), format="xport")
    smq = pd.read_sas(xpt_path(base, f"SMQ_{suffix}"), format="xport")

    required_demo = [
        "SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "DMDEDUC2",
        "SDMVPSU", "SDMVSTRA",
    ]
    missing = [c for c in required_demo if c not in demo.columns]
    if missing:
        raise ValueError(f"{cycle} DEMO missing {missing}")
    if "WTSAF2YR" not in glu.columns:
        raise ValueError(f"{cycle} GLU missing WTSAF2YR")

    race = "RIDRETH1" if "RIDRETH1" in demo.columns else None
    if race is None:
        raise ValueError(f"{cycle} expected RIDRETH1")

    demo_cols = required_demo + [race]
    if "RIDEXPRG" in demo.columns:
        demo_cols.append("RIDEXPRG")

    d = demo[demo_cols].copy().rename(columns={race: "RACE"})
    d = d.merge(glu[["SEQN", "WTSAF2YR"]], on="SEQN", how="left", validate="one_to_one")

    # Build the full positive-fasting-weight design frame first.
    d["WTSAF2YR"] = pd.to_numeric(d["WTSAF2YR"], errors="coerce")
    d = d[d["WTSAF2YR"] > 0].copy()

    d = d.merge(clean_phq(dpq), on="SEQN", how="left", validate="one_to_one")
    d = d.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left", validate="one_to_one")
    d = d.merge(bio[["SEQN", "LBXSCR"]], on="SEQN", how="left", validate="one_to_one")
    d = d.merge(smq[["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left", validate="one_to_one")

    d["CYCLE"] = cycle
    d["CALENDAR_MIDYEAR"] = 2006 if cycle == "0506" else 2008
    d["SIM_ROLE"] = "CALIBRATION_FIT" if cycle == "0506" else "CALIBRATION_CHECK"
    d["EDUC3"] = educ3(d["DMDEDUC2"])
    d["SMOKING3"] = smoking3(d)
    d["EGFR_2021"] = egfr_2021(d["LBXSCR"], d["RIDAGEYR"], d["RIAGENDR"])

    age_ok = pd.to_numeric(d["RIDAGEYR"], errors="coerce") >= 20
    preg_ok = pd.Series(True, index=d.index)
    if "RIDEXPRG" in d.columns:
        preg_ok = ~pd.to_numeric(d["RIDEXPRG"], errors="coerce").eq(1)
    d["ELIGIBLE_ADULT_NONPREG"] = (age_ok & preg_ok).astype(int)

    # Keep both native 2-year weight and combined 4-year discovery weight.
    d["SURVEY_WT_2YR"] = d["WTSAF2YR"]
    d["SURVEY_WT_4YR"] = d["WTSAF2YR"] / 2.0
    d["STRATUM"] = d["CYCLE"].astype(str) + "_" + d["SDMVSTRA"].astype("Int64").astype(str)
    d["PSU"] = d["STRATUM"] + "_" + d["SDMVPSU"].astype("Int64").astype(str)
    d["SOURCE_ID"] = d["CYCLE"].astype(str) + "_" + d["SEQN"].astype("Int64").astype(str)
    return d


# Raw covariates / phenotype from discovery cycles only.
base = pd.concat([build_cycle(cy, suf) for cy, suf in DISC.items()], ignore_index=True)

# Frozen outcome-independent physiology from Script 46.
a = pd.read_parquet(A_FILE)
g = pd.read_parquet(G_FILE)

if not set(a["CYCLE"].astype(str).unique()).issubset(DISC):
    raise RuntimeError("Leakage guard failed: A discovery file contains non-discovery cycles")
if not set(g["CYCLE"].astype(str).unique()).issubset(DISC):
    raise RuntimeError("Leakage guard failed: G discovery file contains non-discovery cycles")

keep_a = ["SEQN", "CYCLE", "A_REP_WEIGHT"] + A_RAW + A_PCS
keep_g = ["SEQN", "CYCLE", "G_REP_WEIGHT", "WTSAF2YR"] + G_RAW + G_PCS

# WTSAF2YR already exists in base; keep G's copy only as an audit check.
g = g[keep_g].rename(columns={"WTSAF2YR": "WTSAF2YR_G_AUDIT"})
full = base.merge(a[keep_a], on=["SEQN", "CYCLE"], how="left", validate="one_to_one")
full = full.merge(g, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

# Audit weight agreement where both are observed.
w1 = pd.to_numeric(full["WTSAF2YR"], errors="coerce")
w2 = pd.to_numeric(full["WTSAF2YR_G_AUDIT"], errors="coerce")
weight_ok = w1.notna() & w2.notna()
if weight_ok.any() and not np.allclose(w1[weight_ok], w2[weight_ok], rtol=0, atol=1e-10):
    raise RuntimeError("Fasting-weight mismatch between raw frame and Script 46 G output")

full["HB_THRESHOLD"] = np.where(full["RIAGENDR"] == 1, 13.0, np.where(full["RIAGENDR"] == 2, 12.0, np.nan))
full["A_SCALAR"] = (full["HB_THRESHOLD"] - pd.to_numeric(full["LBXHGB"], errors="coerce")).clip(lower=0)
full["G_SCALAR"] = (pd.to_numeric(full["LBXGH"], errors="coerce") - 5.7).clip(lower=0)

# This is the complete baseline pool from which simulator parameters may be learned.
# Representation learning itself remains outcome-independent because A/G coordinates
# were frozen upstream in Script 46.
required = A_RAW + G_RAW + A_PCS + G_PCS + X3 + [
    "SOMATIC_SCORE", "SURVEY_WT_4YR", "SDMVPSU", "SDMVSTRA"
]
full["SIM_CALIBRATION_COMPLETE"] = (
    full["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & full[required].notna().all(axis=1)
).astype(int)

# Keep the full design frame, not only complete cases. Later scripts must explicitly
# choose their calibration domain; this prevents silent complete-case conditioning.
keep_cols = [
    "SOURCE_ID", "SEQN", "CYCLE", "CALENDAR_MIDYEAR", "SIM_ROLE",
    "ELIGIBLE_ADULT_NONPREG", "SIM_CALIBRATION_COMPLETE",
    "SURVEY_WT_2YR", "SURVEY_WT_4YR", "SDMVPSU", "SDMVSTRA", "STRATUM", "PSU",
] + X3 + [
    "A_SCALAR", "G_SCALAR",
] + A_RAW + G_RAW + A_PCS + G_PCS + PHQ + [
    "PHQ9_TOTAL", "SOMATIC_SCORE", "COGAFF_SUM"
]
keep_cols = list(dict.fromkeys(keep_cols))
out = full[keep_cols].copy()

if not set(out["CYCLE"].astype(str).unique()).issubset(DISC):
    raise RuntimeError("Leakage guard failed after merge")

out.to_parquet(OUT_PARQUET, index=False)

summary_rows = []
for cycle in ["0506", "0708", "ALL"]:
    q = out if cycle == "ALL" else out[out["CYCLE"].eq(cycle)]
    qc = q[q["SIM_CALIBRATION_COMPLETE"].eq(1)]
    summary_rows.append({
        "cycle": cycle,
        "design_n": int(len(q)),
        "adult_nonpreg_n": int(q["ELIGIBLE_ADULT_NONPREG"].sum()),
        "complete_calibration_n": int(len(qc)),
        "weighted_mean_age_complete": float(np.average(qc["RIDAGEYR"], weights=qc["SURVEY_WT_4YR"])) if len(qc) else np.nan,
        "weighted_mean_somatic_complete": float(np.average(qc["SOMATIC_SCORE"], weights=qc["SURVEY_WT_4YR"])) if len(qc) else np.nan,
    })
pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

sha256 = hashlib.sha256(OUT_PARQUET.read_bytes()).hexdigest()
manifest = {
    "script": "51_prepare_simulation_calibration.py",
    "purpose": "Create leakage-protected 2005-08 NHANES calibration pool for synthetic longitudinal simulation.",
    "allowed_cycles": ["0506", "0708"],
    "forbidden_for_calibration": "All NHANES cycles from 2009 onward",
    "chronology": {
        "0506": "CALIBRATION_FIT",
        "0708": "CALIBRATION_CHECK",
        "2009_plus": "RESERVED_EXTERNAL_VALIDATION_OR_STRESS_TEST",
    },
    "representation_source": {
        "A": str(A_FILE.relative_to(ROOT)),
        "G3": str(G_FILE.relative_to(ROOT)),
        "outcome_independent": True,
    },
    "primary_state": {
        "A_raw": A_RAW,
        "G_raw": G_RAW,
        "A_components": A_PCS,
        "G_components": G_PCS,
        "X3": X3,
        "phenotype": "SOMATIC_SCORE",
    },
    "output": str(OUT_PARQUET.relative_to(ROOT)),
    "output_sha256": sha256,
    "important_rule": "No longitudinal transition parameter may be estimated from 2009+ data before external validation.",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"PASS design rows: {len(out):,}")
print(f"PASS complete calibration rows: {int(out['SIM_CALIBRATION_COMPLETE'].sum()):,}")
print(f"PASS 0506 fit rows: {int(((out['CYCLE'] == '0506') & (out['SIM_CALIBRATION_COMPLETE'] == 1)).sum()):,}")
print(f"PASS 0708 check rows: {int(((out['CYCLE'] == '0708') & (out['SIM_CALIBRATION_COMPLETE'] == 1)).sum()):,}")
print(f"PASS saved: {OUT_PARQUET}")
print(f"PASS manifest: {OUT_MANIFEST}")
print(f"PASS SHA256: {sha256}")
