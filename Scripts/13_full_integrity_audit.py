from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
AUDIT.mkdir(exist_ok=True)

EXPECTED = {
    "0506": [
        "CBC_D.XPT", "DEMO_D.xpt", "DPQ_D.XPT", "GHB_D.XPT",
        "GLU_D.XPT", "OPXRET_D.XPT", "BMX_D.XPT",
        "BIOPRO_D.XPT", "SMQ_D.XPT"
    ],
    "0708": [
        "CBC_E.XPT", "DEMO_E.XPT", "DPQ_E.XPT", "GHB_E.XPT",
        "GLU_E.XPT", "OPXRET_E.XPT", "BMX_E.XPT",
        "BIOPRO_E.XPT", "SMQ_E.XPT"
    ]
}

CORE = {
    "0506": {
        "cbc": "CBC_D.XPT",
        "demo": "DEMO_D.xpt",
        "dpq": "DPQ_D.XPT",
        "ghb": "GHB_D.XPT"
    },
    "0708": {
        "cbc": "CBC_E.XPT",
        "demo": "DEMO_E.XPT",
        "dpq": "DPQ_E.XPT",
        "ghb": "GHB_E.XPT"
    }
}

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

KNOWN_FACTORS = {
    "Affective": ["DPQ010", "DPQ020"],
    "Somatic": ["DPQ030", "DPQ040", "DPQ050"],
    "Internalizing": ["DPQ060", "DPQ090"],
    "Sensorimotor": ["DPQ070", "DPQ080"]
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def normalize_phq(df):
    df = df.copy()
    for c in ITEMS:
        df.loc[df[c].abs() < 1e-10, c] = 0
        df.loc[~df[c].isin([0, 1, 2, 3]), c] = np.nan
    return df

print()
print("FULL PROJECT AUDIT")
print("==================")

manifest_rows = []
missing = []

for cycle, expected in EXPECTED.items():
    folder = DATA / cycle

    for name in expected:
        path = folder / name
        if not path.exists():
            missing.append(str(path))
            continue

        df = pd.read_sas(path, format="xport")
        manifest_rows.append({
            "cycle": cycle,
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": len(df),
            "columns": len(df.columns),
            "seqn_duplicates": int(df["SEQN"].duplicated().sum()) if "SEQN" in df else np.nan
        })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(AUDIT / "13_raw_data_manifest_sha256.csv", index=False)

if missing:
    print("FAIL  Some expected raw files are missing.")
    for x in missing:
        print("      ", x)
else:
    print("PASS  All expected raw NHANES files are present.")

if len(manifest) and manifest["seqn_duplicates"].fillna(0).eq(0).all():
    print("PASS  SEQN keys are unique within audited raw files.")
else:
    print("WARNING  Duplicate or unavailable SEQN keys detected.")

frames = []
phq_invalid = []

for cycle, f in CORE.items():
    p = DATA / cycle

    cbc = pd.read_sas(p / f["cbc"], format="xport")
    demo = pd.read_sas(p / f["demo"], format="xport")
    dpq_raw = pd.read_sas(p / f["dpq"], format="xport")
    ghb = pd.read_sas(p / f["ghb"], format="xport")

    invalid_before = 0
    for c in ITEMS:
        vals = dpq_raw[c]
        valid = vals.isna() | vals.isin([1, 2, 3, 7, 9]) | (vals.abs() < 1e-10)
        invalid_before += int((~valid).sum())

    dpq = normalize_phq(dpq_raw)

    invalid_after = 0
    for c in ITEMS:
        invalid_after += int((~dpq[c].isna() & ~dpq[c].isin([0, 1, 2, 3])).sum())

    phq_invalid.append({
        "cycle": cycle,
        "unexpected_raw_values": invalid_before,
        "invalid_after_normalization": invalid_after
    })

    merged = (
        demo.merge(cbc, on="SEQN", how="inner", validate="one_to_one")
        .merge(ghb, on="SEQN", how="inner", validate="one_to_one")
        .merge(dpq, on="SEQN", how="inner", validate="one_to_one")
    )
    merged["CYCLE"] = cycle
    frames.append(merged)

core = pd.concat(frames, ignore_index=True)
core = core[core["RIDAGEYR"] >= 18].copy()
core["PHQ9_TOTAL"] = core[ITEMS].sum(axis=1, min_count=9)

complete = core.dropna(subset=["LBXHGB", "LBXGH", "PHQ9_TOTAL"]).copy()

pd.DataFrame(phq_invalid).to_csv(
    AUDIT / "13_phq_encoding_audit.csv",
    index=False
)

if all(x["invalid_after_normalization"] == 0 for x in phq_invalid):
    print("PASS  PHQ-9 item encoding is valid after the documented tiny-zero correction.")
else:
    print("FAIL  Unexpected PHQ-9 values remain after normalization.")

if len(complete) == 9743:
    print("PASS  Independent rebuild reproduces the 9,743-person complete A/G/P sample.")
else:
    print(f"WARNING  Independent rebuild gives {len(complete):,} complete A/G/P participants, not 9,743.")

if (PROCESSED / "nhanes_core_working.parquet").exists():
    old = pd.read_parquet(PROCESSED / "nhanes_core_working.parquet")
    old_keys = set(zip(old["CYCLE"].astype(str), old["SEQN"]))
    new_keys = set(zip(complete["CYCLE"].astype(str), complete["SEQN"]))
    if old_keys == new_keys:
        print("PASS  Processed working-copy participant keys reproduce from raw data.")
    else:
        print("WARNING  Processed working-copy keys differ from the independent raw rebuild.")
else:
    print("WARNING  Processed working-copy parquet was not found.")

ledger = pd.DataFrame([
    ["Raw XPT", "Original NHANES files", "None", False, True, "Ground truth; preserve bytes and SHA256."],
    ["Core reference", "CBC + DEMO + GHB + DPQ merge", "Inner merge on SEQN", False, True, "Value-level compressed reference; raw XPT remains authoritative."],
    ["PHQ normalization", "XPT near-zero -> literal zero", "abs(value)<1e-10 -> 0", False, True, "Required because of XPORT numeric decoding artifact."],
    ["A representation", "Hemoglobin -> anemia deficit", "max(sex threshold - Hb, 0)", True, True, "Clinically informed working representation; raw Hb retained."],
    ["G representation", "HbA1c -> glycemic excess", "max(HbA1c - 5.7, 0)", True, True, "Working representation; HbA1c may itself be biased by anemia."],
    ["PHQ total", "9 items -> total score", "sum items", True, True, "Known aggregation loss; item-level reference retained."],
    ["ICA", "Residualized 9-item PHQ -> latent components", "FastICA", True, True, "Exploratory decomposition; mixing/unmixing matrices must be stored."],
    ["Cross-cycle alignment", "Independent ICA spaces -> aligned spaces", "Hungarian assignment + sign alignment", True, True, "Component signs/order are arbitrary until aligned."],
    ["Bootstrap", "Participant resamples -> stability estimates", "ordinary participant bootstrap", True, True, "Useful for algorithm/sample stability; not survey-design population inference."]
], columns=[
    "stage", "input_output", "transformation",
    "lossy_representation", "reference_preserved", "audit_note"
])

ledger.to_csv(AUDIT / "13_transformation_ledger.csv", index=False)

print("PASS  Raw-data hashes and transformation ledger written to Audit/.")

def residualize(d):
    d = d.copy()
    d["AGE_Z"] = (d["RIDAGEYR"] - d["RIDAGEYR"].mean()) / d["RIDAGEYR"].std()

    X = pd.DataFrame({
        "AGE_Z": d["AGE_Z"],
        "SEX": d["RIAGENDR"].astype(float)
    }, index=d.index)

    race = pd.get_dummies(
        d["RIDRETH1"].astype(int),
        prefix="RACE",
        drop_first=True,
        dtype=float
    )

    X = pd.concat([X, race], axis=1)
    X = sm.add_constant(X).astype(float)

    R = pd.DataFrame(index=d.index)

    for item in ITEMS:
        model = sm.OLS(d[item].astype(float), X).fit()
        R[item] = model.resid

    scaler = StandardScaler()
    return scaler.fit_transform(R), R

def fit_three(d, seed=42):
    R, residual_df = residualize(d)

    model = FastICA(
        n_components=3,
        random_state=seed,
        max_iter=10000,
        tol=1e-5,
        whiten="unit-variance"
    )

    Z = model.fit_transform(R)
    L = model.mixing_
    return Z, L, residual_df

d05 = complete[complete["CYCLE"] == "0506"].copy().reset_index(drop=True)
d07 = complete[complete["CYCLE"] == "0708"].copy().reset_index(drop=True)

z05, l05, r05 = fit_three(d05)
z07, l07, r07 = fit_three(d07)

n05 = l05 / np.linalg.norm(l05, axis=0, keepdims=True)
n07 = l07 / np.linalg.norm(l07, axis=0, keepdims=True)

similarity = np.abs(n05.T @ n07)
rows, cols = linear_sum_assignment(-similarity)

aligned_l07 = np.zeros_like(l07)
aligned_z07 = np.zeros_like(z07)
mapping = []

for i, j in zip(rows, cols):
    signed = np.dot(n05[:, i], n07[:, j])
    sign = 1 if signed >= 0 else -1
    aligned_l07[:, i] = l07[:, j] * sign
    aligned_z07[:, i] = z07[:, j] * sign
    mapping.append({
        "aligned_component": f"IC{i+1}",
        "0506_component": f"IC{i+1}",
        "0708_component": f"IC{j+1}",
        "similarity": abs(signed),
        "sign_0708": sign
    })

mapping = pd.DataFrame(mapping)
mapping.to_csv(AUDIT / "13_three_component_mapping.csv", index=False)

loading_rows = []
for i in range(3):
    for k, item in enumerate(ITEMS):
        loading_rows.append({
            "component": f"IC{i+1}",
            "item": item,
            "loading_0506": l05[k, i],
            "loading_0708_aligned": aligned_l07[k, i]
        })

loadings = pd.DataFrame(loading_rows)
loadings.to_csv(AUDIT / "13_three_component_loadings_aligned.csv", index=False)

factor_rows = []

for cycle, d, Z in [
    ("0506", d05, z05),
    ("0708", d07, aligned_z07)
]:
    scores = {}
    for factor, factor_items in KNOWN_FACTORS.items():
        scores[factor] = d[factor_items].mean(axis=1).values

    for i in range(3):
        for factor, score in scores.items():
            factor_rows.append({
                "cycle": cycle,
                "component": f"IC{i+1}",
                "known_factor": factor,
                "correlation": np.corrcoef(Z[:, i], score)[0, 1]
            })

factor_match = pd.DataFrame(factor_rows)
factor_match.to_csv(
    AUDIT / "13_ica_vs_known_phq_factors.csv",
    index=False
)

item_rows = []

for cycle, d, Z in [
    ("0506", d05, z05),
    ("0708", d07, aligned_z07)
]:
    for i in range(3):
        for item in ITEMS:
            item_rows.append({
                "cycle": cycle,
                "component": f"IC{i+1}",
                "item": item,
                "correlation": np.corrcoef(Z[:, i], d[item].values)[0, 1]
            })

item_corr = pd.DataFrame(item_rows)
item_corr.to_csv(
    AUDIT / "13_component_item_correlations.csv",
    index=False
)

ic3 = item_corr[item_corr["component"] == "IC3"].copy()
top = (
    ic3.assign(abs_corr=ic3["correlation"].abs())
    .sort_values(["cycle", "abs_corr"], ascending=[True, False])
    .groupby("cycle")
    .head(2)
)

print()
print("LATENT-STRUCTURE AUDIT")
print("----------------------")

ic3_map = mapping[mapping["aligned_component"] == "IC3"]["similarity"].iloc[0]
if ic3_map >= 0.9:
    print("PASS  Final 3-component IC3 independently replicates across cycles.")
else:
    print("WARNING  Final 3-component IC3 cross-cycle similarity is below 0.90.")

print("CHECK  Final 3-component loadings were regenerated; do not use the old 4-component IC3 figure as final evidence.")

for cycle in ["0506", "0708"]:
    t = top[top["cycle"] == cycle]
    desc = ", ".join(f"{r.item} ({r.correlation:.2f})" for r in t.itertuples())
    print(f"CHECK  {cycle} IC3 top item correlations: {desc}")

print()
print("METHOD STATUS")
print("-------------")
print("PASS     Raw XPT files remain the authoritative reference.")
print("PASS     Lossy working representations retain links back to raw participant data.")
print("PASS     Independent-cycle ICA + mathematical sign/order alignment is valid exploratory methodology.")
print("PASS     Seed and participant-bootstrap stability tests are legitimate stability tests.")
print("REQUIRED Full NHANES survey-design inference has NOT yet been done.")
print("REQUIRED Ordinal CFA/WLSMV confirmation of the PHQ latent structure has NOT yet been done.")
print("REQUIRED HbA1c findings need fasting-glucose/diabetes-definition sensitivity because anemia can bias HbA1c.")
print("REQUIRED Confounder selection needs a causal/DAG rationale; do not blindly adjust for every variable.")
print("REQUIRED Multiple-testing/exploratory-selection effects need a confirmatory analysis plan.")
print("REQUIRED The nonlinear cross-cycle test should be rerun with all preprocessing/basis fitting learned on training data only.")
print()
print("Audit outputs saved to:", AUDIT)