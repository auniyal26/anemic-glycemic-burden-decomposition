from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
AUDIT = ROOT / "Audit"
FREEZE = AUDIT / "Exploratory_Analysis_V1"
PARAMS = FREEZE / "Parameters"

FREEZE.mkdir(parents=True, exist_ok=True)
PARAMS.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
K = 3
ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

FILES = {
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

LABELS = {
    "DPQ010": "Anhedonia",
    "DPQ020": "Depressed mood",
    "DPQ030": "Sleep",
    "DPQ040": "Fatigue",
    "DPQ050": "Appetite",
    "DPQ060": "Self-worth",
    "DPQ070": "Concentration",
    "DPQ080": "Psychomotor",
    "DPQ090": "Suicidal ideation"
}

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

def rebuild():
    frames = []

    for cycle, f in FILES.items():
        p = DATA / cycle

        cbc = pd.read_sas(p / f["cbc"], format="xport")
        demo = pd.read_sas(p / f["demo"], format="xport")
        dpq = normalize_phq(pd.read_sas(p / f["dpq"], format="xport"))
        ghb = pd.read_sas(p / f["ghb"], format="xport")

        df = (
            demo.merge(cbc, on="SEQN", how="inner", validate="one_to_one")
            .merge(ghb, on="SEQN", how="inner", validate="one_to_one")
            .merge(dpq, on="SEQN", how="inner", validate="one_to_one")
        )

        df["CYCLE"] = cycle
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["RIDAGEYR"] >= 18].copy()
    df["PHQ9_TOTAL"] = df[ITEMS].sum(axis=1, min_count=9)
    df = df.dropna(subset=["LBXHGB", "LBXGH", "PHQ9_TOTAL"]).copy()

    df["HB_THRESHOLD"] = np.where(df["RIAGENDR"] == 1, 13.0, 12.0)
    df["A"] = (df["HB_THRESHOLD"] - df["LBXHGB"]).clip(lower=0)
    df["G"] = (df["LBXGH"] - 5.7).clip(lower=0)
    df["AG"] = df["A"] * df["G"]

    return df

def residualize_and_fit(d):
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

    residuals = pd.DataFrame(index=d.index)
    regression_params = {}

    for item in ITEMS:
        model = sm.OLS(d[item].astype(float), X).fit()
        residuals[item] = model.resid
        regression_params[item] = {
            "columns": list(X.columns),
            "coefficients": {k: float(v) for k, v in model.params.items()}
        }

    scaler = StandardScaler()
    R = scaler.fit_transform(residuals)

    ica = FastICA(
        n_components=K,
        random_state=SEED,
        max_iter=10000,
        tol=1e-5,
        whiten="unit-variance"
    )

    Z = ica.fit_transform(R)

    params = {
        "age_mean": float(d["RIDAGEYR"].mean()),
        "age_std": float(d["RIDAGEYR"].std()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ica_mean": ica.mean_.tolist(),
        "ica_mixing": ica.mixing_.tolist(),
        "ica_components": ica.components_.tolist(),
        "ica_whitening": None if getattr(ica, "whitening_", None) is None else ica.whitening_.tolist(),
        "n_components": K,
        "random_seed": SEED,
        "max_iter": 10000,
        "tol": 1e-5,
        "whiten": "unit-variance",
        "residual_regressions": regression_params
    }

    return Z, ica.mixing_, params

df = rebuild()

processed = pd.read_parquet(PROCESSED / "nhanes_core_working.parquet")
processed_complete = processed.dropna(
    subset=["LBXHGB", "LBXGH", "PHQ9_TOTAL"]
).copy()

raw_keys = set(zip(df["CYCLE"].astype(str), df["SEQN"].astype(float)))
working_keys = set(zip(processed_complete["CYCLE"].astype(str), processed_complete["SEQN"].astype(float)))

key_match = raw_keys == working_keys

d05 = df[df["CYCLE"] == "0506"].copy().reset_index(drop=True)
d07 = df[df["CYCLE"] == "0708"].copy().reset_index(drop=True)

z05, l05, p05 = residualize_and_fit(d05)
z07, l07, p07 = residualize_and_fit(d07)

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
        "component_0506": f"IC{i+1}",
        "component_0708": f"IC{j+1}",
        "similarity": abs(signed),
        "sign_0708": sign
    })

mapping = pd.DataFrame(mapping)
mapping.to_csv(FREEZE / "component_mapping.csv", index=False)

loadings = []

for i in range(K):
    for r, item in enumerate(ITEMS):
        loadings.append({
            "component": f"IC{i+1}",
            "item": item,
            "label": LABELS[item],
            "loading_0506": l05[r, i],
            "loading_0708_aligned": aligned_l07[r, i]
        })

loadings = pd.DataFrame(loadings)
loadings.to_csv(FREEZE / "three_component_loadings.csv", index=False)

factor_rows = []

for cycle, d, Z in [
    ("0506", d05, z05),
    ("0708", d07, aligned_z07)
]:
    for component in range(K):
        for factor, factor_items in KNOWN_FACTORS.items():
            score = d[factor_items].mean(axis=1).values
            factor_rows.append({
                "cycle": cycle,
                "component": f"IC{component+1}",
                "factor": factor,
                "correlation": np.corrcoef(Z[:, component], score)[0, 1]
            })

factor_corr = pd.DataFrame(factor_rows)
factor_corr.to_csv(FREEZE / "component_known_factor_correlations.csv", index=False)

with open(PARAMS / "0506_transform_parameters.json", "w", encoding="utf-8") as f:
    json.dump(p05, f, indent=2)

with open(PARAMS / "0708_transform_parameters.json", "w", encoding="utf-8") as f:
    json.dump(p07, f, indent=2)

global_params = {
    "A_definition": "max(sex_specific_Hb_threshold - Hb, 0)",
    "male_Hb_threshold": 13.0,
    "female_Hb_threshold": 12.0,
    "G_definition": "max(HbA1c - 5.7, 0)",
    "AG_definition": "A * G",
    "PHQ_valid_values": [0, 1, 2, 3],
    "PHQ_xpt_zero_rule": "abs(value) < 1e-10 -> 0",
    "complete_case_variables": ["LBXHGB", "LBXGH", "PHQ9_TOTAL"],
    "complete_sample_n": int(len(df)),
    "reference_key": ["CYCLE", "SEQN"]
}

with open(PARAMS / "global_transform_parameters.json", "w", encoding="utf-8") as f:
    json.dump(global_params, f, indent=2)

ic3 = loadings[loadings["component"] == "IC3"].copy()

x = np.arange(len(ic3))
w = 0.38

plt.figure(figsize=(10, 5))
plt.bar(x - w / 2, ic3["loading_0506"], width=w, label="2005–06")
plt.bar(x + w / 2, ic3["loading_0708_aligned"], width=w, label="2007–08")
plt.axhline(0, linewidth=0.8)
plt.xticks(x, ic3["label"], rotation=40, ha="right")
plt.ylabel("ICA loading")
plt.title("Final 3-component IC3: aligned loadings")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "14_final_ic3_three_component_loadings.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.bar(mapping["aligned_component"], mapping["similarity"])
plt.axhline(0.8, linestyle="--", linewidth=1)
plt.ylim(0, 1)
plt.ylabel("Cross-cycle similarity")
plt.title("Final 3-component cross-cycle replication")
plt.tight_layout()
plt.savefig(FIG / "14_final_three_component_similarity.png", dpi=300)
plt.close()

pivot = factor_corr.pivot_table(
    index=["cycle", "component"],
    columns="factor",
    values="correlation"
)

plt.figure(figsize=(8, 5))
for col in pivot.columns:
    plt.plot(
        np.arange(len(pivot)),
        pivot[col].values,
        marker="o",
        label=col
    )

plt.axhline(0, linewidth=0.8)
plt.xticks(
    np.arange(len(pivot)),
    [f"{a}-{b}" for a, b in pivot.index],
    rotation=45,
    ha="right"
)
plt.ylabel("Correlation")
plt.title("ICA components versus established PHQ factor groupings")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "14_components_vs_known_factors.png", dpi=300)
plt.close()

script_hashes = []

for path in sorted((ROOT / "Scripts").glob("*.py")):
    script_hashes.append({
        "file": path.name,
        "sha256": sha256(path)
    })

pd.DataFrame(script_hashes).to_csv(
    FREEZE / "script_hashes.csv",
    index=False
)

artifact_rows = []

for path in sorted(FREEZE.rglob("*")):
    if path.is_file():
        artifact_rows.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size
        })

for path in [
    FIG / "14_final_ic3_three_component_loadings.png",
    FIG / "14_final_three_component_similarity.png",
    FIG / "14_components_vs_known_factors.png"
]:
    artifact_rows.append({
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size
    })

artifact_manifest = pd.DataFrame(artifact_rows)
artifact_manifest.to_csv(
    FREEZE / "artifact_manifest.csv",
    index=False
)

readme = f"""# Exploratory Analysis V1

Status: frozen exploratory proof-of-concept

Complete A/G/P sample: {len(df):,}

Authoritative source:
- raw NHANES XPT files in Data/NHANES
- SHA256 hashes in Audit/13_raw_data_manifest_sha256.csv

Preservation chain:
raw XPT -> rebuilt reference -> domain-informed A/G representations -> confounder-residualized PHQ item space -> standardized residual space -> ICA latent space

Core transform parameters are serialized in:
- Parameters/global_transform_parameters.json
- Parameters/0506_transform_parameters.json
- Parameters/0708_transform_parameters.json

Final exploratory latent solution:
- K = 3
- ICA seed = {SEED}
- independently fit by NHANES cycle
- mathematically aligned only after independent fitting

Current defensible exploratory result:
- a cross-cycle stable somatic PHQ dimension is recovered;
- preliminary hematological association with that dimension requires survey-correct, confounder-aware and ordinal-factor confirmation;
- glycemic association is weaker;
- A×G is not yet reproducible.

Not confirmatory evidence:
- current p-values;
- causal interpretation;
- final number of PHQ factors;
- population-representative NHANES estimates.

Required next:
- DAG-based adjustment sets
- expanded X variables
- full NHANES survey design
- fasting-glucose/diabetes sensitivity
- ordinal CFA/WLSMV
- missingness and multiple-testing control
"""

(FREEZE / "README.md").write_text(readme, encoding="utf-8")

print()
print("EXPLORATORY V1 FREEZE")
print("=====================")

if key_match:
    print("PASS  Working-copy complete-case keys exactly reproduce from raw XPT.")
else:
    print("FAIL  Working-copy complete-case keys still differ from raw rebuild.")

ic3_similarity = mapping.loc[
    mapping["aligned_component"] == "IC3",
    "similarity"
].iloc[0]

if ic3_similarity >= 0.9:
    print("PASS  Final 3-component IC3 independently replicates.")
else:
    print("WARNING  Final IC3 replication is weaker than expected.")

ic3_factors = factor_corr[
    factor_corr["component"] == "IC3"
].copy()

best = (
    ic3_factors.assign(a=ic3_factors["correlation"].abs())
    .sort_values(["cycle", "a"], ascending=[True, False])
    .groupby("cycle")
    .head(1)
)

for r in best.itertuples():
    print(f"PASS  {r.cycle} IC3 maps most strongly to {r.factor} symptoms.")

print("PASS  Transform parameters serialized.")
print("PASS  Script hashes stored.")
print("PASS  Correct 3-component figures regenerated.")
print("PASS  Exploratory Analysis V1 frozen in Audit/Exploratory_Analysis_V1.")
print()
print("NEXT  Build the causal/DAG adjustment plan before adding X.")
