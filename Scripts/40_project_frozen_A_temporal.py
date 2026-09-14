from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "NHANES_Transfer"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
PARAMETERS = ROOT / "Parameters"

for p in [PROCESSED, RESULTS, AUDIT]:
    p.mkdir(parents=True, exist_ok=True)

PARAM_FILE = PARAMETERS / "39_frozen_reduced_A_transform.json"
TRANSFER_FILE = PROCESSED / "31_transfer_harmonized_FINAL_PREMODEL.parquet"
DISCOVERY_RECON_FILE = RESULTS / "39_reduced_A_reconstruction_audit.csv"

if not PARAM_FILE.exists():
    raise FileNotFoundError(f"Missing frozen parameter file: {PARAM_FILE}")
if not TRANSFER_FILE.exists():
    raise FileNotFoundError(f"Missing transfer cohort: {TRANSFER_FILE}")
if not DISCOVERY_RECON_FILE.exists():
    raise FileNotFoundError(f"Missing discovery reconstruction audit: {DISCOVERY_RECON_FILE}")

with open(PARAM_FILE, "r", encoding="utf-8") as f:
    frozen = json.load(f)

A_VARS = frozen["variables"]
K = int(frozen["pca"]["retained_components"])

mu = np.array(
    [frozen["standardization"]["mean"][v] for v in A_VARS],
    dtype=float,
)
scale = np.array(
    [frozen["standardization"]["scale"][v] for v in A_VARS],
    dtype=float,
)
loadings = np.array(
    frozen["pca"]["loadings_columns_are_components"],
    dtype=float,
)

if loadings.shape != (len(A_VARS), len(A_VARS)):
    raise ValueError(
        f"Unexpected frozen loading matrix shape {loadings.shape}; "
        f"expected {(len(A_VARS), len(A_VARS))}"
    )

# Reuse the same representation-transfer gate that was set in script 39.
RECON_GATE = float(
    frozen["internal_structural_gate"]["required_minimum_0708_reconstruction_energy"]
)

CYCLES = {
    "0910": "F",
    "1112": "G",
    "1314": "H",
    "1516": "I",
    "1718": "J",
    "2123": "L",
}

print()
print("FROZEN HEMATOLOGY TEMPORAL REPRESENTATION TRANSFER")
print("==================================================")
print(f"Frozen basis: {' + '.join(A_VARS)}")
print(f"Frozen retained components: K={K}")
print("No PCA/scaler refitting is allowed in temporal data.")
print()

# ---------------------------------------------------------------------
# 1) Load the already-harmonized temporal cohort.
# ---------------------------------------------------------------------
cohort = pd.read_parquet(TRANSFER_FILE).copy()
cohort["CYCLE"] = (
    cohort["CYCLE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

# ---------------------------------------------------------------------
# 2) Pull the reduced CBC variables from each temporal CBC file.
#    The old scalar transfer cohort only needed Hb, so we add RBC/MCV/RDW here.
# ---------------------------------------------------------------------
cbc_frames = []

for cycle, suffix in CYCLES.items():
    path = DATA / cycle / f"CBC_{suffix}.XPT"
    if not path.exists():
        raise FileNotFoundError(path)

    cbc = pd.read_sas(path, format="xport")
    missing = [c for c in ["SEQN"] + A_VARS if c not in cbc.columns]
    if missing:
        raise ValueError(f"{cycle}: missing reduced-CBC variables: {missing}")

    x = cbc[["SEQN"] + A_VARS].copy()
    x["CYCLE"] = cycle
    cbc_frames.append(x)

cbc_all = pd.concat(cbc_frames, ignore_index=True)

# Avoid duplicate Hb from the premodel cohort.
drop_existing = [v for v in A_VARS if v in cohort.columns]
base = cohort.drop(columns=drop_existing, errors="ignore")

d = base.merge(
    cbc_all,
    on=["SEQN", "CYCLE"],
    how="left",
    validate="one_to_one",
)

# ---------------------------------------------------------------------
# 3) Define transfer-era groups and correct analysis weights.
# ---------------------------------------------------------------------
d["TRANSFER_PERIOD"] = np.where(
    d["CYCLE"].isin(["0910", "1112", "1314", "1516", "1718"]),
    "2009-2018",
    np.where(d["CYCLE"] == "2123", "2021-2023", np.nan),
)

d["REP_WEIGHT"] = np.where(
    d["TRANSFER_PERIOD"] == "2009-2018",
    pd.to_numeric(d["WEIGHT_0918_POOLED"], errors="coerce"),
    pd.to_numeric(d["WEIGHT_2123"], errors="coerce"),
)

required = A_VARS + ["REP_WEIGHT", "TRANSFER_PERIOD"]
rep = d.dropna(subset=required).copy()
rep = rep[rep["REP_WEIGHT"] > 0].copy()

if rep.empty:
    raise RuntimeError("No complete temporal reduced-CBC observations with positive weights.")

# ---------------------------------------------------------------------
# 4) Frozen projection. No temporal means/scales/loadings are learned.
# ---------------------------------------------------------------------
X = rep[A_VARS].to_numpy(float)
Z = (X - mu) / scale
scores = Z @ loadings

for i, v in enumerate(A_VARS):
    rep[f"{v}_DISCOVERY_Z"] = Z[:, i]

for j in range(loadings.shape[1]):
    rep[f"A_FROZEN_PC{j+1}"] = scores[:, j]
    rep[f"A_FROZEN_PC{j+1}_RETAINED"] = int(j < K)

# Reconstruct only from retained components.
Zhat = scores[:, :K] @ loadings[:, :K].T
err = Z - Zhat

for i, v in enumerate(A_VARS):
    rep[f"{v}_RECON_Z"] = Zhat[:, i]
    rep[f"{v}_RESID_Z"] = err[:, i]

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan
    return float(np.sum(w * x) / np.sum(w))

def weighted_var(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan
    m = np.sum(w * x) / np.sum(w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))

def weighted_quantile(x, w, probs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return [np.nan] * len(probs)

    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return [float(np.interp(p, cw, x)) for p in probs]

def representation_metrics(frame, label):
    w = frame["REP_WEIGHT"].to_numpy(float)
    zz = frame[[f"{v}_DISCOVERY_Z" for v in A_VARS]].to_numpy(float)
    ss = frame[[f"A_FROZEN_PC{j+1}" for j in range(loadings.shape[1])]].to_numpy(float)
    zhat = ss[:, :K] @ loadings[:, :K].T
    ee = zz - zhat

    subject_sse = np.sum(ee ** 2, axis=1)
    subject_energy = np.sum(zz ** 2, axis=1)

    sse = np.sum(w * subject_sse)
    energy = np.sum(w * subject_energy)
    retained = 1.0 - sse / energy if energy > 0 else np.nan

    rmse_z = np.sqrt(np.sum(w[:, None] * ee ** 2, axis=0) / np.sum(w))

    out = {
        "period": label,
        "n": int(len(frame)),
        "weight_sum": float(np.sum(w)),
        "retained_components": K,
        "reconstruction_energy_fraction": float(retained),
        "overall_rmse_z": float(np.sqrt(sse / (np.sum(w) * len(A_VARS)))),
        "passes_prespecified_0p98_gate": bool(retained >= RECON_GATE),
    }

    for i, v in enumerate(A_VARS):
        out[f"{v}_rmse_z"] = float(rmse_z[i])
        out[f"{v}_rmse_original_units"] = float(rmse_z[i] * scale[i])

    return out

# ---------------------------------------------------------------------
# 5) Representation transfer metrics by pooled period and individual cycle.
# ---------------------------------------------------------------------
metric_rows = []

for period in ["2009-2018", "2021-2023"]:
    q = rep[rep["TRANSFER_PERIOD"] == period].copy()
    if len(q):
        metric_rows.append(representation_metrics(q, period))

for cycle in CYCLES:
    q = rep[rep["CYCLE"] == cycle].copy()
    if len(q):
        metric_rows.append(representation_metrics(q, cycle))

metrics = pd.DataFrame(metric_rows)

# ---------------------------------------------------------------------
# 6) Distribution shift audit in the frozen discovery coordinate system.
#    These are descriptive; we do not adapt the transform.
# ---------------------------------------------------------------------
shift_rows = []

for label, q in (
    [("2009-2018", rep[rep["TRANSFER_PERIOD"] == "2009-2018"])]
    + [("2021-2023", rep[rep["TRANSFER_PERIOD"] == "2021-2023"])]
    + [(cy, rep[rep["CYCLE"] == cy]) for cy in CYCLES]
):
    if len(q) == 0:
        continue

    w = q["REP_WEIGHT"].to_numpy(float)

    for v in A_VARS:
        x = q[f"{v}_DISCOVERY_Z"].to_numpy(float)
        q05, q50, q95 = weighted_quantile(x, w, [0.05, 0.50, 0.95])
        shift_rows.append({
            "period": label,
            "space": "biomarker_discovery_z",
            "dimension": v,
            "weighted_mean": weighted_mean(x, w),
            "weighted_sd": np.sqrt(weighted_var(x, w)),
            "weighted_q05": q05,
            "weighted_q50": q50,
            "weighted_q95": q95,
        })

    for j in range(K):
        name = f"A_FROZEN_PC{j+1}"
        x = q[name].to_numpy(float)
        q05, q50, q95 = weighted_quantile(x, w, [0.05, 0.50, 0.95])
        shift_rows.append({
            "period": label,
            "space": "frozen_component_score",
            "dimension": name,
            "weighted_mean": weighted_mean(x, w),
            "weighted_sd": np.sqrt(weighted_var(x, w)),
            "weighted_q05": q05,
            "weighted_q50": q50,
            "weighted_q95": q95,
        })

shift = pd.DataFrame(shift_rows)

# ---------------------------------------------------------------------
# 7) Compare temporal reconstruction to the already-frozen discovery audit.
# ---------------------------------------------------------------------
disc_recon = pd.read_csv(DISCOVERY_RECON_FILE)
disc_final = disc_recon[
    disc_recon["training_cycle"] == "POOLED_0506_0708_FINAL"
].copy()

if len(disc_final) != 1:
    raise RuntimeError("Could not uniquely identify final discovery reconstruction row.")

discovery_energy = float(disc_final["reconstruction_energy_fraction"].iloc[0])
discovery_rmse = float(disc_final["overall_rmse_z"].iloc[0])

comparison_rows = [{
    "period": "2005-2008_DISCOVERY",
    "n": int(disc_final["n"].iloc[0]),
    "reconstruction_energy_fraction": discovery_energy,
    "overall_rmse_z": discovery_rmse,
    "absolute_energy_change_vs_discovery": 0.0,
    "rmse_ratio_vs_discovery": 1.0,
}]

for period in ["2009-2018", "2021-2023"]:
    x = metrics[metrics["period"] == period]
    if len(x) != 1:
        continue
    energy = float(x["reconstruction_energy_fraction"].iloc[0])
    rmse = float(x["overall_rmse_z"].iloc[0])

    comparison_rows.append({
        "period": period,
        "n": int(x["n"].iloc[0]),
        "reconstruction_energy_fraction": energy,
        "overall_rmse_z": rmse,
        "absolute_energy_change_vs_discovery": energy - discovery_energy,
        "rmse_ratio_vs_discovery": rmse / discovery_rmse if discovery_rmse > 0 else np.nan,
    })

comparison = pd.DataFrame(comparison_rows)

# ---------------------------------------------------------------------
# 8) Save exact projected scores for phenotype-transfer models.
# ---------------------------------------------------------------------
keep = [
    "SEQN", "CYCLE", "TRANSFER_PERIOD", "REP_WEIGHT",
    "WTMEC2YR", "WEIGHT_0918_POOLED", "WEIGHT_2123",
    "STRATUM_TRANSFER", "PSU_TRANSFER",
]
keep = [c for c in keep if c in rep.columns]

# Carry forward already-harmonized outcomes/exposures/covariates without using
# any of them to define the frozen component representation.
carry_candidates = [
    "SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C", "AG_HBA1C",
    "RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN",
    "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021",
]
keep += [c for c in carry_candidates if c in rep.columns]

keep += A_VARS
keep += [f"{v}_DISCOVERY_Z" for v in A_VARS]
keep += [f"A_FROZEN_PC{j+1}" for j in range(loadings.shape[1])]
keep += [f"{v}_RESID_Z" for v in A_VARS]

projected = rep[keep].copy()
projected.to_parquet(
    PROCESSED / "40_frozen_A_temporal_projection.parquet",
    index=False,
)

metrics.to_csv(
    RESULTS / "40_frozen_A_representation_transfer_metrics.csv",
    index=False,
)
shift.to_csv(
    RESULTS / "40_frozen_A_distribution_shift.csv",
    index=False,
)
comparison.to_csv(
    RESULTS / "40_frozen_A_discovery_vs_temporal_reconstruction.csv",
    index=False,
)

param_sha256 = hashlib.sha256(PARAM_FILE.read_bytes()).hexdigest()

audit = {
    "script": "40_project_frozen_A_temporal.py",
    "frozen_parameter_file": str(PARAM_FILE.relative_to(ROOT)),
    "frozen_parameter_sha256": param_sha256,
    "variables": A_VARS,
    "retained_components": K,
    "representation_refit_in_temporal_data": False,
    "temporal_periods": {
        "2009-2018": ["0910", "1112", "1314", "1516", "1718"],
        "2021-2023": ["2123"],
    },
    "weights": {
        "2009-2018": "WEIGHT_0918_POOLED = WTMEC2YR / 5",
        "2021-2023": "WEIGHT_2123 = WTPH2YR",
    },
    "prespecified_reconstruction_energy_gate": RECON_GATE,
    "outcomes_used_to_fit_or_modify_representation": False,
}

(AUDIT / "40_project_frozen_A_temporal.json").write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8",
)

# ---------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------
print(f"PASS  Temporal reduced-CBC projection sample: n={len(rep):,}")
print(
    "PASS  2009-2018 complete reduced-CBC n="
    f"{int((rep['TRANSFER_PERIOD'] == '2009-2018').sum()):,}"
)
print(
    "PASS  2021-2023 complete reduced-CBC n="
    f"{int((rep['TRANSFER_PERIOD'] == '2021-2023').sum()):,}"
)
print("PASS  Discovery scaler/loadings/K used unchanged.")
print("PASS  No temporal PCA refit performed.")
print()

print("REPRESENTATION TRANSFER")
for period in ["2009-2018", "2021-2023"]:
    x = metrics[metrics["period"] == period]
    if len(x):
        row = x.iloc[0]
        status = "PASS" if bool(row["passes_prespecified_0p98_gate"]) else "FAIL"
        print(
            f"{period}: n={int(row['n']):,}, "
            f"reconstruction energy={row['reconstruction_energy_fraction']:.6f}, "
            f"overall RMSE(z)={row['overall_rmse_z']:.6f}, "
            f"gate={status}"
        )

print()
print("DISCOVERY VS TEMPORAL RECONSTRUCTION")
print(comparison.to_string(index=False))

print()
print("TEMPORAL CYCLE DETAIL")
cols = [
    "period", "n", "reconstruction_energy_fraction",
    "overall_rmse_z", "passes_prespecified_0p98_gate"
]
print(
    metrics[metrics["period"].isin(CYCLES.keys())][cols]
    .sort_values("period")
    .to_string(index=False)
)

print()
print("Saved:")
print("  Data/Processed/40_frozen_A_temporal_projection.parquet")
print("  Results/40_frozen_A_representation_transfer_metrics.csv")
print("  Results/40_frozen_A_distribution_shift.csv")
print("  Results/40_frozen_A_discovery_vs_temporal_reconstruction.csv")
print("  Audit/40_project_frozen_A_temporal.json")
