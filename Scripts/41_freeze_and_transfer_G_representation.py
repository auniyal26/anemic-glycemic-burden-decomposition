from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TRANSFER = ROOT / "Data" / "NHANES_Transfer"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
PARAMETERS = ROOT / "Parameters"

for p in [PROCESSED, RESULTS, AUDIT, PARAMETERS]:
    p.mkdir(parents=True, exist_ok=True)

G_VARS = ["LBXGH", "LBXGLU"]
G_LABELS = {
    "LBXGH": "HbA1c",
    "LBXGLU": "Fasting glucose",
}

DISC_CYCLES = {
    "0506": {"ghb": "GHB_D.XPT", "glu": "GLU_D.XPT"},
    "0708": {"ghb": "GHB_E.XPT", "glu": "GLU_E.XPT"},
}

TRANSFER_CYCLES = {
    "0910": "F",
    "1112": "G",
    "1314": "H",
    "1516": "I",
    "1718": "J",
    "2123": "L",
}

print()
print("FREEZE AND TRANSFER GLYCEMIC REPRESENTATION")
print("===========================================")
print("Basis: HbA1c + fasting glucose")
print("Full representation retains both components.")
print("PC1 is oriented as the common glycemic axis.")
print("PC2 is oriented as fasting-glucose-vs-HbA1c discordance.")
print("Representation learning is outcome-independent.")
print()

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def weighted_mean_scale(X, w):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X, w = X[ok], w[ok]

    sw = w.sum()
    mu = (w[:, None] * X).sum(axis=0) / sw
    var = (w[:, None] * (X - mu) ** 2).sum(axis=0) / sw
    scale = np.sqrt(var)

    if np.any(scale <= 0):
        raise ValueError("At least one glycemic variable has zero weighted variance.")

    return mu, scale

def orient_g_loadings(loadings):
    L = loadings.copy()

    # PC1: common glycemic direction; require both HbA1c and glucose positive.
    # With a positive biomarker correlation, the leading eigenvector should
    # already have same-sign entries; flip globally if needed.
    if np.sum(L[:, 0]) < 0:
        L[:, 0] *= -1

    # PC2: orient so fasting glucose loading is positive.
    if L[1, 1] < 0:
        L[:, 1] *= -1

    return L

def weighted_pca_fit(df, weight_col):
    X = df[G_VARS].to_numpy(float)
    w = df[weight_col].to_numpy(float)

    mu, scale = weighted_mean_scale(X, w)
    Z = (X - mu) / scale

    cov = (Z.T * w) @ Z / w.sum()
    cov = (cov + cov.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    loadings = orient_g_loadings(eigvecs[:, order])

    scores = Z @ loadings
    evr = eigvals / eigvals.sum()

    return {
        "mean": mu,
        "scale": scale,
        "loadings": loadings,
        "eigenvalues": eigvals,
        "explained_variance_ratio": evr,
        "scores": scores,
        "Z": Z,
    }

def project(df, model):
    X = df[G_VARS].to_numpy(float)
    Z = (X - model["mean"]) / model["scale"]
    scores = Z @ model["loadings"]
    return Z, scores

def weighted_corr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 2:
        return np.nan

    sw = w.sum()
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    cov = np.sum(w * (x - mx) * (y - my)) / sw
    vx = np.sum(w * (x - mx) ** 2) / sw
    vy = np.sum(w * (y - my) ** 2) / sw
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan

def weighted_var(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan
    m = np.sum(w * x) / np.sum(w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))

def align_loadings(ref, target):
    sim = np.abs(ref.T @ target)
    row, col = linear_sum_assignment(-sim)
    order = col[np.argsort(row)]
    aligned = target[:, order].copy()

    signs = np.sign(np.sum(ref * aligned, axis=0))
    signs[signs == 0] = 1
    aligned *= signs

    similarities = np.abs(np.sum(ref * aligned, axis=0))
    return order, signs, aligned, similarities

def reconstruction_energy(Z, scores, loadings, w, k):
    Zhat = scores[:, :k] @ loadings[:, :k].T
    err = Z - Zhat
    w = np.asarray(w, float)

    sse = np.sum(w * np.sum(err ** 2, axis=1))
    energy = np.sum(w * np.sum(Z ** 2, axis=1))
    retained = 1.0 - sse / energy if energy > 0 else np.nan
    rmse = np.sqrt(sse / (np.sum(w) * Z.shape[1]))

    return float(retained), float(rmse)

# ---------------------------------------------------------------------
# 1) Discovery fasting cohort: 2005-06 and 2007-08.
# ---------------------------------------------------------------------
cohort_path = PROCESSED / "nhanes_adjustment_cohort_20plus.parquet"
if not cohort_path.exists():
    raise FileNotFoundError(cohort_path)

disc_ids = pd.read_parquet(cohort_path).copy()
disc_ids["CYCLE"] = (
    disc_ids["CYCLE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .replace({"506": "0506", "708": "0708"})
)
disc_ids = disc_ids[disc_ids["CYCLE"].isin(DISC_CYCLES)].copy()

keep_ids = ["SEQN", "CYCLE"]
if "G_HBA1C" in disc_ids.columns:
    keep_ids.append("G_HBA1C")
disc_ids = disc_ids[keep_ids].drop_duplicates(["SEQN", "CYCLE"])

disc_frames = []

for cy, files in DISC_CYCLES.items():
    ghb_path = DATA_DISC / cy / files["ghb"]
    glu_path = DATA_DISC / cy / files["glu"]

    if not ghb_path.exists():
        raise FileNotFoundError(ghb_path)
    if not glu_path.exists():
        raise FileNotFoundError(glu_path)

    ghb = pd.read_sas(ghb_path, format="xport")
    glu = pd.read_sas(glu_path, format="xport")

    missing_ghb = [c for c in ["SEQN", "LBXGH"] if c not in ghb.columns]
    missing_glu = [c for c in ["SEQN", "LBXGLU", "WTSAF2YR"] if c not in glu.columns]
    if missing_ghb or missing_glu:
        raise ValueError(
            f"{cy}: missing GHB={missing_ghb}; GLU={missing_glu}"
        )

    x = ghb[["SEQN", "LBXGH"]].merge(
        glu[["SEQN", "LBXGLU", "WTSAF2YR"]],
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )
    x["CYCLE"] = cy

    x = disc_ids.merge(
        x,
        on=["SEQN", "CYCLE"],
        how="inner",
        validate="one_to_one",
    )

    x = x.dropna(subset=G_VARS + ["WTSAF2YR"]).copy()
    x = x[x["WTSAF2YR"] > 0].copy()
    x["WTFAST4YR"] = x["WTSAF2YR"] / 2.0

    # Conventional scalar G, if not already carried in.
    if "G_HBA1C" not in x.columns:
        x["G_HBA1C"] = np.maximum(x["LBXGH"] - 5.7, 0)

    disc_frames.append(x)

disc = pd.concat(disc_frames, ignore_index=True)
disc_counts = disc["CYCLE"].value_counts().sort_index().to_dict()

if set(disc_counts) != {"0506", "0708"}:
    raise RuntimeError(f"Unexpected discovery cycles: {disc_counts}")

# ---------------------------------------------------------------------
# 2) Independent-cycle structural check.
# ---------------------------------------------------------------------
cycle_models = {}
structure_rows = []

for cy in ["0506", "0708"]:
    q = disc[disc["CYCLE"] == cy].reset_index(drop=True)
    m = weighted_pca_fit(q, "WTSAF2YR")
    cycle_models[cy] = m

    r = weighted_corr(
        q["LBXGH"], q["LBXGLU"], q["WTSAF2YR"]
    )

    structure_rows.append({
        "cycle": cy,
        "n": len(q),
        "weighted_corr_hba1c_fasting_glucose": r,
        "pc1_explained_variance": m["explained_variance_ratio"][0],
        "pc2_explained_variance": m["explained_variance_ratio"][1],
    })

order, signs, aligned_0708, similarities = align_loadings(
    cycle_models["0506"]["loadings"],
    cycle_models["0708"]["loadings"],
)

for row in structure_rows:
    row["pc1_cross_cycle_loading_similarity"] = float(similarities[0])
    row["pc2_cross_cycle_loading_similarity"] = float(similarities[1])

structure = pd.DataFrame(structure_rows)

# ---------------------------------------------------------------------
# 3) Final frozen pooled 2005-08 glycemic representation.
#    Keep BOTH components: this is a decomposition, not forced compression.
# ---------------------------------------------------------------------
frozen_model = weighted_pca_fit(disc, "WTFAST4YR")
FULL_K = 2
COMMON_AXIS_K = 1

disc_Z = frozen_model["Z"]
disc_scores = frozen_model["scores"]
disc_w = disc["WTFAST4YR"].to_numpy(float)

disc_full_energy, disc_full_rmse = reconstruction_energy(
    disc_Z, disc_scores, frozen_model["loadings"], disc_w, FULL_K
)
disc_pc1_energy, disc_pc1_rmse = reconstruction_energy(
    disc_Z, disc_scores, frozen_model["loadings"], disc_w, COMMON_AXIS_K
)

disc_score_out = disc[
    ["SEQN", "CYCLE", "WTSAF2YR", "WTFAST4YR", "LBXGH", "LBXGLU", "G_HBA1C"]
].copy()
disc_score_out["G_FROZEN_PC1"] = disc_scores[:, 0]
disc_score_out["G_FROZEN_PC2"] = disc_scores[:, 1]
disc_score_out.to_parquet(
    PROCESSED / "41_frozen_G_discovery_scores.parquet",
    index=False,
)

# ---------------------------------------------------------------------
# 4) Temporal fasting cohort and frozen projection.
# ---------------------------------------------------------------------
transfer_path = PROCESSED / "31_transfer_harmonized_FINAL_PREMODEL.parquet"
if not transfer_path.exists():
    raise FileNotFoundError(transfer_path)

base = pd.read_parquet(transfer_path).copy()
base["CYCLE"] = (
    base["CYCLE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)
base = base[base["CYCLE"].isin(TRANSFER_CYCLES)].copy()

g_frames = []

for cy, suffix in TRANSFER_CYCLES.items():
    ghb_path = DATA_TRANSFER / cy / f"GHB_{suffix}.XPT"
    glu_path = DATA_TRANSFER / cy / f"GLU_{suffix}.XPT"

    if not ghb_path.exists():
        raise FileNotFoundError(ghb_path)
    if not glu_path.exists():
        raise FileNotFoundError(glu_path)

    ghb = pd.read_sas(ghb_path, format="xport")
    glu = pd.read_sas(glu_path, format="xport")

    missing_ghb = [c for c in ["SEQN", "LBXGH"] if c not in ghb.columns]
    missing_glu = [c for c in ["SEQN", "LBXGLU", "WTSAF2YR"] if c not in glu.columns]
    if missing_ghb or missing_glu:
        raise ValueError(
            f"{cy}: missing GHB={missing_ghb}; GLU={missing_glu}"
        )

    g = ghb[["SEQN", "LBXGH"]].merge(
        glu[["SEQN", "LBXGLU", "WTSAF2YR"]],
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )
    g["CYCLE"] = cy
    g_frames.append(g)

g_all = pd.concat(g_frames, ignore_index=True)

# Existing cohort may already contain LBXGH; use the laboratory source above.
transfer_base = base.drop(columns=["LBXGH", "LBXGLU", "WTSAF2YR"], errors="ignore")
temporal = transfer_base.merge(
    g_all,
    on=["SEQN", "CYCLE"],
    how="inner",
    validate="one_to_one",
)

temporal = temporal.dropna(subset=G_VARS + ["WTSAF2YR"]).copy()
temporal = temporal[temporal["WTSAF2YR"] > 0].copy()

temporal["TRANSFER_PERIOD"] = np.where(
    temporal["CYCLE"].isin(["0910", "1112", "1314", "1516", "1718"]),
    "2009-2018",
    "2021-2023",
)

# Most restrictive analytic subsample is fasting glucose, so use fasting weights.
temporal["G_WEIGHT"] = np.where(
    temporal["TRANSFER_PERIOD"] == "2009-2018",
    temporal["WTSAF2YR"] / 5.0,
    temporal["WTSAF2YR"],
)

temporal["G_HBA1C_CHECK"] = np.maximum(temporal["LBXGH"] - 5.7, 0)

temp_Z, temp_scores = project(temporal, frozen_model)
temporal["G_FROZEN_PC1"] = temp_scores[:, 0]
temporal["G_FROZEN_PC2"] = temp_scores[:, 1]

# ---------------------------------------------------------------------
# 5) Temporal representation audit.
# ---------------------------------------------------------------------
transfer_rows = []

groups = [
    ("2005-2008_DISCOVERY", disc, disc_Z, disc_scores, "WTFAST4YR"),
]

for period in ["2009-2018", "2021-2023"]:
    mask = temporal["TRANSFER_PERIOD"].eq(period).to_numpy()
    groups.append(
        (
            period,
            temporal.loc[mask],
            temp_Z[mask],
            temp_scores[mask],
            "G_WEIGHT",
        )
    )

for cy in TRANSFER_CYCLES:
    mask = temporal["CYCLE"].eq(cy).to_numpy()
    groups.append(
        (
            cy,
            temporal.loc[mask],
            temp_Z[mask],
            temp_scores[mask],
            "G_WEIGHT",
        )
    )

for label, q, Zq, Sq, wcol in groups:
    if len(q) == 0:
        continue

    w = q[wcol].to_numpy(float)

    full_energy, full_rmse = reconstruction_energy(
        Zq, Sq, frozen_model["loadings"], w, FULL_K
    )
    pc1_energy, pc1_rmse = reconstruction_energy(
        Zq, Sq, frozen_model["loadings"], w, COMMON_AXIS_K
    )

    r = weighted_corr(q["LBXGH"], q["LBXGLU"], w)

    # Score-space variance share using the frozen axes, not refitted axes.
    v1 = weighted_var(Sq[:, 0], w)
    v2 = weighted_var(Sq[:, 1], w)
    frozen_pc1_variance_share = v1 / (v1 + v2) if (v1 + v2) > 0 else np.nan

    scalar_g = (
        q["G_HBA1C"].to_numpy(float)
        if "G_HBA1C" in q.columns
        else q["G_HBA1C_CHECK"].to_numpy(float)
    )

    transfer_rows.append({
        "period": label,
        "n": len(q),
        "weighted_corr_hba1c_fasting_glucose": r,
        "frozen_pc1_variance_share": frozen_pc1_variance_share,
        "full_2pc_reconstruction_energy": full_energy,
        "full_2pc_rmse_z": full_rmse,
        "pc1_only_reconstruction_energy": pc1_energy,
        "pc1_only_rmse_z": pc1_rmse,
        "corr_scalar_G_with_frozen_PC1": weighted_corr(
            scalar_g, Sq[:, 0], w
        ),
        "corr_scalar_G_with_frozen_PC2": weighted_corr(
            scalar_g, Sq[:, 1], w
        ),
    })

transfer_metrics = pd.DataFrame(transfer_rows)

# ---------------------------------------------------------------------
# 6) Carry projected G scores into a later combined phenotype model.
# ---------------------------------------------------------------------
carry = [
    "SEQN", "CYCLE", "TRANSFER_PERIOD", "WTSAF2YR", "G_WEIGHT",
    "LBXGH", "LBXGLU", "G_HBA1C_CHECK",
    "G_FROZEN_PC1", "G_FROZEN_PC2",
]

carry_candidates = [
    "SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C", "AG_HBA1C",
    "RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN",
    "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021",
    "STRATUM_TRANSFER", "PSU_TRANSFER",
]
carry += [c for c in carry_candidates if c in temporal.columns]
carry = list(dict.fromkeys(carry))

temporal[carry].to_parquet(
    PROCESSED / "41_frozen_G_temporal_projection.parquet",
    index=False,
)

# ---------------------------------------------------------------------
# 7) Serialize frozen G parameters.
# ---------------------------------------------------------------------
param_payload = {
    "transform_name": "glycemia_hba1c_fasting_glucose_weighted_pca_frozen_discovery",
    "training_period": "NHANES 2005-2008 fasting subsample",
    "training_cycles": ["0506", "0708"],
    "outcome_used_in_representation_learning": False,
    "variables": G_VARS,
    "labels": G_LABELS,
    "full_components_retained": 2,
    "interpretive_axes": {
        "PC1": "common glycemic axis",
        "PC2": "fasting-glucose-vs-HbA1c discordance axis",
    },
    "weighting": {
        "discovery_pooled": "WTSAF2YR / 2",
        "2009_2018_pooled": "WTSAF2YR / 5",
        "2021_2023": "WTSAF2YR",
    },
    "standardization": {
        "mean": {
            v: float(frozen_model["mean"][i])
            for i, v in enumerate(G_VARS)
        },
        "scale": {
            v: float(frozen_model["scale"][i])
            for i, v in enumerate(G_VARS)
        },
    },
    "pca": {
        "eigenvalues": [float(x) for x in frozen_model["eigenvalues"]],
        "explained_variance_ratio": [
            float(x) for x in frozen_model["explained_variance_ratio"]
        ],
        "loadings_columns_are_components": frozen_model["loadings"].tolist(),
        "sign_rule": (
            "PC1 oriented with HbA1c and fasting glucose positive; "
            "PC2 oriented with fasting glucose positive."
        ),
    },
    "future_rule": (
        "Do not refit means, scales, loadings, ordering, or signs in temporal data. "
        "Both PCs are retained for information preservation. PC1-only is evaluated "
        "separately as a compression/common-burden sensitivity."
    ),
}

param_path = PARAMETERS / "41_frozen_G_transform.json"
param_text = json.dumps(param_payload, indent=2)
param_path.write_text(param_text, encoding="utf-8")
param_sha256 = hashlib.sha256(param_text.encode("utf-8")).hexdigest()

audit_payload = {
    "script": "41_freeze_and_transfer_G_representation.py",
    "discovery_complete_n": int(len(disc)),
    "discovery_cycle_n": {str(k): int(v) for k, v in disc_counts.items()},
    "temporal_complete_n": int(len(temporal)),
    "temporal_2009_2018_n": int(
        temporal["TRANSFER_PERIOD"].eq("2009-2018").sum()
    ),
    "temporal_2021_2023_n": int(
        temporal["TRANSFER_PERIOD"].eq("2021-2023").sum()
    ),
    "full_representation_components": 2,
    "representation_refit_in_temporal_data": False,
    "parameter_file": str(param_path.relative_to(ROOT)),
    "parameter_sha256": param_sha256,
}

(AUDIT / "41_freeze_and_transfer_G_representation.json").write_text(
    json.dumps(audit_payload, indent=2),
    encoding="utf-8",
)

structure.to_csv(
    RESULTS / "41_G_discovery_cross_cycle_structure.csv",
    index=False,
)

loading_rows = []
for name, model in [
    ("0506_cycle_fit", cycle_models["0506"]),
    ("0708_cycle_fit", cycle_models["0708"]),
    ("FINAL_POOLED_0506_0708_FROZEN", frozen_model),
]:
    L = (
        aligned_0708
        if name == "0708_cycle_fit"
        else model["loadings"]
    )
    for j in range(2):
        for i, v in enumerate(G_VARS):
            loading_rows.append({
                "model": name,
                "component": j + 1,
                "variable": v,
                "label": G_LABELS[v],
                "loading": float(L[i, j]),
            })

pd.DataFrame(loading_rows).to_csv(
    RESULTS / "41_G_frozen_loadings.csv",
    index=False,
)

transfer_metrics.to_csv(
    RESULTS / "41_G_representation_transfer_metrics.csv",
    index=False,
)

print(f"PASS  Discovery fasting sample: n={len(disc):,}")
print(
    "PASS  Discovery cycles: "
    + ", ".join(f"{k} n={v:,}" for k, v in sorted(disc_counts.items()))
)
print()

print("DISCOVERY GLYCEMIC STRUCTURE")
print(structure.to_string(index=False))
print()

print("FROZEN POOLED LOADINGS")
for j in range(2):
    print(
        f"PC{j+1}: "
        f"HbA1c={frozen_model['loadings'][0,j]:+.6f}, "
        f"FastingGlucose={frozen_model['loadings'][1,j]:+.6f}, "
        f"variance={frozen_model['explained_variance_ratio'][j]:.6f}"
    )

print()
print(
    f"PASS  Full 2-PC discovery reconstruction energy = {disc_full_energy:.6f}"
)
print(
    f"INFO  PC1-only discovery reconstruction energy = {disc_pc1_energy:.6f}"
)
print()

print(f"PASS  Temporal fasting projection sample: n={len(temporal):,}")
print(
    f"PASS  2009-2018 fasting n="
    f"{int(temporal['TRANSFER_PERIOD'].eq('2009-2018').sum()):,}"
)
print(
    f"PASS  2021-2023 fasting n="
    f"{int(temporal['TRANSFER_PERIOD'].eq('2021-2023').sum()):,}"
)
print("PASS  Discovery G scaler/loadings used unchanged.")
print("PASS  No temporal G PCA refit performed.")
print()

print("GLYCEMIC REPRESENTATION TRANSFER")
cols = [
    "period", "n",
    "weighted_corr_hba1c_fasting_glucose",
    "frozen_pc1_variance_share",
    "pc1_only_reconstruction_energy",
    "corr_scalar_G_with_frozen_PC1",
    "corr_scalar_G_with_frozen_PC2",
]
print(
    transfer_metrics[
        transfer_metrics["period"].isin(
            ["2005-2008_DISCOVERY", "2009-2018", "2021-2023"]
        )
    ][cols].to_string(index=False)
)

print()
print(f"PARAMETER SHA256  {param_sha256}")
print()
print("Saved:")
print("  Parameters/41_frozen_G_transform.json")
print("  Data/Processed/41_frozen_G_discovery_scores.parquet")
print("  Data/Processed/41_frozen_G_temporal_projection.parquet")
print("  Results/41_G_discovery_cross_cycle_structure.csv")
print("  Results/41_G_frozen_loadings.csv")
print("  Results/41_G_representation_transfer_metrics.csv")
print("  Audit/41_freeze_and_transfer_G_representation.json")
