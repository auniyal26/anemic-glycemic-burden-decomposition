from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
PARAMETERS = ROOT / "Parameters"

for p in [PROCESSED, RESULTS, AUDIT, PARAMETERS]:
    p.mkdir(parents=True, exist_ok=True)

CYCLES = {
    "0506": {
        "cbc": DATA / "0506" / "CBC_D.XPT",
        "demo": DATA / "0506" / "DEMO_D.xpt",
    },
    "0708": {
        "cbc": DATA / "0708" / "CBC_E.XPT",
        "demo": DATA / "0708" / "DEMO_E.XPT",
    },
}

# Final reduced hematological basis: avoids obvious algebraic redundancy
# in the earlier 7-variable CBC representation.
A_VARS = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
A_LABELS = {
    "LBXHGB": "Hemoglobin",
    "LBXRBCSI": "RBC count",
    "LBXMCVSI": "MCV",
    "LBXRDW": "RDW",
}

VARIANCE_TARGET = 0.99
MIN_CROSS_CYCLE_SIMILARITY = 0.90
MIN_0708_RECONSTRUCTION_ENERGY = 0.98

print()
print("FREEZE REDUCED HEMATOLOGICAL REPRESENTATION")
print("===========================================")
print("Basis: Hb + RBC count + MCV + RDW")
print("Representation learning is outcome-independent.")
print()

for cy, paths in CYCLES.items():
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} file for {cy}: {path}")

cohort_path = PROCESSED / "nhanes_adjustment_cohort_20plus.parquet"
if not cohort_path.exists():
    raise FileNotFoundError(f"Missing {cohort_path}")

cohort = pd.read_parquet(cohort_path).copy()
cohort["CYCLE"] = (
    cohort["CYCLE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .replace({"506": "0506", "708": "0708"})
)
cohort = cohort[cohort["CYCLE"].isin(CYCLES)].copy()

# Keep the physiological representation population aligned to the adult
# analysis cohort, but do not condition on PHQ or other outcomes.
base_ids = cohort[["SEQN", "CYCLE"]].drop_duplicates().copy()

# Carry scalar A only for an outcome-independent proxy comparison if present.
if "A" in cohort.columns:
    base_ids = base_ids.merge(
        cohort[["SEQN", "CYCLE", "A"]].drop_duplicates(),
        on=["SEQN", "CYCLE"],
        how="left",
        validate="one_to_one",
    )

frames = []

for cy, paths in CYCLES.items():
    cbc = pd.read_sas(paths["cbc"], format="xport")
    demo = pd.read_sas(paths["demo"], format="xport")

    missing_cbc = [c for c in ["SEQN"] + A_VARS if c not in cbc.columns]
    missing_demo = [c for c in ["SEQN", "WTMEC2YR"] if c not in demo.columns]
    if missing_cbc or missing_demo:
        raise ValueError(
            f"{cy}: required variables missing. CBC={missing_cbc}; DEMO={missing_demo}"
        )

    x = cbc[["SEQN"] + A_VARS].merge(
        demo[["SEQN", "WTMEC2YR"]],
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )
    x["CYCLE"] = cy
    x = base_ids.merge(
        x,
        on=["SEQN", "CYCLE"],
        how="inner",
        validate="one_to_one",
    )
    x = x.dropna(subset=A_VARS + ["WTMEC2YR"]).copy()
    x = x[x["WTMEC2YR"] > 0].copy()

    # Equal contribution of the two 2-year cycles in the pooled discovery period.
    x["WTMEC4YR_FREEZE"] = x["WTMEC2YR"] / 2.0
    frames.append(x)

d = pd.concat(frames, ignore_index=True)
d["CYCLE"] = d["CYCLE"].astype(str)

cycle_counts = d["CYCLE"].value_counts().sort_index().to_dict()
if set(cycle_counts) != {"0506", "0708"}:
    raise RuntimeError(f"Unexpected cycles after assembly: {cycle_counts}")

def weighted_mean_scale(X, w):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Non-finite values or non-positive weights in PCA input.")

    sw = w.sum()
    mu = (w[:, None] * X).sum(axis=0) / sw
    var = (w[:, None] * (X - mu) ** 2).sum(axis=0) / sw
    scale = np.sqrt(var)

    if np.any(scale <= 0):
        raise ValueError("At least one reduced CBC variable has zero weighted variance.")

    return mu, scale

def weighted_pca_fit(df, variables, weight_col):
    X = df[variables].to_numpy(float)
    w = df[weight_col].to_numpy(float)

    mu, scale = weighted_mean_scale(X, w)
    Z = (X - mu) / scale

    sw = w.sum()
    cov = (Z.T * w) @ Z / sw
    cov = (cov + cov.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    loadings = eigvecs[:, order]

    # Deterministic sign convention: the largest-magnitude loading in each PC
    # is positive. This removes arbitrary PCA sign flips from future projection.
    for j in range(loadings.shape[1]):
        anchor = np.argmax(np.abs(loadings[:, j]))
        if loadings[anchor, j] < 0:
            loadings[:, j] *= -1

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

def project_with_model(df, variables, model):
    X = df[variables].to_numpy(float)
    Z = (X - model["mean"]) / model["scale"]
    scores = Z @ model["loadings"]
    return Z, scores

def align_loadings(ref_loadings, target_loadings):
    sim = np.abs(ref_loadings.T @ target_loadings)
    row, col = linear_sum_assignment(-sim)
    order = col[np.argsort(row)]

    aligned = target_loadings[:, order].copy()
    signs = np.sign(np.sum(ref_loadings * aligned, axis=0))
    signs[signs == 0] = 1
    aligned *= signs

    similarities = np.abs(np.sum(ref_loadings * aligned, axis=0))
    return order, signs, aligned, similarities

def weighted_reconstruction_metrics(Z, scores, loadings, w, k, train_scale=None):
    Zhat = scores[:, :k] @ loadings[:, :k].T
    err = Z - Zhat
    w = np.asarray(w, float)
    sw = w.sum()

    per_subject_sse = np.sum(err ** 2, axis=1)
    per_subject_energy = np.sum(Z ** 2, axis=1)

    sse = np.sum(w * per_subject_sse)
    total_energy = np.sum(w * per_subject_energy)
    energy_fraction = 1.0 - sse / total_energy if total_energy > 0 else np.nan

    rmse_z = np.sqrt(np.sum(w[:, None] * err ** 2, axis=0) / sw)

    out = {
        "reconstruction_energy_fraction": float(energy_fraction),
        "overall_rmse_z": float(np.sqrt(sse / (sw * Z.shape[1]))),
        "rmse_z": rmse_z,
    }

    if train_scale is not None:
        out["rmse_original_units"] = rmse_z * np.asarray(train_scale)

    return out

# ---------------------------------------------------------------------
# 1) Internal structural validation: fit each discovery cycle separately.
# ---------------------------------------------------------------------
models = {}
for cy in ["0506", "0708"]:
    dc = d[d["CYCLE"] == cy].reset_index(drop=True)
    models[cy] = weighted_pca_fit(dc, A_VARS, "WTMEC2YR")

order_0708, signs_0708, aligned_0708, similarities = align_loadings(
    models["0506"]["loadings"],
    models["0708"]["loadings"],
)

validation_rows = []
loading_rows = []

for j in range(len(A_VARS)):
    validation_rows.append({
        "component": j + 1,
        "variance_0506": models["0506"]["explained_variance_ratio"][j],
        "variance_0708_aligned": models["0708"]["explained_variance_ratio"][order_0708[j]],
        "cumulative_variance_0506": models["0506"]["explained_variance_ratio"][:j+1].sum(),
        "cross_cycle_loading_similarity": similarities[j],
        "0708_original_component": int(order_0708[j] + 1),
        "0708_sign_for_alignment": int(signs_0708[j]),
    })

    for i, var in enumerate(A_VARS):
        loading_rows.append({
            "model": "0506_cycle_fit",
            "component": j + 1,
            "variable": var,
            "label": A_LABELS[var],
            "loading": models["0506"]["loadings"][i, j],
        })
        loading_rows.append({
            "model": "0708_cycle_fit_aligned_to_0506",
            "component": j + 1,
            "variable": var,
            "label": A_LABELS[var],
            "loading": aligned_0708[i, j],
        })

validation = pd.DataFrame(validation_rows)

# Determine retained dimensionality using an outcome-independent rule.
k_candidates = np.where(
    models["0506"]["explained_variance_ratio"].cumsum() >= VARIANCE_TARGET
)[0]
if len(k_candidates) == 0:
    K = len(A_VARS)
else:
    K = int(k_candidates[0] + 1)

# Require the retained dimension to also explain >= target in 0708.
while K < len(A_VARS):
    evr_0708_aligned = models["0708"]["explained_variance_ratio"][order_0708]
    if evr_0708_aligned[:K].sum() >= VARIANCE_TARGET:
        break
    K += 1

# ---------------------------------------------------------------------
# 2) Strict internal transfer check: learn ONLY on 0506, project 0708.
# ---------------------------------------------------------------------
reconstruction_rows = []

for target_cycle in ["0506", "0708"]:
    dt = d[d["CYCLE"] == target_cycle].reset_index(drop=True)
    Zt, St = project_with_model(dt, A_VARS, models["0506"])
    met = weighted_reconstruction_metrics(
        Zt,
        St,
        models["0506"]["loadings"],
        dt["WTMEC2YR"].to_numpy(float),
        K,
        train_scale=models["0506"]["scale"],
    )

    row = {
        "training_cycle": "0506",
        "projection_cycle": target_cycle,
        "retained_components": K,
        "n": len(dt),
        "reconstruction_energy_fraction": met["reconstruction_energy_fraction"],
        "overall_rmse_z": met["overall_rmse_z"],
    }
    for i, var in enumerate(A_VARS):
        row[f"{var}_rmse_z"] = met["rmse_z"][i]
        row[f"{var}_rmse_original_units"] = met["rmse_original_units"][i]
    reconstruction_rows.append(row)

reconstruction = pd.DataFrame(reconstruction_rows)

# Structural gate before creating the final pooled frozen transform.
min_similarity_retained = float(similarities[:K].min())
recon_0708 = float(
    reconstruction.loc[
        reconstruction["projection_cycle"] == "0708",
        "reconstruction_energy_fraction"
    ].iloc[0]
)

gate_pass = (
    min_similarity_retained >= MIN_CROSS_CYCLE_SIMILARITY
    and recon_0708 >= MIN_0708_RECONSTRUCTION_ENERGY
)

# ---------------------------------------------------------------------
# 3) Final frozen discovery transform.
# Once the reduced basis and K pass internal structural validation,
# refit using ALL 2005-08 discovery physiology, without PHQ outcomes.
# ---------------------------------------------------------------------
pooled = d.reset_index(drop=True)
final_model = weighted_pca_fit(pooled, A_VARS, "WTMEC4YR_FREEZE")

# Keep the already chosen K. Do not reselect it from outcome data.
final_evr = final_model["explained_variance_ratio"]

for j in range(len(A_VARS)):
    for i, var in enumerate(A_VARS):
        loading_rows.append({
            "model": "FINAL_POOLED_0506_0708_FROZEN",
            "component": j + 1,
            "variable": var,
            "label": A_LABELS[var],
            "loading": final_model["loadings"][i, j],
        })

# Final pooled reconstruction audit.
final_recon = weighted_reconstruction_metrics(
    final_model["Z"],
    final_model["scores"],
    final_model["loadings"],
    pooled["WTMEC4YR_FREEZE"].to_numpy(float),
    K,
    train_scale=final_model["scale"],
)

final_recon_row = {
    "training_cycle": "POOLED_0506_0708_FINAL",
    "projection_cycle": "POOLED_0506_0708",
    "retained_components": K,
    "n": len(pooled),
    "reconstruction_energy_fraction": final_recon["reconstruction_energy_fraction"],
    "overall_rmse_z": final_recon["overall_rmse_z"],
}
for i, var in enumerate(A_VARS):
    final_recon_row[f"{var}_rmse_z"] = final_recon["rmse_z"][i]
    final_recon_row[f"{var}_rmse_original_units"] = final_recon["rmse_original_units"][i]

reconstruction = pd.concat(
    [reconstruction, pd.DataFrame([final_recon_row])],
    ignore_index=True,
)

# Scores from the exact frozen pooled transform.
score_cols = ["SEQN", "CYCLE", "WTMEC2YR", "WTMEC4YR_FREEZE"]
if "A" in pooled.columns:
    score_cols.append("A")

scores_out = pooled[score_cols].copy()
for j in range(len(A_VARS)):
    scores_out[f"A_FROZEN_PC{j+1}"] = final_model["scores"][:, j]
    scores_out[f"A_FROZEN_PC{j+1}_RETAINED"] = int(j < K)

# Outcome-independent correlation with conventional scalar A.
proxy_rows = []
if "A" in scores_out.columns:
    for cy in ["0506", "0708", "POOLED"]:
        if cy == "POOLED":
            sx = scores_out.copy()
            w = sx["WTMEC4YR_FREEZE"].to_numpy(float)
        else:
            sx = scores_out[scores_out["CYCLE"] == cy].copy()
            w = sx["WTMEC2YR"].to_numpy(float)

        y = pd.to_numeric(sx["A"], errors="coerce").to_numpy(float)
        valid_y = np.isfinite(y)

        for j in range(K):
            x = sx[f"A_FROZEN_PC{j+1}"].to_numpy(float)
            valid = valid_y & np.isfinite(x) & np.isfinite(w) & (w > 0)

            if valid.sum() < 10:
                r = np.nan
            else:
                xv = x[valid]
                yv = y[valid]
                wv = w[valid]
                sw = wv.sum()
                mx = np.sum(wv * xv) / sw
                my = np.sum(wv * yv) / sw
                cov = np.sum(wv * (xv - mx) * (yv - my)) / sw
                vx = np.sum(wv * (xv - mx) ** 2) / sw
                vy = np.sum(wv * (yv - my) ** 2) / sw
                r = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan

            proxy_rows.append({
                "cycle": cy,
                "component": j + 1,
                "weighted_corr_with_scalar_A": r,
            })

proxy_corr = pd.DataFrame(proxy_rows)

# ---------------------------------------------------------------------
# 4) Serialize exact transform for later NHANES projection.
# ---------------------------------------------------------------------
param_payload = {
    "transform_name": "reduced_hematology_weighted_pca_frozen_discovery",
    "training_period": "NHANES 2005-2008",
    "training_cycles": ["0506", "0708"],
    "training_population": "20+ adjustment cohort with complete reduced CBC basis",
    "outcome_used_in_representation_learning": False,
    "variables": A_VARS,
    "labels": A_LABELS,
    "weighting": {
        "cycle_specific_validation": "WTMEC2YR",
        "pooled_discovery_training": "WTMEC2YR / 2",
    },
    "standardization": {
        "mean": {
            var: float(final_model["mean"][i])
            for i, var in enumerate(A_VARS)
        },
        "scale": {
            var: float(final_model["scale"][i])
            for i, var in enumerate(A_VARS)
        },
    },
    "pca": {
        "retained_components": K,
        "variance_target": VARIANCE_TARGET,
        "eigenvalues": [float(x) for x in final_model["eigenvalues"]],
        "explained_variance_ratio": [
            float(x) for x in final_model["explained_variance_ratio"]
        ],
        "cumulative_explained_variance": [
            float(x) for x in final_model["explained_variance_ratio"].cumsum()
        ],
        "loadings_columns_are_components": final_model["loadings"].tolist(),
        "sign_rule": (
            "For each component, the variable with largest absolute loading "
            "is oriented positive."
        ),
    },
    "internal_structural_gate": {
        "minimum_retained_cross_cycle_loading_similarity": min_similarity_retained,
        "required_minimum_similarity": MIN_CROSS_CYCLE_SIMILARITY,
        "0506_frozen_to_0708_reconstruction_energy_fraction": recon_0708,
        "required_minimum_0708_reconstruction_energy": MIN_0708_RECONSTRUCTION_ENERGY,
        "passed": bool(gate_pass),
    },
    "future_projection_rule": (
        "Do not refit means, scales, loadings, component count, ordering, or signs "
        "in 2009-2018 or 2021-2023. Standardize with these frozen discovery "
        "means/scales and multiply by these frozen loadings."
    ),
}

param_path = PARAMETERS / "39_frozen_reduced_A_transform.json"
param_text = json.dumps(param_payload, indent=2)
param_path.write_text(param_text, encoding="utf-8")
param_sha256 = hashlib.sha256(param_text.encode("utf-8")).hexdigest()

audit_payload = {
    "script": "39_freeze_reduced_A_representation.py",
    "basis": A_VARS,
    "variance_target": VARIANCE_TARGET,
    "retained_components": K,
    "cycle_complete_n": {str(k): int(v) for k, v in cycle_counts.items()},
    "pooled_complete_n": int(len(pooled)),
    "min_retained_cross_cycle_similarity": min_similarity_retained,
    "0506_to_0708_reconstruction_energy_fraction": recon_0708,
    "final_pooled_reconstruction_energy_fraction": float(
        final_recon["reconstruction_energy_fraction"]
    ),
    "structural_gate_passed": bool(gate_pass),
    "parameter_file": str(param_path.relative_to(ROOT)),
    "parameter_sha256": param_sha256,
}

(AUDIT / "39_freeze_reduced_A_representation.json").write_text(
    json.dumps(audit_payload, indent=2),
    encoding="utf-8",
)

# ---------------------------------------------------------------------
# 5) Save analysis artifacts.
# ---------------------------------------------------------------------
validation.to_csv(
    RESULTS / "39_reduced_A_cross_cycle_structure.csv",
    index=False,
)
pd.DataFrame(loading_rows).to_csv(
    RESULTS / "39_reduced_A_loadings.csv",
    index=False,
)
reconstruction.to_csv(
    RESULTS / "39_reduced_A_reconstruction_audit.csv",
    index=False,
)
proxy_corr.to_csv(
    RESULTS / "39_reduced_A_scalar_proxy_correlations.csv",
    index=False,
)
scores_out.to_parquet(
    PROCESSED / "39_frozen_reduced_A_scores.parquet",
    index=False,
)

# ---------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------
print(f"PASS  Reduced-CBC complete sample: n={len(pooled):,}")
print(
    "PASS  Cycle samples: "
    + ", ".join(f"{k} n={v:,}" for k, v in sorted(cycle_counts.items()))
)
print()

print("CROSS-CYCLE STRUCTURE")
for row in validation.itertuples():
    print(
        f"PC{int(row.component)}: "
        f"variance={row.variance_0506:.4f}/{row.variance_0708_aligned:.4f}, "
        f"similarity={row.cross_cycle_loading_similarity:.6f}"
    )

print()
print(
    f"RETAINED K={K} using >= {VARIANCE_TARGET:.0%} "
    "outcome-independent variance criterion in both discovery cycles."
)
print(
    f"Minimum loading similarity across retained PCs = "
    f"{min_similarity_retained:.6f}"
)
print(
    f"0506-frozen -> 0708 reconstruction energy retained = "
    f"{recon_0708:.6f}"
)
print(
    f"Final pooled frozen transform cumulative variance at K = "
    f"{final_evr[:K].sum():.6f}"
)
print(
    f"Final pooled reconstruction energy retained = "
    f"{final_recon['reconstruction_energy_fraction']:.6f}"
)

if gate_pass:
    print("PASS  Internal structural gate passed.")
    print("PASS  Final 2005-08 reduced hematology transform is FROZEN for temporal projection.")
else:
    print("WARNING  Internal structural gate did not pass.")
    print("DO NOT use this transform for temporal claims until the failure is resolved.")

print()
print(f"PARAMETER SHA256  {param_sha256}")
print()
print("Saved:")
print("  Parameters/39_frozen_reduced_A_transform.json")
print("  Data/Processed/39_frozen_reduced_A_scores.parquet")
print("  Results/39_reduced_A_cross_cycle_structure.csv")
print("  Results/39_reduced_A_loadings.csv")
print("  Results/39_reduced_A_reconstruction_audit.csv")
print("  Results/39_reduced_A_scalar_proxy_correlations.csv")
print("  Audit/39_freeze_reduced_A_representation.json")
