from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import requests
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

# ---------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------
G3_VARS = ["LBXGH", "LBXGLU", "LOG_IN"]
G3_LABELS = {
    "LBXGH": "HbA1c",
    "LBXGLU": "Fasting glucose",
    "LOG_IN": "log fasting insulin",
}

G4_VARS = ["LBXGH", "LBXGLU", "LOG_IN", "LBXGLT"]
G4_LABELS = {
    **G3_LABELS,
    "LBXGLT": "2-hour OGTT glucose",
}

DISC = {
    "0506": {"suffix": "D", "year": "2005"},
    "0708": {"suffix": "E", "year": "2007"},
}

TEMP = {
    "0910": {"suffix": "F", "year": "2009"},
    "1112": {"suffix": "G", "year": "2011"},
    "1314": {"suffix": "H", "year": "2013"},
    "1516": {"suffix": "I", "year": "2015"},
    "1718": {"suffix": "J", "year": "2017"},
    "2123": {"suffix": "L", "year": "2021"},
}

OGTT_AVAILABLE = {"0506", "0708", "0910", "1112", "1314", "1516"}

INSULIN_METHOD_ERA = {
    "0506": "Mercodia_ELISA",
    "0708": "Mercodia_ELISA",
    "0910": "Mercodia_to_Roche_transition_NHANES_corrected",
    "1112": "Roche_Elecsys",
    "1314": "Tosoh_AIA",
    "1516": "Tosoh_AIA",
    "1718": "Tosoh_AIA",
    "2123": "Tosoh_AIA",
}

# These are deliberately NOT treated as equivalent raw assay eras.
# The script audits the discontinuity rather than silently harmonizing it.
VAR_TARGETS = [0.80, 0.90, 0.95, 0.99]

print()
print("DEEP GLYCEMIC STRUCTURE + ASSAY AUDIT")
print("=====================================")
print("Core G3: HbA1c + fasting glucose + log fasting insulin")
print("Extended G4: G3 + 2-hour OGTT glucose")
print("No depression outcome is used anywhere in this script.")
print("Insulin assay changes are audited explicitly; no silent calibration is performed.")
print()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return "existing"

    print(f"DOWNLOAD  {path.name}")
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return "downloaded"

def official_url(year, name):
    return f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"

def weighted_mean_scale(X, w):
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X, w = X[ok], w[ok]

    sw = w.sum()
    mu = np.sum(w[:, None] * X, axis=0) / sw
    var = np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw
    sd = np.sqrt(var)

    if np.any(sd <= 0):
        raise ValueError("Zero variance in physiological representation input.")

    return mu, sd

def orient_loadings(loadings, variables):
    L = loadings.copy()

    # Deterministic sign: make largest absolute loading positive.
    for j in range(L.shape[1]):
        anchor = np.argmax(np.abs(L[:, j]))
        if L[anchor, j] < 0:
            L[:, j] *= -1

    # For G3/G4, if PC1 is a broad common glycemic axis but happens to
    # be globally negative, prefer positive sum orientation.
    if np.sum(L[:, 0]) < 0:
        L[:, 0] *= -1

    return L

def weighted_pca(df, variables, weight_col):
    X = df[variables].to_numpy(float)
    w = df[weight_col].to_numpy(float)

    mu, sd = weighted_mean_scale(X, w)
    Z = (X - mu) / sd

    cov = (Z.T * w) @ Z / w.sum()
    cov = (cov + cov.T) / 2.0

    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    L = orient_loadings(vecs[:, order], variables)

    scores = Z @ L
    evr = vals / vals.sum()

    return {
        "mean": mu,
        "scale": sd,
        "loadings": L,
        "eigenvalues": vals,
        "evr": evr,
        "scores": scores,
        "Z": Z,
    }

def align_to_reference(ref, target):
    sim = np.abs(ref.T @ target)
    row, col = linear_sum_assignment(-sim)
    order = col[np.argsort(row)]

    aligned = target[:, order].copy()
    signs = np.sign(np.sum(ref * aligned, axis=0))
    signs[signs == 0] = 1
    aligned *= signs

    similarity = np.abs(np.sum(ref * aligned, axis=0))
    return order, signs, aligned, similarity

def weighted_corr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return np.nan

    sw = w.sum()
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    cov = np.sum(w * (x - mx) * (y - my)) / sw
    vx = np.sum(w * (x - mx) ** 2) / sw
    vy = np.sum(w * (y - my) ** 2) / sw

    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan

def weighted_corr_matrix(df, variables, weight_col):
    out = np.eye(len(variables))
    w = df[weight_col].to_numpy(float)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            r = weighted_corr(
                df[variables[i]].to_numpy(float),
                df[variables[j]].to_numpy(float),
                w,
            )
            out[i, j] = r
            out[j, i] = r
    return out

def weighted_mean_sd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    sw = w.sum()
    m = np.sum(w * x) / sw
    v = np.sum(w * (x - m) ** 2) / sw
    return float(m), float(np.sqrt(v))

def choose_k(evr, target):
    c = np.cumsum(evr)
    idx = np.where(c >= target)[0]
    return int(idx[0] + 1) if len(idx) else len(evr)

def project(df, variables, model):
    X = df[variables].to_numpy(float)
    Z = (X - model["mean"]) / model["scale"]
    S = Z @ model["loadings"]
    return Z, S

def weighted_reconstruction_energy(Z, S, L, w, k):
    Zhat = S[:, :k] @ L[:, :k].T
    e = Z - Zhat
    w = np.asarray(w, float)

    sse = np.sum(w * np.sum(e ** 2, axis=1))
    total = np.sum(w * np.sum(Z ** 2, axis=1))
    return float(1 - sse / total) if total > 0 else np.nan

def file_for(cycle, component):
    if cycle in DISC:
        meta = DISC[cycle]
        base = DATA_DISC / cycle
    else:
        meta = TEMP[cycle]
        base = DATA_TRANSFER / cycle

    suffix = meta["suffix"]
    year = meta["year"]

    # Insulin was bundled in GLU through 2011-12, then split into INS.
    if component == "INS":
        if cycle in {"0506", "0708", "0910", "1112"}:
            name = f"GLU_{suffix}"
        else:
            name = f"INS_{suffix}"
    else:
        name = f"{component}_{suffix}"

    return base / f"{name}.XPT", official_url(year, name)

def read_cycle(cycle, need_ogtt=False):
    ghb_path, ghb_url = file_for(cycle, "GHB")
    glu_path, glu_url = file_for(cycle, "GLU")
    ins_path, ins_url = file_for(cycle, "INS")

    # GHB/GLU should already exist from prior scripts; only download if needed.
    if not ghb_path.exists():
        download_file(ghb_url, ghb_path)
    if not glu_path.exists():
        download_file(glu_url, glu_path)
    if not ins_path.exists():
        download_file(ins_url, ins_path)

    ghb = pd.read_sas(ghb_path, format="xport")
    glu = pd.read_sas(glu_path, format="xport")
    ins = pd.read_sas(ins_path, format="xport")

    for name, frame, vars_ in [
        ("GHB", ghb, ["SEQN", "LBXGH"]),
        ("GLU", glu, ["SEQN", "LBXGLU", "WTSAF2YR"]),
        ("INS", ins, ["SEQN", "LBXIN"]),
    ]:
        missing = [v for v in vars_ if v not in frame.columns]
        if missing:
            raise ValueError(f"{cycle} {name}: missing {missing}")

    d = ghb[["SEQN", "LBXGH"]].merge(
        glu[["SEQN", "LBXGLU", "WTSAF2YR"]],
        on="SEQN", how="inner", validate="one_to_one"
    ).merge(
        ins[["SEQN", "LBXIN"]],
        on="SEQN", how="inner", validate="one_to_one"
    )

    d["CYCLE"] = cycle
    d["INSULIN_METHOD_ERA"] = INSULIN_METHOD_ERA[cycle]

    # Positive insulin required before log-transform.
    d.loc[pd.to_numeric(d["LBXIN"], errors="coerce") <= 0, "LBXIN"] = np.nan
    d["LOG_IN"] = np.log(pd.to_numeric(d["LBXIN"], errors="coerce"))

    if need_ogtt:
        if cycle not in OGTT_AVAILABLE:
            raise ValueError(f"OGTT not available for {cycle}")
        ogtt_path, ogtt_url = file_for(cycle, "OGTT")
        if not ogtt_path.exists():
            download_file(ogtt_url, ogtt_path)

        ogtt = pd.read_sas(ogtt_path, format="xport")
        missing = [v for v in ["SEQN", "LBXGLT", "WTSOG2YR"] if v not in ogtt.columns]
        if missing:
            raise ValueError(f"{cycle} OGTT: missing {missing}")

        d = d.merge(
            ogtt[["SEQN", "LBXGLT", "WTSOG2YR"]],
            on="SEQN", how="inner", validate="one_to_one"
        )

    return d

# ---------------------------------------------------------------------
# Adult cohort IDs: preserve eligibility without conditioning on depression.
# ---------------------------------------------------------------------
disc_cohort = pd.read_parquet(
    PROCESSED / "nhanes_adjustment_cohort_20plus.parquet"
)[["SEQN", "CYCLE"]].drop_duplicates().copy()

disc_cohort["CYCLE"] = (
    disc_cohort["CYCLE"].astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .replace({"506": "0506", "708": "0708"})
)

transfer_cohort = pd.read_parquet(
    PROCESSED / "31_transfer_harmonized_FINAL_PREMODEL.parquet"
)[["SEQN", "CYCLE"]].drop_duplicates().copy()

transfer_cohort["CYCLE"] = (
    transfer_cohort["CYCLE"].astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

# ---------------------------------------------------------------------
# 1) Build G3 discovery data.
# ---------------------------------------------------------------------
disc_g3_frames = []

for cy in ["0506", "0708"]:
    q = read_cycle(cy, need_ogtt=False)
    q = disc_cohort.merge(
        q, on=["SEQN", "CYCLE"], how="inner", validate="one_to_one"
    )
    q = q.dropna(subset=G3_VARS + ["WTSAF2YR"]).copy()
    q = q[q["WTSAF2YR"] > 0].copy()
    q["WT_G3_DISC"] = q["WTSAF2YR"] / 2.0
    disc_g3_frames.append(q)

disc_g3 = pd.concat(disc_g3_frames, ignore_index=True)

# Independent discovery-cycle fits.
g3_models = {}
for cy in ["0506", "0708"]:
    q = disc_g3[disc_g3["CYCLE"] == cy].reset_index(drop=True)
    g3_models[cy] = weighted_pca(q, G3_VARS, "WTSAF2YR")

order_g3, signs_g3, aligned_g3_0708, sim_g3 = align_to_reference(
    g3_models["0506"]["loadings"],
    g3_models["0708"]["loadings"],
)

# Pooled discovery freeze.
g3_frozen = weighted_pca(disc_g3, G3_VARS, "WT_G3_DISC")

# ---------------------------------------------------------------------
# 2) Extended G4 discovery data with OGTT.
# ---------------------------------------------------------------------
disc_g4_frames = []

for cy in ["0506", "0708"]:
    q = read_cycle(cy, need_ogtt=True)
    q = disc_cohort.merge(
        q, on=["SEQN", "CYCLE"], how="inner", validate="one_to_one"
    )
    q = q.dropna(subset=G4_VARS + ["WTSOG2YR"]).copy()
    q = q[q["WTSOG2YR"] > 0].copy()
    q["WT_G4_DISC"] = q["WTSOG2YR"] / 2.0
    disc_g4_frames.append(q)

disc_g4 = pd.concat(disc_g4_frames, ignore_index=True)

g4_models = {}
for cy in ["0506", "0708"]:
    q = disc_g4[disc_g4["CYCLE"] == cy].reset_index(drop=True)
    g4_models[cy] = weighted_pca(q, G4_VARS, "WTSOG2YR")

order_g4, signs_g4, aligned_g4_0708, sim_g4 = align_to_reference(
    g4_models["0506"]["loadings"],
    g4_models["0708"]["loadings"],
)

g4_frozen = weighted_pca(disc_g4, G4_VARS, "WT_G4_DISC")

# ---------------------------------------------------------------------
# 3) Discovery structure tables + physiological anchors.
# ---------------------------------------------------------------------
structure_rows = []
loading_rows = []
anchor_rows = []

for block, variables, labels, data, models, frozen, sim, aligned0708, wcycle, wpool in [
    (
        "G3_core",
        G3_VARS,
        G3_LABELS,
        disc_g3,
        g3_models,
        g3_frozen,
        sim_g3,
        aligned_g3_0708,
        "WTSAF2YR",
        "WT_G3_DISC",
    ),
    (
        "G4_extended",
        G4_VARS,
        G4_LABELS,
        disc_g4,
        g4_models,
        g4_frozen,
        sim_g4,
        aligned_g4_0708,
        "WTSOG2YR",
        "WT_G4_DISC",
    ),
]:
    for cy in ["0506", "0708"]:
        q = data[data["CYCLE"] == cy].copy()
        m = models[cy]
        aligned_evr = (
            m["evr"] if cy == "0506"
            else m["evr"][order_g3 if block == "G3_core" else order_g4]
        )

        row = {
            "block": block,
            "cycle": cy,
            "n": len(q),
        }
        for j in range(len(variables)):
            row[f"pc{j+1}_variance"] = float(aligned_evr[j])
            row[f"pc{j+1}_cross_cycle_similarity"] = float(sim[j])
        structure_rows.append(row)

    for model_name, L in [
        ("0506_cycle_fit", models["0506"]["loadings"]),
        ("0708_cycle_fit_aligned", aligned0708),
        ("FINAL_POOLED_DISCOVERY_FROZEN", frozen["loadings"]),
    ]:
        for j in range(len(variables)):
            for i, v in enumerate(variables):
                loading_rows.append({
                    "block": block,
                    "model": model_name,
                    "component": j + 1,
                    "variable": v,
                    "label": labels[v],
                    "loading": float(L[i, j]),
                })

    # Pooled physiological anchor correlations.
    Zp, Sp = project(data, variables, frozen)
    wp = data[wpool].to_numpy(float)

    homa_ir = (
        pd.to_numeric(data["LBXGLU"], errors="coerce").to_numpy(float)
        * pd.to_numeric(data["LBXIN"], errors="coerce").to_numpy(float)
        / 405.0
    )
    log_homa = np.log(homa_ir)

    for j in range(len(variables)):
        anchor_rows.append({
            "block": block,
            "sample": "POOLED_0506_0708",
            "component": j + 1,
            "anchor": "log_HOMA_IR",
            "weighted_correlation": weighted_corr(Sp[:, j], log_homa, wp),
        })

        if "LBXGLT" in variables:
            anchor_rows.append({
                "block": block,
                "sample": "POOLED_0506_0708",
                "component": j + 1,
                "anchor": "2h_OGTT_glucose",
                "weighted_correlation": weighted_corr(
                    Sp[:, j],
                    data["LBXGLT"].to_numpy(float),
                    wp,
                ),
            })

structure = pd.DataFrame(structure_rows)
loadings = pd.DataFrame(loading_rows)
anchors = pd.DataFrame(anchor_rows)

# ---------------------------------------------------------------------
# 4) Temporal G3 audit: frozen projection + diagnostic within-cycle PCA.
#    The diagnostic refit is NEVER used for downstream scores; it is only
#    used to assess correlation/loading structure independent of assay scale.
# ---------------------------------------------------------------------
temp_g3_frames = []

for cy in TEMP:
    q = read_cycle(cy, need_ogtt=False)
    q = transfer_cohort.merge(
        q, on=["SEQN", "CYCLE"], how="inner", validate="one_to_one"
    )
    q = q.dropna(subset=G3_VARS + ["WTSAF2YR"]).copy()
    q = q[q["WTSAF2YR"] > 0].copy()

    if cy in {"0910", "1112", "1314", "1516", "1718"}:
        q["WT_G3_TEMP"] = q["WTSAF2YR"] / 5.0
        q["TRANSFER_PERIOD"] = "2009-2018"
    else:
        q["WT_G3_TEMP"] = q["WTSAF2YR"]
        q["TRANSFER_PERIOD"] = "2021-2023"

    temp_g3_frames.append(q)

temp_g3 = pd.concat(temp_g3_frames, ignore_index=True)

# Frozen discovery projection.
Zt, St = project(temp_g3, G3_VARS, g3_frozen)
for j in range(3):
    temp_g3[f"G3_FROZEN_PC{j+1}"] = St[:, j]

# Discovery reference correlation matrix.
disc_corr = weighted_corr_matrix(disc_g3, G3_VARS, "WT_G3_DISC")

transfer_rows = []
score_shift_rows = []
corr_rows = []

for label, q in (
    [("2009-2018", temp_g3[temp_g3["TRANSFER_PERIOD"] == "2009-2018"])]
    + [("2021-2023", temp_g3[temp_g3["TRANSFER_PERIOD"] == "2021-2023"])]
    + [(cy, temp_g3[temp_g3["CYCLE"] == cy]) for cy in TEMP]
):
    if len(q) == 0:
        continue

    w = q["WT_G3_TEMP"].to_numpy(float)

    # Frozen score variance share.
    vars_pc = []
    for j in range(3):
        _, sd = weighted_mean_sd(q[f"G3_FROZEN_PC{j+1}"], w)
        vars_pc.append(sd ** 2)
    shares = np.array(vars_pc) / np.sum(vars_pc)

    # Diagnostic refit for structure only.
    diag = weighted_pca(q, G3_VARS, "WT_G3_TEMP")
    _, _, _, diag_sim = align_to_reference(
        g3_frozen["loadings"], diag["loadings"]
    )

    # Correlation matrix stability; robust to affine assay shifts.
    C = weighted_corr_matrix(q, G3_VARS, "WT_G3_TEMP")
    corr_distance = float(np.linalg.norm(C - disc_corr, ord="fro"))

    transfer_rows.append({
        "period": label,
        "n": len(q),
        "insulin_method_era": (
            q["INSULIN_METHOD_ERA"].iloc[0]
            if q["INSULIN_METHOD_ERA"].nunique() == 1
            else "MIXED"
        ),
        "frozen_pc1_variance_share": float(shares[0]),
        "frozen_pc2_variance_share": float(shares[1]),
        "frozen_pc3_variance_share": float(shares[2]),
        "diagnostic_pc1_loading_similarity_to_discovery": float(diag_sim[0]),
        "diagnostic_pc2_loading_similarity_to_discovery": float(diag_sim[1]),
        "diagnostic_pc3_loading_similarity_to_discovery": float(diag_sim[2]),
        "correlation_matrix_frobenius_distance_from_discovery": corr_distance,
    })

    # Raw/frozen-coordinate score shifts.
    for j in range(3):
        m, sd = weighted_mean_sd(q[f"G3_FROZEN_PC{j+1}"], w)
        score_shift_rows.append({
            "period": label,
            "dimension": f"G3_FROZEN_PC{j+1}",
            "weighted_mean": m,
            "weighted_sd": sd,
        })

    # Pairwise correlation table.
    for i in range(len(G3_VARS)):
        for j in range(i + 1, len(G3_VARS)):
            corr_rows.append({
                "period": label,
                "var1": G3_VARS[i],
                "var2": G3_VARS[j],
                "weighted_correlation": C[i, j],
            })

transfer = pd.DataFrame(transfer_rows)
score_shift = pd.DataFrame(score_shift_rows)
pair_corr = pd.DataFrame(corr_rows)

# ---------------------------------------------------------------------
# 5) Extended G4 temporal audit through 2015-16 only.
# ---------------------------------------------------------------------
temp_g4_frames = []

for cy in ["0910", "1112", "1314", "1516"]:
    q = read_cycle(cy, need_ogtt=True)
    q = transfer_cohort.merge(
        q, on=["SEQN", "CYCLE"], how="inner", validate="one_to_one"
    )
    q = q.dropna(subset=G4_VARS + ["WTSOG2YR"]).copy()
    q = q[q["WTSOG2YR"] > 0].copy()
    q["WT_G4_TEMP"] = q["WTSOG2YR"] / 4.0
    temp_g4_frames.append(q)

temp_g4 = pd.concat(temp_g4_frames, ignore_index=True)

Z4t, S4t = project(temp_g4, G4_VARS, g4_frozen)
for j in range(4):
    temp_g4[f"G4_FROZEN_PC{j+1}"] = S4t[:, j]

g4_transfer_rows = []

for cy in ["0910", "1112", "1314", "1516"]:
    q = temp_g4[temp_g4["CYCLE"] == cy].copy()
    if len(q) == 0:
        continue

    diag = weighted_pca(q, G4_VARS, "WTSOG2YR")
    _, _, _, sim = align_to_reference(
        g4_frozen["loadings"], diag["loadings"]
    )

    row = {
        "cycle": cy,
        "n": len(q),
        "insulin_method_era": INSULIN_METHOD_ERA[cy],
    }
    for j in range(4):
        row[f"diagnostic_pc{j+1}_loading_similarity_to_discovery"] = float(sim[j])
    g4_transfer_rows.append(row)

g4_transfer = pd.DataFrame(g4_transfer_rows)

# ---------------------------------------------------------------------
# 6) Compression / dimensionality summary.
# ---------------------------------------------------------------------
dimension_rows = []

for block, frozen in [("G3_core", g3_frozen), ("G4_extended", g4_frozen)]:
    for target in VAR_TARGETS:
        k = choose_k(frozen["evr"], target)
        dimension_rows.append({
            "block": block,
            "variance_target": target,
            "components_needed": k,
            "cumulative_variance": float(np.cumsum(frozen["evr"])[k - 1]),
        })

dimensions = pd.DataFrame(dimension_rows)

# ---------------------------------------------------------------------
# 7) Save exact discovery-frozen parameters. These are frozen as discovery
#    transforms, but the insulin-containing temporal interpretation remains
#    assay-qualified until the audit is reviewed.
# ---------------------------------------------------------------------
def serialize_model(name, variables, labels, model, weight_desc, caveat):
    payload = {
        "transform_name": name,
        "training_period": "NHANES 2005-2008",
        "outcome_used_in_representation_learning": False,
        "variables": variables,
        "labels": labels,
        "weighting": weight_desc,
        "standardization": {
            "mean": {
                v: float(model["mean"][i]) for i, v in enumerate(variables)
            },
            "scale": {
                v: float(model["scale"][i]) for i, v in enumerate(variables)
            },
        },
        "pca": {
            "eigenvalues": [float(x) for x in model["eigenvalues"]],
            "explained_variance_ratio": [float(x) for x in model["evr"]],
            "cumulative_explained_variance": [
                float(x) for x in np.cumsum(model["evr"])
            ],
            "loadings_columns_are_components": model["loadings"].tolist(),
        },
        "temporal_interpretation_caveat": caveat,
    }
    return payload

g3_payload = serialize_model(
    "G3_HbA1c_fasting_glucose_log_insulin_discovery_PCA",
    G3_VARS,
    G3_LABELS,
    g3_frozen,
    "WTSAF2YR / 2 pooled across 2005-2008",
    (
        "Serum insulin assay methods changed across NHANES cycles. "
        "Do not interpret raw temporal score shifts as biological until the "
        "assay-era audit is reviewed. Frozen projection is preserved for audit."
    ),
)

g4_payload = serialize_model(
    "G4_HbA1c_fasting_glucose_log_insulin_OGTT_discovery_PCA",
    G4_VARS,
    G4_LABELS,
    g4_frozen,
    "WTSOG2YR / 2 pooled across 2005-2008",
    (
        "OGTT is unavailable after 2015-2016, so this extended representation "
        "cannot support the 2021-2023 holdout. Insulin assay changes also require "
        "explicit assay-era interpretation."
    ),
)

g3_text = json.dumps(g3_payload, indent=2)
g4_text = json.dumps(g4_payload, indent=2)

g3_param = PARAMETERS / "42_frozen_G3_deep_transform.json"
g4_param = PARAMETERS / "42_frozen_G4_extended_transform.json"
g3_param.write_text(g3_text, encoding="utf-8")
g4_param.write_text(g4_text, encoding="utf-8")

g3_sha = hashlib.sha256(g3_text.encode("utf-8")).hexdigest()
g4_sha = hashlib.sha256(g4_text.encode("utf-8")).hexdigest()

# Save projected samples for later, but do not run phenotype models yet.
disc_g3_out = disc_g3.copy()
_, Sd3 = project(disc_g3_out, G3_VARS, g3_frozen)
for j in range(3):
    disc_g3_out[f"G3_FROZEN_PC{j+1}"] = Sd3[:, j]

disc_g4_out = disc_g4.copy()
_, Sd4 = project(disc_g4_out, G4_VARS, g4_frozen)
for j in range(4):
    disc_g4_out[f"G4_FROZEN_PC{j+1}"] = Sd4[:, j]

disc_g3_out.to_parquet(
    PROCESSED / "42_G3_discovery_frozen_scores.parquet", index=False
)
temp_g3.to_parquet(
    PROCESSED / "42_G3_temporal_frozen_scores.parquet", index=False
)
disc_g4_out.to_parquet(
    PROCESSED / "42_G4_discovery_frozen_scores.parquet", index=False
)
temp_g4.to_parquet(
    PROCESSED / "42_G4_temporal_frozen_scores.parquet", index=False
)

structure.to_csv(RESULTS / "42_G_deep_discovery_structure.csv", index=False)
loadings.to_csv(RESULTS / "42_G_deep_loadings.csv", index=False)
anchors.to_csv(RESULTS / "42_G_deep_physiological_anchors.csv", index=False)
dimensions.to_csv(RESULTS / "42_G_deep_dimensionality.csv", index=False)
transfer.to_csv(RESULTS / "42_G3_temporal_structure_audit.csv", index=False)
score_shift.to_csv(RESULTS / "42_G3_frozen_score_shift.csv", index=False)
pair_corr.to_csv(RESULTS / "42_G3_pairwise_correlations.csv", index=False)
g4_transfer.to_csv(RESULTS / "42_G4_temporal_structure_audit.csv", index=False)

audit = {
    "script": "42_deep_G_structure_and_assay_audit.py",
    "G3_variables": G3_VARS,
    "G4_variables": G4_VARS,
    "insulin_method_eras": INSULIN_METHOD_ERA,
    "discovery_G3_n": int(len(disc_g3)),
    "discovery_G4_n": int(len(disc_g4)),
    "temporal_G3_n": int(len(temp_g3)),
    "temporal_G4_n_2009_2016": int(len(temp_g4)),
    "outcomes_used": False,
    "silent_insulin_assay_harmonization": False,
    "G3_parameter_sha256": g3_sha,
    "G4_parameter_sha256": g4_sha,
}
(AUDIT / "42_deep_G_structure_and_assay_audit.json").write_text(
    json.dumps(audit, indent=2), encoding="utf-8"
)

# ---------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------
print(f"PASS  G3 discovery sample: n={len(disc_g3):,}")
print(f"PASS  G4 extended discovery sample: n={len(disc_g4):,}")
print()

print("G3 DISCOVERY STRUCTURE")
for cy in ["0506", "0708"]:
    row = structure[
        (structure["block"] == "G3_core") & (structure["cycle"] == cy)
    ].iloc[0]
    print(
        f"{cy}: "
        f"PC variance="
        f"{row['pc1_variance']:.4f}, "
        f"{row['pc2_variance']:.4f}, "
        f"{row['pc3_variance']:.4f}"
    )
print(
    "G3 cross-cycle loading similarity: "
    + ", ".join(f"PC{j+1}={sim_g3[j]:.6f}" for j in range(3))
)
print()

print("G4 EXTENDED DISCOVERY STRUCTURE")
for cy in ["0506", "0708"]:
    row = structure[
        (structure["block"] == "G4_extended") & (structure["cycle"] == cy)
    ].iloc[0]
    print(
        f"{cy}: "
        f"PC variance="
        + ", ".join(
            f"{row[f'pc{j+1}_variance']:.4f}" for j in range(4)
        )
    )
print(
    "G4 cross-cycle loading similarity: "
    + ", ".join(f"PC{j+1}={sim_g4[j]:.6f}" for j in range(4))
)
print()

print("DIMENSIONALITY")
print(dimensions.to_string(index=False))
print()

print("PHYSIOLOGICAL ANCHORS")
print(anchors.to_string(index=False))
print()

print("G3 TEMPORAL STRUCTURE / ASSAY AUDIT")
cols = [
    "period", "n", "insulin_method_era",
    "frozen_pc1_variance_share",
    "frozen_pc2_variance_share",
    "frozen_pc3_variance_share",
    "diagnostic_pc1_loading_similarity_to_discovery",
    "diagnostic_pc2_loading_similarity_to_discovery",
    "diagnostic_pc3_loading_similarity_to_discovery",
    "correlation_matrix_frobenius_distance_from_discovery",
]
print(
    transfer[transfer["period"].isin(list(TEMP.keys()))][cols]
    .sort_values("period")
    .to_string(index=False)
)
print()

print("IMPORTANT")
print("=========")
print(
    "Serum insulin methods changed across NHANES eras. "
    "Any discontinuity around 2009-2010 or 2013-2014 may be analytical, "
    "not biological."
)
print(
    "Therefore this script freezes the discovery transforms and audits transfer, "
    "but it does NOT yet declare G3 temporally validated."
)
print(
    "The combined A/G phenotype model should wait until these assay-era results "
    "are inspected."
)
print()
print(f"G3 PARAMETER SHA256  {g3_sha}")
print(f"G4 PARAMETER SHA256  {g4_sha}")
print()
print("Saved:")
print("  Parameters/42_frozen_G3_deep_transform.json")
print("  Parameters/42_frozen_G4_extended_transform.json")
print("  Data/Processed/42_G3_discovery_frozen_scores.parquet")
print("  Data/Processed/42_G3_temporal_frozen_scores.parquet")
print("  Data/Processed/42_G4_discovery_frozen_scores.parquet")
print("  Data/Processed/42_G4_temporal_frozen_scores.parquet")
print("  Results/42_G_deep_discovery_structure.csv")
print("  Results/42_G_deep_loadings.csv")
print("  Results/42_G_deep_physiological_anchors.csv")
print("  Results/42_G_deep_dimensionality.csv")
print("  Results/42_G3_temporal_structure_audit.csv")
print("  Results/42_G3_frozen_score_shift.csv")
print("  Results/42_G3_pairwise_correlations.csv")
print("  Results/42_G4_temporal_structure_audit.csv")
print("  Audit/42_deep_G_structure_and_assay_audit.json")
