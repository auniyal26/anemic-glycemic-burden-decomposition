from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
PARAMETERS = ROOT / "Parameters"

for p in [PROCESSED, RESULTS, AUDIT, PARAMETERS]:
    p.mkdir(parents=True, exist_ok=True)

DISC = {
    "0506": {"suffix": "D"},
    "0708": {"suffix": "E"},
}
TEMP = {
    "0910": {"suffix": "F"},
    "1112": {"suffix": "G"},
    "1314": {"suffix": "H"},
    "1516": {"suffix": "I"},
    "1718": {"suffix": "J"},
    "2123": {"suffix": "L"},
}

A_VARS = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G_VARS = ["LBXGH", "LBXGLU", "LOG_IN"]
A_VARIANCE_TARGET = 0.99
A_RECON_GATE = 0.98

print()
print("OUTCOME-INDEPENDENT REPRESENTATION REFREEZE")
print("===========================================")
print("Eligibility is built directly from raw DEMO + laboratory files.")
print("No PHQ/depression variable is read anywhere in this script.")
print("Existing 39/42 outputs are not modified.")
print()


def xpt_path(base: Path, stem: str) -> Path:
    """Resolve .XPT/.xpt case safely without downloading or overwriting."""
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")


def raw_base(cycle: str) -> Path:
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def suffix(cycle: str) -> str:
    return (DISC if cycle in DISC else TEMP)[cycle]["suffix"]


def adult_eligibility(cycle: str) -> pd.DataFrame:
    s = suffix(cycle)
    demo = pd.read_sas(xpt_path(raw_base(cycle), f"DEMO_{s}"), format="xport")
    required = ["SEQN", "RIDAGEYR", "RIAGENDR"]
    missing = [c for c in required if c not in demo.columns]
    if missing:
        raise ValueError(f"{cycle} DEMO missing {missing}")

    cols = required + [c for c in ["RIDEXPRG", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"] if c in demo.columns]
    d = demo[cols].copy()
    d = d[pd.to_numeric(d["RIDAGEYR"], errors="coerce") >= 20].copy()
    if "RIDEXPRG" in d.columns:
        d = d[d["RIDEXPRG"] != 1].copy()
    d["CYCLE"] = cycle
    return d


def insulin_file(cycle: str) -> Path:
    s = suffix(cycle)
    # NHANES insulin is in GLU through 2011-12, then in INS.
    stem = f"GLU_{s}" if cycle in {"0506", "0708", "0910", "1112"} else f"INS_{s}"
    return xpt_path(raw_base(cycle), stem)


def build_A_cycle(cycle: str) -> pd.DataFrame:
    s = suffix(cycle)
    elig = adult_eligibility(cycle)
    cbc = pd.read_sas(xpt_path(raw_base(cycle), f"CBC_{s}"), format="xport")
    needed = ["SEQN"] + A_VARS
    missing = [c for c in needed if c not in cbc.columns]
    if missing:
        raise ValueError(f"{cycle} CBC missing {missing}")

    # 2021-23 blood-analyte inference uses phlebotomy weight WTPH2YR.
    extra = ["WTPH2YR"] if "WTPH2YR" in cbc.columns else []
    d = elig.merge(cbc[needed + extra], on="SEQN", how="inner", validate="one_to_one")
    weight_col = "WTPH2YR" if cycle == "2123" else "WTMEC2YR"
    if weight_col not in d.columns:
        raise ValueError(f"{cycle} A cohort missing required weight {weight_col}")

    d = d.dropna(subset=A_VARS + [weight_col]).copy()
    d = d[pd.to_numeric(d[weight_col], errors="coerce") > 0].copy()
    if cycle in DISC:
        d["A_REP_WEIGHT"] = pd.to_numeric(d[weight_col], errors="coerce") / 2.0
    elif cycle == "2123":
        d["A_REP_WEIGHT"] = pd.to_numeric(d[weight_col], errors="coerce")
    else:
        d["A_REP_WEIGHT"] = pd.to_numeric(d[weight_col], errors="coerce") / 5.0
    return d


def build_G_cycle(cycle: str) -> pd.DataFrame:
    s = suffix(cycle)
    elig = adult_eligibility(cycle)
    ghb = pd.read_sas(xpt_path(raw_base(cycle), f"GHB_{s}"), format="xport")
    glu = pd.read_sas(xpt_path(raw_base(cycle), f"GLU_{s}"), format="xport")
    ins = pd.read_sas(insulin_file(cycle), format="xport")

    for name, frame, cols in [
        ("GHB", ghb, ["SEQN", "LBXGH"]),
        ("GLU", glu, ["SEQN", "LBXGLU", "WTSAF2YR"]),
        ("INS", ins, ["SEQN", "LBXIN"]),
    ]:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{cycle} {name} missing {missing}")

    g = ghb[["SEQN", "LBXGH"]].merge(
        glu[["SEQN", "LBXGLU", "WTSAF2YR"]], on="SEQN", how="inner", validate="one_to_one"
    ).merge(
        ins[["SEQN", "LBXIN"]], on="SEQN", how="inner", validate="one_to_one"
    )
    d = elig.merge(g, on="SEQN", how="inner", validate="one_to_one")
    d.loc[pd.to_numeric(d["LBXIN"], errors="coerce") <= 0, "LBXIN"] = np.nan
    d["LOG_IN"] = np.log(pd.to_numeric(d["LBXIN"], errors="coerce"))
    d = d.dropna(subset=G_VARS + ["WTSAF2YR"]).copy()
    d = d[pd.to_numeric(d["WTSAF2YR"], errors="coerce") > 0].copy()
    d["G_REP_WEIGHT"] = pd.to_numeric(d["WTSAF2YR"], errors="coerce")
    if cycle in DISC:
        d["G_REP_WEIGHT"] = d["G_REP_WEIGHT"] / 2.0
    elif cycle != "2123":
        d["G_REP_WEIGHT"] = d["G_REP_WEIGHT"] / 5.0
    return d


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
        raise ValueError("Zero variance in PCA input")
    return mu, sd


def orient_loadings(L, prefer_positive_pc1_sum=False):
    L = L.copy()
    for j in range(L.shape[1]):
        anchor = np.argmax(np.abs(L[:, j]))
        if L[anchor, j] < 0:
            L[:, j] *= -1
    if prefer_positive_pc1_sum and np.sum(L[:, 0]) < 0:
        L[:, 0] *= -1
    return L


def weighted_pca(df, variables, weight_col, prefer_positive_pc1_sum=False):
    X = df[variables].to_numpy(float)
    w = df[weight_col].to_numpy(float)
    mu, sd = weighted_mean_scale(X, w)
    Z = (X - mu) / sd
    cov = (Z.T * w) @ Z / w.sum()
    cov = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    L = orient_loadings(vecs[:, order], prefer_positive_pc1_sum)
    S = Z @ L
    return {"mean": mu, "scale": sd, "loadings": L, "eigenvalues": vals, "evr": vals / vals.sum(), "scores": S, "Z": Z}


def align(ref, target):
    sim = np.abs(ref.T @ target)
    row, col = linear_sum_assignment(-sim)
    order = col[np.argsort(row)]
    aligned = target[:, order].copy()
    signs = np.sign(np.sum(ref * aligned, axis=0))
    signs[signs == 0] = 1
    aligned *= signs
    similarity = np.abs(np.sum(ref * aligned, axis=0))
    return order, signs, aligned, similarity


def project(df, variables, model):
    X = df[variables].to_numpy(float)
    Z = (X - model["mean"]) / model["scale"]
    S = Z @ model["loadings"]
    return Z, S


def reconstruction_energy(Z, S, L, w, k):
    Zhat = S[:, :k] @ L[:, :k].T
    err = Z - Zhat
    w = np.asarray(w, float)
    sse = np.sum(w * np.sum(err ** 2, axis=1))
    total = np.sum(w * np.sum(Z ** 2, axis=1))
    return float(1 - sse / total) if total > 0 else np.nan


def model_json(model, variables, retained_components, notes):
    return {
        "variables": variables,
        "eligibility": "age>=20; exclude known pregnancy; laboratory/weight completeness only; no PHQ conditioning",
        "standardization": {
            "mean": {v: float(model["mean"][i]) for i, v in enumerate(variables)},
            "scale": {v: float(model["scale"][i]) for i, v in enumerate(variables)},
        },
        "pca": {
            "eigenvalues": [float(x) for x in model["eigenvalues"]],
            "explained_variance_ratio": [float(x) for x in model["evr"]],
            "loadings_columns_are_components": model["loadings"].tolist(),
            "retained_components": int(retained_components),
        },
        "notes": notes,
    }


# ------------------------------------------------------------------
# Discovery refreeze: A
# ------------------------------------------------------------------
a_disc_parts = {cy: build_A_cycle(cy) for cy in DISC}
a_cycle_models = {cy: weighted_pca(df, A_VARS, "A_REP_WEIGHT") for cy, df in a_disc_parts.items()}
order_a, signs_a, aligned_a, sim_a = align(a_cycle_models["0506"]["loadings"], a_cycle_models["0708"]["loadings"])

evr_05 = a_cycle_models["0506"]["evr"]
evr_07 = a_cycle_models["0708"]["evr"][order_a]
K_A = len(A_VARS)
for k in range(1, len(A_VARS) + 1):
    if evr_05[:k].sum() >= A_VARIANCE_TARGET and evr_07[:k].sum() >= A_VARIANCE_TARGET:
        K_A = k
        break

# Strict 0506 -> 0708 structural gate before pooled freeze.
Z07, S07 = project(a_disc_parts["0708"], A_VARS, a_cycle_models["0506"])
a_0506_to_0708_energy = reconstruction_energy(
    Z07, S07, a_cycle_models["0506"]["loadings"], a_disc_parts["0708"]["A_REP_WEIGHT"], K_A
)
if a_0506_to_0708_energy < A_RECON_GATE:
    raise RuntimeError(
        f"A structural gate failed: 0506->0708 energy {a_0506_to_0708_energy:.6f} < {A_RECON_GATE}"
    )

a_disc = pd.concat(a_disc_parts.values(), ignore_index=True)
a_frozen = weighted_pca(a_disc, A_VARS, "A_REP_WEIGHT")
ZA, SA = project(a_disc, A_VARS, a_frozen)
for j in range(len(A_VARS)):
    a_disc[f"A_OI_PC{j+1}"] = SA[:, j]

a_disc_out = a_disc[["SEQN", "CYCLE", "A_REP_WEIGHT"] + A_VARS + [f"A_OI_PC{i}" for i in range(1, 5)]].copy()
a_disc_out.to_parquet(PROCESSED / "46_outcome_independent_A_discovery_scores.parquet", index=False)

A_PARAM = PARAMETERS / "46_outcome_independent_A_transform.json"
A_PARAM.write_text(json.dumps(model_json(
    a_frozen, A_VARS, K_A,
    {
        "variance_target": A_VARIANCE_TARGET,
        "strict_0506_to_0708_reconstruction_gate": A_RECON_GATE,
        "strict_0506_to_0708_reconstruction_energy": a_0506_to_0708_energy,
        "representation_learning_uses_depression": False,
    },
), indent=2), encoding="utf-8")

# ------------------------------------------------------------------
# Discovery refreeze: G3
# ------------------------------------------------------------------
g_disc_parts = {cy: build_G_cycle(cy) for cy in DISC}
g_cycle_models = {cy: weighted_pca(df, G_VARS, "G_REP_WEIGHT", True) for cy, df in g_disc_parts.items()}
order_g, signs_g, aligned_g, sim_g = align(g_cycle_models["0506"]["loadings"], g_cycle_models["0708"]["loadings"])
g_disc = pd.concat(g_disc_parts.values(), ignore_index=True)
g_frozen = weighted_pca(g_disc, G_VARS, "G_REP_WEIGHT", True)
ZG, SG = project(g_disc, G_VARS, g_frozen)
for j in range(3):
    g_disc[f"G3_OI_PC{j+1}"] = SG[:, j]

g_disc_out = g_disc[["SEQN", "CYCLE", "G_REP_WEIGHT", "WTSAF2YR", "LBXGH", "LBXGLU", "LBXIN", "LOG_IN"] + [f"G3_OI_PC{i}" for i in range(1, 4)]].copy()
g_disc_out.to_parquet(PROCESSED / "46_outcome_independent_G3_discovery_scores.parquet", index=False)

G_PARAM = PARAMETERS / "46_outcome_independent_G3_transform.json"
G_PARAM.write_text(json.dumps(model_json(
    g_frozen, G_VARS, 3,
    {
        "all_three_coordinates_retained_for_phenotype_mapping": True,
        "two_component_reconstruction_energy_discovery": reconstruction_energy(ZG, SG, g_frozen["loadings"], g_disc["G_REP_WEIGHT"], 2),
        "representation_learning_uses_depression": False,
        "insulin_assay_era_requires_temporal_caution": True,
    },
), indent=2), encoding="utf-8")

# ------------------------------------------------------------------
# Frozen temporal projections from the NEW outcome-independent transforms
# ------------------------------------------------------------------
a_temp_parts = []
g_temp_parts = []
for cy in TEMP:
    ad = build_A_cycle(cy)
    Z, S = project(ad, A_VARS, a_frozen)
    for j in range(4):
        ad[f"A_OI_PC{j+1}"] = S[:, j]
    ad["TRANSFER_PERIOD"] = "2021-2023" if cy == "2123" else "2009-2018"
    a_temp_parts.append(ad[["SEQN", "CYCLE", "TRANSFER_PERIOD", "A_REP_WEIGHT"] + A_VARS + [f"A_OI_PC{i}" for i in range(1, 5)]])

    gd = build_G_cycle(cy)
    Z, S = project(gd, G_VARS, g_frozen)
    for j in range(3):
        gd[f"G3_OI_PC{j+1}"] = S[:, j]
    gd["TRANSFER_PERIOD"] = "2021-2023" if cy == "2123" else "2009-2018"
    g_temp_parts.append(gd[["SEQN", "CYCLE", "TRANSFER_PERIOD", "G_REP_WEIGHT", "WTSAF2YR", "LBXGH", "LBXGLU", "LBXIN", "LOG_IN"] + [f"G3_OI_PC{i}" for i in range(1, 4)]])

a_temp = pd.concat(a_temp_parts, ignore_index=True)
g_temp = pd.concat(g_temp_parts, ignore_index=True)
a_temp.to_parquet(PROCESSED / "46_outcome_independent_A_temporal_scores.parquet", index=False)
g_temp.to_parquet(PROCESSED / "46_outcome_independent_G3_temporal_scores.parquet", index=False)

# ------------------------------------------------------------------
# Structural audit tables
# ------------------------------------------------------------------
rows = []
for block, variables, sim, cycle_models, order, frozen, disc, weight_col in [
    ("A", A_VARS, sim_a, a_cycle_models, order_a, a_frozen, a_disc, "A_REP_WEIGHT"),
    ("G3", G_VARS, sim_g, g_cycle_models, order_g, g_frozen, g_disc, "G_REP_WEIGHT"),
]:
    for j in range(len(variables)):
        rows.append({
            "block": block,
            "component": j + 1,
            "variance_0506": float(cycle_models["0506"]["evr"][j]),
            "variance_0708_aligned": float(cycle_models["0708"]["evr"][order[j]]),
            "cross_cycle_loading_similarity": float(sim[j]),
            "final_pooled_variance": float(frozen["evr"][j]),
        })

pd.DataFrame(rows).to_csv(RESULTS / "46_outcome_independent_structure_audit.csv", index=False)

# Temporal reconstruction audit for A; G full 3-PC reconstruction is identity and is not promoted.
recon_rows = []
for label, q in [("DISCOVERY_2005-2008", a_disc)] + [
    ("2009-2018", a_temp[a_temp["TRANSFER_PERIOD"] == "2009-2018"]),
    ("2021-2023", a_temp[a_temp["TRANSFER_PERIOD"] == "2021-2023"]),
]:
    if label == "DISCOVERY_2005-2008":
        weight_col = "A_REP_WEIGHT"
    else:
        weight_col = "A_REP_WEIGHT"
    Z, S = project(q, A_VARS, a_frozen)
    recon_rows.append({
        "period": label,
        "n": len(q),
        "retained_components": K_A,
        "reconstruction_energy_fraction": reconstruction_energy(Z, S, a_frozen["loadings"], q[weight_col], K_A),
    })
pd.DataFrame(recon_rows).to_csv(RESULTS / "46_outcome_independent_A_reconstruction_transfer.csv", index=False)

# Cohort counts make the PHQ-independence explicit.
count_rows = []
for cy, q in a_disc_parts.items():
    count_rows.append({"block": "A", "cycle": cy, "n": len(q), "source": "raw eligibility; no PHQ filter"})
for cy, q in g_disc_parts.items():
    count_rows.append({"block": "G3", "cycle": cy, "n": len(q), "source": "raw eligibility; no PHQ filter"})
for cy in TEMP:
    count_rows.append({"block": "A", "cycle": cy, "n": int((a_temp["CYCLE"] == cy).sum()), "source": "raw eligibility; no PHQ filter"})
    count_rows.append({"block": "G3", "cycle": cy, "n": int((g_temp["CYCLE"] == cy).sum()), "source": "raw eligibility; no PHQ filter"})
pd.DataFrame(count_rows).to_csv(RESULTS / "46_outcome_independent_cohort_counts.csv", index=False)

# Hashes freeze the correction without touching old parameter files.
a_sha = hashlib.sha256(A_PARAM.read_bytes()).hexdigest()
g_sha = hashlib.sha256(G_PARAM.read_bytes()).hexdigest()
audit = {
    "script": "46_refreeze_outcome_independent_representations.py",
    "purpose": "Remove outcome-availability conditioning from representation learning.",
    "reads_PHQ_or_depression": False,
    "overwrites_previous_39_42_outputs": False,
    "A_parameter_file": str(A_PARAM.relative_to(ROOT)),
    "A_parameter_sha256": a_sha,
    "G3_parameter_file": str(G_PARAM.relative_to(ROOT)),
    "G3_parameter_sha256": g_sha,
    "A_retained_components": int(K_A),
    "A_0506_to_0708_reconstruction_energy": float(a_0506_to_0708_energy),
    "A_cross_cycle_loading_similarity": [float(x) for x in sim_a],
    "G3_cross_cycle_loading_similarity": [float(x) for x in sim_g],
}
(AUDIT / "46_outcome_independent_refreeze.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print(f"PASS  A outcome-independent discovery n={len(a_disc):,}; K={K_A}")
print(f"PASS  A strict 0506->0708 reconstruction energy={a_0506_to_0708_energy:.6f}")
print(f"PASS  G3 outcome-independent discovery n={len(g_disc):,}")
print("PASS  New transforms and temporal projections saved under prefix 46.")
print("PASS  Scripts 39/42 and their outputs remain untouched.")
