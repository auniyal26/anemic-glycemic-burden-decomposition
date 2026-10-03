#!/usr/bin/env python
# =============================================================================
# 81_minimum_residual_augmentation.py
#
# Goal:
#   Quantify how much of frozen physiological state Z can be recovered from
#   phenotype Y, and how many extra residual coordinates R are needed for
#   near-exact reconstruction.
#
# Frozen discovery setup:
#   Y = [SOMATIC_SCORE, COGAFF_SUM]
#   Z = 7D shared/private physiological coordinates
#
# Learn on 2005-08 only:
#   1) g: Y -> Z
#   2) residual basis from E = Z - g(Y)
#
# Then test frozen on:
#   2005-08, 2009-18, 2021-23
#
# Augmented bridge:
#   (Y, R_k) -> Z
#
# where R_k are the first k residual coordinates.
#
# This is an information/reconstruction audit, not a causal model.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    RESULTS / "73_frozen_phenotype_map_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
]
INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), None)
if INPUT is None:
    raise FileNotFoundError(
        "Need EDA/Results/73_frozen_phenotype_map_input.csv. Run Script 73 first."
    )

ZCOLS = [
    "SP_SHARED1", "SP_SHARED2",
    "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]

YCOLS = ["SOMATIC_SCORE", "COGAFF_SUM"]
PERIODS = ["2005-2008", "2009-2018", "2021-2023"]


def weighted_mean_sd(X, w):
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    sw = w.sum()
    mu = np.sum(w[:, None] * X, axis=0) / sw
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def standardize(X, mu, sd):
    return (np.asarray(X, float) - mu) / sd


def weighted_mse(Y, P, w):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    w = np.asarray(w, float)
    return float(
        np.sum(w[:, None] * (Y - P) ** 2)
        / (np.sum(w) * Y.shape[1])
    )


def multivar_r2(Y, P, w):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    w = np.asarray(w, float)
    mu = np.sum(w[:, None] * Y, axis=0) / np.sum(w)
    sse = np.sum(w[:, None] * (Y - P) ** 2)
    sst = np.sum(w[:, None] * (Y - mu) ** 2)
    return float(1 - sse / sst) if sst > 0 else np.nan


def per_dim_r2(Y, P, w, names):
    out = []
    for j, name in enumerate(names):
        y = Y[:, j]
        p = P[:, j]
        mu = np.sum(w * y) / np.sum(w)
        sse = np.sum(w * (y - p) ** 2)
        sst = np.sum(w * (y - mu) ** 2)
        out.append({
            "dimension": name,
            "R2": float(1 - sse / sst) if sst > 0 else np.nan,
        })
    return out


def tune_ridge(A, B, w, seed=42):
    grid = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    scored = []
    for alpha in grid:
        vals = []
        for tr, va in kf.split(A):
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(A[tr], B[tr], sample_weight=w[tr])
            p = m.predict(A[va])
            vals.append(weighted_mse(B[va], p, w[va]))
        scored.append((float(np.mean(vals)), alpha))

    scored.sort(key=lambda x: x[0])
    alpha = scored[0][1]

    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(A, B, sample_weight=w)
    return m, alpha


def weighted_residual_basis(E, w):
    """
    Weighted PCA/SVD of residual matrix E.
    Returns orthonormal basis vectors in Z-space ordered by residual variance.
    """
    E = np.asarray(E, float)
    w = np.asarray(w, float)
    sw = w.sum()

    mu = np.sum(w[:, None] * E, axis=0) / sw
    Ec = E - mu

    Xw = Ec * np.sqrt(w[:, None] / sw)
    _, s, vt = np.linalg.svd(Xw, full_matrices=False)

    eig = s ** 2
    frac = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)

    return mu, vt.T, eig, frac


d = pd.read_csv(INPUT)

required = (
    ["PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG"]
    + ZCOLS + YCOLS
)
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for c in ["SURVEY_WT"] + ZCOLS + YCOLS:
    d[c] = pd.to_numeric(d[c], errors="coerce")

d = d[
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[ZCOLS + YCOLS + ["SURVEY_WT"]].notna().all(axis=1)
    & (d["SURVEY_WT"] > 0)
].copy()

disc = d[d["PERIOD"].eq("2005-2008")].copy()
if len(disc) < 100:
    raise RuntimeError(
        f"Too few discovery rows after required-field filtering: n={len(disc)}"
    )

Z0 = disc[ZCOLS].to_numpy(float)
Y0 = disc[YCOLS].to_numpy(float)
w0 = disc["SURVEY_WT"].to_numpy(float)

zmu, zsd = weighted_mean_sd(Z0, w0)
ymu, ysd = weighted_mean_sd(Y0, w0)

Z0s = standardize(Z0, zmu, zsd)
Y0s = standardize(Y0, ymu, ysd)

# -------------------------------------------------------------------------
# 1. Learn phenotype -> physiology map only on 2005-08.
# -------------------------------------------------------------------------
g, alpha = tune_ridge(Y0s, Z0s, w0, seed=42)
Z0hat = g.predict(Y0s)

base_r2 = multivar_r2(Z0s, Z0hat, w0)

# -------------------------------------------------------------------------
# 2. Learn residual physiological basis.
# -------------------------------------------------------------------------
E0 = Z0s - Z0hat
resid_mu, basis, eig, frac = weighted_residual_basis(E0, w0)

rank_g = int(np.linalg.matrix_rank(np.atleast_2d(g.coef_)))
effective_rank = int(np.sum(eig > (eig.max() * 1e-8))) if eig.max() > 0 else 0

basis_rows = []
cum = 0.0
for k in range(len(frac)):
    cum += float(frac[k])
    basis_rows.append({
        "residual_component": k + 1,
        "residual_variance_fraction": float(frac[k]),
        "cumulative_residual_variance_fraction": cum,
        **{f"loading_{ZCOLS[j]}": float(basis[j, k]) for j in range(len(ZCOLS))}
    })

pd.DataFrame(basis_rows).to_csv(
    RESULTS / "81_residual_basis.csv",
    index=False
)

# -------------------------------------------------------------------------
# 3. Frozen temporal test for k = 0..7 residual coordinates.
# -------------------------------------------------------------------------
summary_rows = []
detail_rows = []

for period in PERIODS:
    q = d[d["PERIOD"].eq(period)].copy()
    if len(q) < 50:
        continue

    Z = q[ZCOLS].to_numpy(float)
    Y = q[YCOLS].to_numpy(float)
    w = q["SURVEY_WT"].to_numpy(float)

    Zs = standardize(Z, zmu, zsd)
    Ys = standardize(Y, ymu, ysd)

    Zbase = g.predict(Ys)
    E = Zs - Zbase

    # Center residual using discovery residual mean, then encode in frozen basis.
    Ec = E - resid_mu
    scores = Ec @ basis

    for k in range(0, len(ZCOLS) + 1):
        if k == 0:
            Zrec = Zbase.copy()
        else:
            Erec = scores[:, :k] @ basis[:, :k].T + resid_mu
            Zrec = Zbase + Erec

        r2 = multivar_r2(Zs, Zrec, w)
        mse = weighted_mse(Zs, Zrec, w)

        summary_rows.append({
            "period": period,
            "n": len(q),
            "k_residual_dimensions": k,
            "Z_reconstruction_R2": r2,
            "weighted_MSE_std": mse,
        })

        for row in per_dim_r2(Zs, Zrec, w, ZCOLS):
            detail_rows.append({
                "period": period,
                "k_residual_dimensions": k,
                **row,
            })

summary = pd.DataFrame(summary_rows)
details = pd.DataFrame(detail_rows)

summary.to_csv(
    RESULTS / "81_residual_augmentation_summary.csv",
    index=False
)
details.to_csv(
    RESULTS / "81_residual_augmentation_per_Z.csv",
    index=False
)

# -------------------------------------------------------------------------
# 4. Minimum k required for reconstruction thresholds.
# -------------------------------------------------------------------------
threshold_rows = []
thresholds = [0.80, 0.90, 0.95, 0.99, 0.999]

for period in PERIODS:
    q = summary[summary["period"].eq(period)].sort_values("k_residual_dimensions")
    if q.empty:
        continue

    for t in thresholds:
        hit = q[q["Z_reconstruction_R2"] >= t]
        threshold_rows.append({
            "period": period,
            "target_R2": t,
            "minimum_k": int(hit.iloc[0]["k_residual_dimensions"]) if not hit.empty else np.nan,
        })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    RESULTS / "81_minimum_residual_dimensions.csv",
    index=False
)

manifest = {
    "script": "81_minimum_residual_augmentation.py",
    "training_period": "2005-2008 only",
    "future_refit": False,
    "Y_state": YCOLS,
    "Z_state": ZCOLS,
    "Y_dimension": len(YCOLS),
    "Z_dimension": len(ZCOLS),
    "rank_Y_to_Z_map": rank_g,
    "residual_effective_rank": effective_rank,
    "ridge_alpha": alpha,
    "discovery_Y_to_Z_R2": base_r2,
    "interpretation": (
        "R_k is an oracle side-channel used only to measure how many extra "
        "dimensions must accompany phenotype Y for faithful Z reconstruction."
    ),
    "causal": False,
}

(RESULTS / "81_residual_augmentation_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 108)
print("SCRIPT 81 — MINIMUM RESIDUAL AUGMENTATION")
print("=" * 108)
print(f"Discovery n = {len(disc)}")
print(f"Y dimensions = {len(YCOLS)} | Z dimensions = {len(ZCOLS)}")
print(f"Rank(Y->Z map) = {rank_g}")
print(f"Discovery Y->Z base R2 = {base_r2:.4f}")
print(f"Residual effective rank = {effective_rank}")
print(f"Selected ridge alpha = {alpha:g}")
print()

for period in PERIODS:
    q = summary[summary["period"].eq(period)]
    if q.empty:
        continue

    print(f"[{period}]")
    for k in range(0, len(ZCOLS) + 1):
        z = q[q["k_residual_dimensions"].eq(k)]
        if z.empty:
            continue
        r = z.iloc[0]
        print(
            f"  k={k}: Z reconstruction R2={r.Z_reconstruction_R2:.4f}"
        )

    tq = threshold_df[threshold_df["period"].eq(period)]
    vals = ", ".join(
        [
            f"R2>={row.target_R2:.3f}: k={row.minimum_k}"
            for _, row in tq.iterrows()
        ]
    )
    print(f"  minimum residual dimensions -> {vals}")
    print()

print("HOW TO READ THIS")
print("  k=0  : phenotype Y alone tries to reconstruct Z.")
print("  k>0  : phenotype plus k extra physiological residual coordinates.")
print("  Small k for high R2 => compact augmented bidirectional bridge is plausible.")
print("  Large k => phenotype carries little of the physiological state by itself.")
print()
print("R_k is an oracle augmentation for information accounting only.")
print("It is NOT yet a deployable predictor and it is NOT causal.")
print("=" * 108)
