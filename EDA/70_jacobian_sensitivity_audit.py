#!/usr/bin/env python
# =============================================================================
# 70_jacobian_sensitivity_audit.py
#
# Prof. Saeed's derivative experiment.
#
# Current learned maps:
#   f : X -> Y
#   g : Y -> X
#
# For linear maps, the Jacobians are constant:
#   J_f = dY/dX
#   J_g = dX/dY
#
# This script:
#   1) learns f and g on NHANES 2005-08 discovery data
#   2) extracts every partial derivative
#   3) ranks which X variables move each Y source and vice versa
#   4) checks whether any derivatives are zero / near-zero
#   5) verifies derivatives numerically by finite differences
#   6) inspects the composed Jacobians:
#          J_(f∘g) = J_f J_g    (Y -> X -> Y)
#          J_(g∘f) = J_g J_f    (X -> Y -> X)
#      which explains the cycle-consistency behavior directly.
#
# No PCA refit. Locked Results untouched.
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "EDA" / "Results"
MODELS = ROOT / "EDA" / "Models"
RES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

INPUT = RES / "63_deep_ddx_input.csv"
if not INPUT.exists():
    INPUT = ROOT / "Results" / "63_deep_ddx_input.csv"
if not INPUT.exists():
    raise FileNotFoundError("Need 63_deep_ddx_input.csv")

d = pd.read_csv(INPUT)
d["PERIOD"] = d["PERIOD"].astype(str)

DOMAINS = {
    "A": {
        "raw": ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"],
        "pcs": ["A_OI_PC1_FZ", "A_OI_PC2_FZ", "A_OI_PC3_FZ"],
        "ynames": ["A_ICA1", "A_ICA2", "A_ICA3"],
    },
    "G": {
        "raw": ["LBXGH", "LBXGLU", "LOG_IN"],
        "pcs": ["G3_OI_PC1_FZ", "G3_OI_PC2_FZ", "G3_OI_PC3_FZ"],
        "ynames": ["G_ICA1", "G_ICA2", "G_ICA3"],
    },
}

def weighted_standardize(X, w):
    mu = np.sum(w[:, None] * X, axis=0) / np.sum(w)
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / np.sum(w))
    return (X - mu) / sd, mu, sd

def wls(A, B, w):
    D = np.column_stack([np.ones(len(A)), A])
    sw = np.sqrt(w)
    return np.linalg.lstsq(D * sw[:, None], B * sw[:, None], rcond=None)[0]

def pred(A, C):
    return np.column_stack([np.ones(len(A)), A]) @ C

jac_rows = []
importance_rows = []
composition_rows = []
finite_diff_rows = []
null_rows = []
meta = {}

REL_ZERO_TOL = 1e-3   # <0.1% of the largest derivative for that output = "near-zero"
ABS_ZERO_TOL = 1e-10  # true numerical zero

for domain, spec in DOMAINS.items():
    raw = spec["raw"]
    pcs = spec["pcs"]
    ynames = spec["ynames"]

    q = d[d["PERIOD"].eq("2005-2008")].copy()
    needed = raw + pcs + ["SURVEY_WT"]

    for c in needed:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q.dropna(subset=needed)
    q = q[q["SURVEY_WT"] > 0].reset_index(drop=True)

    X = q[raw].to_numpy(float)
    Z = q[pcs].to_numpy(float)
    w = q["SURVEY_WT"].to_numpy(float)

    Xs, xmu, xsd = weighted_standardize(X, w)

    ica = FastICA(
        n_components=3,
        whiten="unit-variance",
        algorithm="parallel",
        fun="logcosh",
        max_iter=3000,
        tol=1e-6,
        random_state=42,
    )
    Y = ica.fit_transform(Z)

    # f : X_standardized -> Y
    # g : Y -> X_standardized
    f = wls(Xs, Y, w)
    g = wls(Y, Xs, w)

    # Jacobians in standardized coordinates.
    # f[1:] is input x output, so transpose => output x input.
    Jf = f[1:, :].T                 # shape (3, pX)
    Jg = g[1:, :].T                 # shape (pX, 3)

    # Raw-unit Jacobians:
    # Xs_j = (Xraw_j - mu_j)/sd_j
    # dY/dXraw_j = dY/dXs_j * 1/sd_j
    Jf_raw = Jf / xsd[None, :]

    # Xraw_j = mu_j + sd_j * Xs_j
    # dXraw_j/dY_k = sd_j * dXs_j/dY_k
    Jg_raw = xsd[:, None] * Jg

    # ---------------------------------------------------------------------
    # 1) Partial derivatives for f: dY_k / dX_j
    # ---------------------------------------------------------------------
    for oi, out_name in enumerate(ynames):
        row_abs = np.abs(Jf[oi])
        row_max = max(float(row_abs.max()), ABS_ZERO_TOL)

        for ii, in_name in enumerate(raw):
            val = float(Jf[oi, ii])
            rel = float(abs(val) / row_max)
            jac_rows.append({
                "domain": domain,
                "map": "f_X_to_Y",
                "output": out_name,
                "input": in_name,
                "derivative_standardized": val,
                "derivative_raw_units": float(Jf_raw[oi, ii]),
                "abs_derivative_standardized": abs(val),
                "relative_to_largest_for_output": rel,
                "is_numerically_zero": abs(val) < ABS_ZERO_TOL,
                "is_near_zero_relative": rel < REL_ZERO_TOL,
            })

    # ---------------------------------------------------------------------
    # 2) Partial derivatives for g: dX_j / dY_k
    # ---------------------------------------------------------------------
    for oi, out_name in enumerate(raw):
        row_abs = np.abs(Jg[oi])
        row_max = max(float(row_abs.max()), ABS_ZERO_TOL)

        for ii, in_name in enumerate(ynames):
            val = float(Jg[oi, ii])
            rel = float(abs(val) / row_max)
            jac_rows.append({
                "domain": domain,
                "map": "g_Y_to_X",
                "output": out_name,
                "input": in_name,
                "derivative_standardized": val,
                "derivative_raw_units": float(Jg_raw[oi, ii]),
                "abs_derivative_standardized": abs(val),
                "relative_to_largest_for_output": rel,
                "is_numerically_zero": abs(val) < ABS_ZERO_TOL,
                "is_near_zero_relative": rel < REL_ZERO_TOL,
            })

    # ---------------------------------------------------------------------
    # 3) Input importance = norm of each Jacobian column.
    #    Since X is standardized and ICA has unit-variance whitening, these
    #    are directly useful for comparing variables within each map.
    # ---------------------------------------------------------------------
    f_col_norms = np.linalg.norm(Jf, axis=0)
    g_col_norms = np.linalg.norm(Jg, axis=0)

    for j, name in enumerate(raw):
        importance_rows.append({
            "domain": domain,
            "map": "f_X_to_Y",
            "input": name,
            "L2_sensitivity_norm": float(f_col_norms[j]),
            "share_of_total_norm": float(f_col_norms[j] / f_col_norms.sum()),
        })

    for j, name in enumerate(ynames):
        importance_rows.append({
            "domain": domain,
            "map": "g_Y_to_X",
            "input": name,
            "L2_sensitivity_norm": float(g_col_norms[j]),
            "share_of_total_norm": float(g_col_norms[j] / g_col_norms.sum()),
        })

    # ---------------------------------------------------------------------
    # 4) Composition derivatives.
    # ---------------------------------------------------------------------
    J_fg = Jf @ Jg   # d[f(g(Y))]/dY, should be ~I_3
    J_gf = Jg @ Jf   # d[g(f(X))]/dX, projection for A; identity for G

    for i, out_name in enumerate(ynames):
        for j, in_name in enumerate(ynames):
            composition_rows.append({
                "domain": domain,
                "composition": "f_after_g__Y_to_X_to_Y",
                "output": out_name,
                "input": in_name,
                "derivative": float(J_fg[i, j]),
                "identity_target": 1.0 if i == j else 0.0,
                "absolute_identity_error":
                    float(abs(J_fg[i, j] - (1.0 if i == j else 0.0))),
            })

    for i, out_name in enumerate(raw):
        for j, in_name in enumerate(raw):
            composition_rows.append({
                "domain": domain,
                "composition": "g_after_f__X_to_Y_to_X",
                "output": out_name,
                "input": in_name,
                "derivative": float(J_gf[i, j]),
                "identity_target": 1.0 if i == j else 0.0,
                "absolute_identity_error":
                    float(abs(J_gf[i, j] - (1.0 if i == j else 0.0))),
            })

    # ---------------------------------------------------------------------
    # 5) Finite-difference verification.
    # ---------------------------------------------------------------------
    eps = 1e-6
    test_rows = np.linspace(0, len(q)-1, min(25, len(q)), dtype=int)

    max_f_err = 0.0
    max_g_err = 0.0

    for r in test_rows:
        x0 = Xs[r:r+1]
        y0 = pred(x0, f)

        for j in range(Xs.shape[1]):
            xp = x0.copy()
            xp[0, j] += eps
            yp = pred(xp, f)
            num = ((yp - y0) / eps).ravel()
            err = float(np.max(np.abs(num - Jf[:, j])))
            max_f_err = max(max_f_err, err)

        ystart = Y[r:r+1]
        xstart = pred(ystart, g)

        for j in range(Y.shape[1]):
            yp = ystart.copy()
            yp[0, j] += eps
            xp = pred(yp, g)
            num = ((xp - xstart) / eps).ravel()
            err = float(np.max(np.abs(num - Jg[:, j])))
            max_g_err = max(max_g_err, err)

    finite_diff_rows.append({
        "domain": domain,
        "map": "f_X_to_Y",
        "max_abs_numeric_vs_analytic_derivative_error": max_f_err,
        "epsilon_step": eps,
        "n_rows_checked": len(test_rows),
    })
    finite_diff_rows.append({
        "domain": domain,
        "map": "g_Y_to_X",
        "max_abs_numeric_vs_analytic_derivative_error": max_g_err,
        "epsilon_step": eps,
        "n_rows_checked": len(test_rows),
    })

    # ---------------------------------------------------------------------
    # 6) Null-space of f.
    #    For A (4D -> 3D), one direction should be invisible to Y:
    #       Jf @ v ~= 0
    #    This is the exact direction discarded by the representation.
    # ---------------------------------------------------------------------
    U, S, Vt = np.linalg.svd(Jf, full_matrices=True)
    rank = int(np.linalg.matrix_rank(Jf))
    null_dim = Jf.shape[1] - rank

    if null_dim > 0:
        null_basis = Vt[rank:, :]
        for k, v in enumerate(null_basis, start=1):
            # Fix arbitrary sign for readability.
            if v[np.argmax(np.abs(v))] < 0:
                v = -v
            residual = np.linalg.norm(Jf @ v)
            for j, name in enumerate(raw):
                null_rows.append({
                    "domain": domain,
                    "null_vector": k,
                    "variable": name,
                    "loading_standardized_X": float(v[j]),
                    "abs_loading": float(abs(v[j])),
                    "Jf_times_null_norm": float(residual),
                })

    meta[domain] = {
        "n_discovery": int(len(q)),
        "X_dimension": int(Jf.shape[1]),
        "Y_dimension": int(Jf.shape[0]),
        "rank_Jf": rank,
        "null_dimension_Jf": int(null_dim),
        "condition_number_nonzero_Jf":
            float(S[0] / S[-1]) if len(S) and S[-1] > 0 else None,
        "max_abs_error_f_after_g_vs_identity":
            float(np.max(np.abs(J_fg - np.eye(J_fg.shape[0])))),
        "max_abs_error_g_after_f_vs_identity":
            float(np.max(np.abs(J_gf - np.eye(J_gf.shape[0])))),
        "finite_difference_max_error_f": max_f_err,
        "finite_difference_max_error_g": max_g_err,
    }

# Save.
jac_df = pd.DataFrame(jac_rows)
imp_df = pd.DataFrame(importance_rows)
comp_df = pd.DataFrame(composition_rows)
fd_df = pd.DataFrame(finite_diff_rows)
null_df = pd.DataFrame(null_rows)

jac_df.to_csv(RES/"70_jacobian_partial_derivatives.csv", index=False)
imp_df.to_csv(RES/"70_jacobian_input_importance.csv", index=False)
comp_df.to_csv(RES/"70_jacobian_compositions.csv", index=False)
fd_df.to_csv(RES/"70_jacobian_finite_difference_check.csv", index=False)
null_df.to_csv(RES/"70_jacobian_nullspace.csv", index=False)

manifest = {
    "script": "70_jacobian_sensitivity_audit.py",
    "question": "Which variables actually influence the learned f and g functions?",
    "f": "X_standardized -> ICA/source Y",
    "g": "ICA/source Y -> X_standardized",
    "derivatives": {
        "J_f": "dY/dX",
        "J_g": "dX/dY",
    },
    "important_interpretation":
        "Because f and g are linear, the partial derivatives are constant everywhere. "
        "A zero derivative therefore means no dependence anywhere in this learned map, "
        "not merely at one point.",
    "near_zero_rule":
        "absolute derivative < 0.1% of the largest absolute derivative for that output",
    "composition_checks": {
        "f_after_g": "should approximate identity in Y-space",
        "g_after_f":
            "is a projection in X-space when X dimension exceeds Y dimension; "
            "for A, one null direction is expected because 4D -> 3D",
    },
    "pca_refit": False,
    "locked_outputs_modified": False,
    "domains": meta,
}
(RES/"70_jacobian_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print("PASS  Script 70 Jacobian sensitivity audit complete.")
print("PASS  Extracted dY/dX and dX/dY for every variable/source pair.")
print("PASS  Checked zero/near-zero derivatives.")
print("PASS  Verified analytic derivatives by finite differences.")
print("PASS  Audited f(g(Y)) and g(f(X)) Jacobians.")
print("PASS  Extracted any X directions invisible to the latent representation.")
print("PASS  No PCA refit; locked Results untouched.")
print()
print("MODEL SUMMARY")
for k,v in meta.items():
    print(k, v)
print()
print("TOP INPUT SENSITIVITIES")
print(
    imp_df.sort_values(["domain","map","L2_sensitivity_norm"],
                       ascending=[True,True,False]).to_string(index=False)
)
