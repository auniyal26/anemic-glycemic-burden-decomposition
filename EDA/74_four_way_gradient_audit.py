#!/usr/bin/env python
# =============================================================================
# 74_four_way_gradient_audit.py
#
# PARTIAL-DERIVATIVE AUDIT OF THE FROZEN SCRIPT-72 DECOMPOSITION
#
# Prof. Saeed question:
#
#   Which variables actually influence each learned component/function?
#
# Forward map:
#
#   f : X -> Z
#
#       dz_i / dx_j
#
# Reverse map:
#
#   g : Z -> X
#
#       dx_j / dz_i
#
# Because the current maps are affine, all first derivatives are constant
# everywhere. Therefore:
#
#   derivative == 0
#
# means that variable has NO DIRECT FUNCTIONAL DEPENDENCE on that input
# anywhere in the learned map.
#
# No Pearson/Spearman correlations are calculated in this script.
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOAD FROZEN TRANSFORM
# =============================================================================

CANDIDATES = [
    RESULTS / "72_shared_private_frozen_transform.npz",
    ROOT / "72_shared_private_frozen_transform.npz",
]

P = next((p for p in CANDIDATES if p.exists()), None)

if P is None:
    raise FileNotFoundError(
        "72_shared_private_frozen_transform.npz not found"
    )

npz = np.load(P)

A_mean = np.asarray(npz["A_mean"], dtype=float).reshape(-1)
A_sd   = np.asarray(npz["A_sd"], dtype=float).reshape(-1)
G_mean = np.asarray(npz["G_mean"], dtype=float).reshape(-1)
G_sd   = np.asarray(npz["G_sd"], dtype=float).reshape(-1)

WA = np.asarray(npz["WA"], dtype=float)
WG = np.asarray(npz["WG"], dtype=float)

rho = np.asarray(npz["rho"], dtype=float).reshape(-1)

shared_rank = int(
    np.asarray(npz["shared_rank"]).reshape(-1)[0]
)

if shared_rank != 2:
    raise ValueError(
        f"Expected shared_rank=2, got {shared_rank}"
    )


# =============================================================================
# NAMES
# =============================================================================

X_NAMES = [
    "Hb",
    "RBC",
    "MCV",
    "RDW",
    "HbA1c",
    "Glucose",
    "logInsulin",
]

Z_NAMES = [
    "Shared1",
    "Shared2",
    "Discord1",
    "Discord2",
    "A_private3",
    "A_private4",
    "G_private3",
]


# =============================================================================
# BUILD f : X -> Z
# =============================================================================
#
# Script 72:
#
# U = ((A - A_mean) / A_sd) @ WA
# V = ((G - G_mean) / G_sd) @ WG
#
# Therefore:
#
# dU/dA = WA.T @ diag(1/A_sd)
# dV/dG = WG.T @ diag(1/G_sd)
#
# =============================================================================

J_uA = WA.T @ np.diag(1.0 / A_sd)
J_vG = WG.T @ np.diag(1.0 / G_sd)

# Canonical coordinate vector:
# c = [u1,u2,u3,u4,v1,v2,v3]

J_cx = np.zeros((7, 7))
J_cx[:4, :4] = J_uA
J_cx[4:, 4:] = J_vG


# =============================================================================
# CANONICAL -> SHARED / DISCORDANT / PRIVATE
# =============================================================================

H = np.zeros((7, 7))

for j in range(2):
    shared_den = np.sqrt(
        2.0 * (1.0 + rho[j])
    )

    discord_den = np.sqrt(
        2.0 * (1.0 - rho[j])
    )

    # Shared direction
    H[j, j] = 1.0 / shared_den
    H[j, 4 + j] = 1.0 / shared_den

    # Discordant direction
    H[2 + j, j] = 1.0 / discord_den
    H[2 + j, 4 + j] = -1.0 / discord_den

# Remaining private coordinates
H[4, 2] = 1.0
H[5, 3] = 1.0
H[6, 6] = 1.0


# =============================================================================
# FORWARD JACOBIAN
#
#             dz_i
# J_f[i,j] = ----
#             dx_j
# =============================================================================

J_f = H @ J_cx

x_mean = np.r_[A_mean, G_mean]
b_f = -J_f @ x_mean


# =============================================================================
# REVERSE JACOBIAN
#
#             dx_j
# J_g[j,i] = ----
#             dz_i
# =============================================================================

rank = np.linalg.matrix_rank(J_f)

if rank != 7:
    raise RuntimeError(
        f"Forward transform is not full rank: rank={rank}/7"
    )

J_g = np.linalg.inv(J_f)
b_g = -J_g @ b_f


# =============================================================================
# FUNCTIONS
# =============================================================================

def f(x):
    x = np.asarray(x, dtype=float)
    return J_f @ x + b_f


def g(z):
    z = np.asarray(z, dtype=float)
    return J_g @ z + b_g


# =============================================================================
# FINITE-DIFFERENCE JACOBIAN CHECK
# =============================================================================

def jacobian_fd(fun, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(fun(x), dtype=float)

    J = np.zeros((len(y), len(x)))

    for j in range(len(x)):
        h = eps * max(1.0, abs(x[j]))

        xp = x.copy()
        xm = x.copy()

        xp[j] += h
        xm[j] -= h

        J[:, j] = (
            np.asarray(fun(xp)) -
            np.asarray(fun(xm))
        ) / (2.0 * h)

    return J


x0 = x_mean.copy()
z0 = f(x0)

J_f_fd = jacobian_fd(f, x0)
J_g_fd = jacobian_fd(g, z0)

I = np.eye(7)

J_fog = J_f @ J_g
J_gof = J_g @ J_f


def maxabs(M):
    return float(np.max(np.abs(M)))


# =============================================================================
# SECOND DERIVATIVE CHECK
#
# Current maps are affine, so all second derivatives should be zero.
# =============================================================================

def hessian_fd_scalar(fun_scalar, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    n = len(x)

    H2 = np.zeros((n, n))

    for i in range(n):
        hi = eps * max(1.0, abs(x[i]))

        for j in range(n):
            hj = eps * max(1.0, abs(x[j]))

            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()

            xpp[i] += hi
            xpp[j] += hj

            xpm[i] += hi
            xpm[j] -= hj

            xmp[i] -= hi
            xmp[j] += hj

            xmm[i] -= hi
            xmm[j] -= hj

            H2[i, j] = (
                fun_scalar(xpp)
                - fun_scalar(xpm)
                - fun_scalar(xmp)
                + fun_scalar(xmm)
            ) / (4.0 * hi * hj)

    return H2


def max_output_hessian(fun, x):
    y = np.asarray(fun(x), dtype=float)

    vals = []

    for k in range(len(y)):
        Hk = hessian_fd_scalar(
            lambda q, kk=k: fun(q)[kk],
            x
        )
        vals.append(
            np.max(np.abs(Hk))
        )

    return float(max(vals))


H_F_MAX = max_output_hessian(
    f,
    x0
)

H_G_MAX = max_output_hessian(
    g,
    z0
)

H_FOG_MAX = max_output_hessian(
    lambda z: f(g(z)),
    z0
)

H_GOF_MAX = max_output_hessian(
    lambda x: g(f(x)),
    x0
)


# =============================================================================
# STANDARDISED PARTIAL DERIVATIVES
#
# Raw derivatives have units, so magnitudes across different biomarkers
# are not directly comparable.
#
# For the forward map:
#
#   dz_i / d(x_j / SD_j)
#
# = dz_i/dx_j * SD_j
#
# This says:
#
#   how much latent component z_i changes when biomarker x_j changes
#   by one original-sample SD.
#
# =============================================================================

X_SD = np.r_[A_sd, G_sd]

J_f_std = J_f @ np.diag(X_SD)


# =============================================================================
# NONZERO / ZERO DEPENDENCE AUDIT
# =============================================================================
#
# Since the transform is affine, exact algebraic zero is meaningful.
#
# Numerical tolerance is used only because floating point arithmetic can
# produce values such as 1e-16 instead of exactly 0.
# =============================================================================

ZERO_TOL = 1e-12

forward_rows = []

for i, z_name in enumerate(Z_NAMES):
    for j, x_name in enumerate(X_NAMES):

        raw = float(J_f[i, j])
        std = float(J_f_std[i, j])

        forward_rows.append({
            "component": z_name,
            "input_variable": x_name,
            "partial_derivative_dz_dx": raw,
            "std_partial_derivative_per_1SD_x": std,
            "abs_std_partial_derivative": abs(std),
            "direct_dependence":
                "YES" if abs(raw) > ZERO_TOL else "NO",
            "sign":
                (
                    "positive"
                    if raw > ZERO_TOL
                    else "negative"
                    if raw < -ZERO_TOL
                    else "zero"
                ),
        })

forward_df = pd.DataFrame(forward_rows)

forward_df.to_csv(
    RESULTS / "74_forward_partial_derivatives.csv",
    index=False
)


# =============================================================================
# REVERSE DEPENDENCE AUDIT
# =============================================================================

reverse_rows = []

for j, x_name in enumerate(X_NAMES):
    for i, z_name in enumerate(Z_NAMES):

        val = float(J_g[j, i])

        reverse_rows.append({
            "reconstructed_variable": x_name,
            "component": z_name,
            "partial_derivative_dx_dz": val,
            "abs_partial_derivative": abs(val),
            "reconstruction_dependence":
                "YES" if abs(val) > ZERO_TOL else "NO",
            "sign":
                (
                    "positive"
                    if val > ZERO_TOL
                    else "negative"
                    if val < -ZERO_TOL
                    else "zero"
                ),
        })

reverse_df = pd.DataFrame(reverse_rows)

reverse_df.to_csv(
    RESULTS / "74_reverse_partial_derivatives.csv",
    index=False
)


# =============================================================================
# COMPONENT-LEVEL SUMMARY
# =============================================================================

component_summary = []

for i, z_name in enumerate(Z_NAMES):

    row = J_f[i, :]
    row_std = J_f_std[i, :]

    nz = np.abs(row) > ZERO_TOL

    order = np.argsort(
        np.abs(row_std)
    )[::-1]

    strongest = X_NAMES[
        order[0]
    ]

    second = X_NAMES[
        order[1]
    ]

    component_summary.append({
        "component": z_name,

        "n_directly_dependent_inputs":
            int(nz.sum()),

        "dependent_inputs":
            "|".join(
                np.array(X_NAMES)[nz]
            ),

        "strongest_input_by_standardized_derivative":
            strongest,

        "strongest_std_partial":
            float(row_std[order[0]]),

        "second_strongest_input":
            second,

        "second_strongest_std_partial":
            float(row_std[order[1]]),

        "hematology_dependency":
            bool(
                np.any(
                    np.abs(row[:4]) >
                    ZERO_TOL
                )
            ),

        "glycemia_dependency":
            bool(
                np.any(
                    np.abs(row[4:]) >
                    ZERO_TOL
                )
            ),
    })

component_summary_df = pd.DataFrame(
    component_summary
)

component_summary_df.to_csv(
    RESULTS / "74_component_dependence_summary.csv",
    index=False
)


# =============================================================================
# BIOMARKER-LEVEL SUMMARY
# =============================================================================

biomarker_summary = []

for j, x_name in enumerate(X_NAMES):

    col = J_f[:, j]
    col_std = J_f_std[:, j]

    nz = np.abs(col) > ZERO_TOL

    order = np.argsort(
        np.abs(col_std)
    )[::-1]

    biomarker_summary.append({
        "input_variable": x_name,

        "n_components_directly_affected":
            int(nz.sum()),

        "affected_components":
            "|".join(
                np.array(Z_NAMES)[nz]
            ),

        "strongest_component_by_standardized_derivative":
            Z_NAMES[order[0]],

        "strongest_std_partial":
            float(col_std[order[0]]),
    })

biomarker_summary_df = pd.DataFrame(
    biomarker_summary
)

biomarker_summary_df.to_csv(
    RESULTS / "74_biomarker_dependence_summary.csv",
    index=False
)


# =============================================================================
# FULL MATRICES
# =============================================================================

forward_matrix = pd.DataFrame(
    J_f,
    index=Z_NAMES,
    columns=X_NAMES
)

forward_std_matrix = pd.DataFrame(
    J_f_std,
    index=Z_NAMES,
    columns=X_NAMES
)

reverse_matrix = pd.DataFrame(
    J_g,
    index=X_NAMES,
    columns=Z_NAMES
)

forward_matrix.to_csv(
    RESULTS / "74_dZ_dX_matrix.csv"
)

forward_std_matrix.to_csv(
    RESULTS / "74_dZ_dX_standardized_matrix.csv"
)

reverse_matrix.to_csv(
    RESULTS / "74_dX_dZ_matrix.csv"
)


# =============================================================================
# MATRIX / INVERTIBILITY SUMMARY
# =============================================================================

svals = np.linalg.svd(
    J_f,
    compute_uv=False
)

cond = np.linalg.cond(
    J_f
)

det = np.linalg.det(
    J_f
)

checks = pd.DataFrame([
    {
        "metric": "rank_J_f",
        "value": float(
            np.linalg.matrix_rank(J_f)
        ),
    },
    {
        "metric": "det_J_f",
        "value": float(det),
    },
    {
        "metric": "condition_number_J_f",
        "value": float(cond),
    },
    {
        "metric": "max_abs_JfJg_minus_I",
        "value": maxabs(
            J_fog - I
        ),
    },
    {
        "metric": "max_abs_JgJf_minus_I",
        "value": maxabs(
            J_gof - I
        ),
    },
    {
        "metric":
            "max_abs_finite_difference_Jf_minus_analytic_Jf",
        "value": maxabs(
            J_f_fd - J_f
        ),
    },
    {
        "metric":
            "max_abs_finite_difference_Jg_minus_analytic_Jg",
        "value": maxabs(
            J_g_fd - J_g
        ),
    },
    {
        "metric": "max_abs_Hessian_f",
        "value": H_F_MAX,
    },
    {
        "metric": "max_abs_Hessian_g",
        "value": H_G_MAX,
    },
    {
        "metric": "max_abs_Hessian_f_after_g",
        "value": H_FOG_MAX,
    },
    {
        "metric": "max_abs_Hessian_g_after_f",
        "value": H_GOF_MAX,
    },
])

checks.to_csv(
    RESULTS / "74_gradient_audit_checks.csv",
    index=False
)


# =============================================================================
# PRINT AUDIT
# =============================================================================

np.set_printoptions(
    precision=6,
    suppress=True,
    linewidth=200
)


print("=" * 100)
print(
    "SCRIPT 74 — COMPLETE PARTIAL-DERIVATIVE / FUNCTIONAL-DEPENDENCE AUDIT"
)
print("=" * 100)

print()
print("FORWARD MAP")
print("  f : X -> Z")
print("  testing dz_i / dx_j for EVERY component × biomarker pair")
print()

print(
    forward_std_matrix.round(5).to_string()
)

print()
print("-" * 100)
print("DIRECT DEPENDENCE BY COMPONENT")
print("-" * 100)

for i, z_name in enumerate(Z_NAMES):

    print()
    print(z_name)

    vals = []

    for j, x_name in enumerate(X_NAMES):

        raw = J_f[i, j]
        std = J_f_std[i, j]

        state = (
            "DEPENDENT"
            if abs(raw) > ZERO_TOL
            else "ZERO"
        )

        vals.append(
            (
                x_name,
                raw,
                std,
                state,
            )
        )

    vals.sort(
        key=lambda t:
        abs(t[2]),
        reverse=True
    )

    for x_name, raw, std, state in vals:

        print(
            f"  {x_name:12s} "
            f"dz/dx={raw:+.8f}   "
            f"std={std:+.6f}   "
            f"{state}"
        )


print()
print("-" * 100)
print("REVERSE MAP")
print("-" * 100)

print()
print("  g : Z -> X")
print("  testing dx_j / dz_i for EVERY biomarker × component pair")
print()

for j, x_name in enumerate(X_NAMES):

    print()
    print(x_name)

    vals = []

    for i, z_name in enumerate(Z_NAMES):

        val = J_g[j, i]

        state = (
            "DEPENDENT"
            if abs(val) > ZERO_TOL
            else "ZERO"
        )

        vals.append(
            (
                z_name,
                val,
                state,
            )
        )

    vals.sort(
        key=lambda t:
        abs(t[1]),
        reverse=True
    )

    for z_name, val, state in vals:

        print(
            f"  {z_name:12s} "
            f"dx/dz={val:+.8f}   "
            f"{state}"
        )


print()
print("-" * 100)
print("MATRIX CHECKS")
print("-" * 100)

print(
    f"rank(J_f)            = "
    f"{np.linalg.matrix_rank(J_f)}/7"
)

print(
    f"det(J_f)             = "
    f"{det:.8e}"
)

print(
    f"condition number     = "
    f"{cond:.6f}"
)

print(
    "singular values      = "
    f"{np.array2string(svals, precision=6)}"
)

print(
    f"max|J_f J_g - I|     = "
    f"{maxabs(J_fog - I):.3e}"
)

print(
    f"max|J_g J_f - I|     = "
    f"{maxabs(J_gof - I):.3e}"
)


print()
print("-" * 100)
print("FINITE-DIFFERENCE VERIFICATION")
print("-" * 100)

print(
    f"max|FD(df/dx)-J_f|   = "
    f"{maxabs(J_f_fd - J_f):.3e}"
)

print(
    f"max|FD(dg/dz)-J_g|   = "
    f"{maxabs(J_g_fd - J_g):.3e}"
)


print()
print("-" * 100)
print("SECOND DERIVATIVES")
print("-" * 100)

print(
    f"max|H_f|             = "
    f"{H_F_MAX:.3e}"
)

print(
    f"max|H_g|             = "
    f"{H_G_MAX:.3e}"
)

print(
    f"max|H_fog|           = "
    f"{H_FOG_MAX:.3e}"
)

print(
    f"max|H_gof|           = "
    f"{H_GOF_MAX:.3e}"
)

print(
    "Expected: approximately zero because "
    "the current maps are affine."
)


print()
print("-" * 100)
print("COMPONENT SUMMARY")
print("-" * 100)

print(
    component_summary_df.to_string(
        index=False
    )
)


print()
print("-" * 100)
print("BIOMARKER SUMMARY")
print("-" * 100)

print(
    biomarker_summary_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("INTERPRETATION RULE")
print("=" * 100)

print("""
For this CURRENT AFFINE decomposition:

    dz_i/dx_j = 0

means component z_i has no direct functional dependence on biomarker x_j
in the frozen transformation.

A nonzero derivative means x_j directly enters that component.

The standardized derivative is provided ONLY to compare the relative
sensitivity to biomarkers measured on different scales.

No correlation coefficient is calculated or required here.

This does NOT establish:
    - causality,
    - biological mechanism,
    - phenotype association,
    - statistical independence.

It establishes the exact direct dependence structure of the learned
representation.
""")

print("=" * 100)
print("OUTPUTS")
print("=" * 100)

for name in [
    "74_forward_partial_derivatives.csv",
    "74_reverse_partial_derivatives.csv",
    "74_component_dependence_summary.csv",
    "74_biomarker_dependence_summary.csv",
    "74_dZ_dX_matrix.csv",
    "74_dZ_dX_standardized_matrix.csv",
    "74_dX_dZ_matrix.csv",
    "74_gradient_audit_checks.csv",
]:
    print(
        RESULTS / name
    )

print("=" * 100)
