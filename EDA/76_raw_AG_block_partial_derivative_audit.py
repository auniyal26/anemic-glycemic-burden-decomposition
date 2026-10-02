#!/usr/bin/env python
# =============================================================================
# 76_raw_AG_block_partial_derivative_audit.py
#
# DIRECT A-BLOCK / G-BLOCK PHENOTYPE DERIVATIVE AUDIT
#
# Raw physiological blocks:
#
#   A = [Hb, RBC, MCV, RDW]
#   G = [HbA1c, Glucose, logInsulin]
#
# Existing frozen bridge:
#
#   X = [A, G]  --f-->  Z
#
# Existing phenotype model:
#
#   Z  -->  P_hat
#
# Chain rule:
#
#   dP/dX = dP/dZ @ dZ/dX
#
# This gives:
#
#   grad_A P = [dP/dHb, dP/dRBC, dP/dMCV, dP/dRDW]
#   grad_G P = [dP/dHbA1c, dP/dGlucose, dP/dlogInsulin]
#
# No correlations are calculated.
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

TRANSFORM_CANDIDATES = [
    RESULTS / "72_shared_private_frozen_transform.npz",
    ROOT / "72_shared_private_frozen_transform.npz",
]

COEF_CANDIDATES = [
    RESULTS / "72_shared_private_coefficients.csv",
    ROOT / "72_shared_private_coefficients.csv",
]

TRANSFORM_FILE = next((p for p in TRANSFORM_CANDIDATES if p.exists()), None)
COEF_FILE = next((p for p in COEF_CANDIDATES if p.exists()), None)

if TRANSFORM_FILE is None:
    raise FileNotFoundError("Could not find 72_shared_private_frozen_transform.npz")

if COEF_FILE is None:
    raise FileNotFoundError("Could not find 72_shared_private_coefficients.csv")

A_NAMES = ["Hb", "RBC", "MCV", "RDW"]
G_NAMES = ["HbA1c", "Glucose", "logInsulin"]
X_NAMES = A_NAMES + G_NAMES

Z_TERMS = [
    "SP_SHARED1",
    "SP_SHARED2",
    "SP_DISCORD1",
    "SP_DISCORD2",
    "SP_A_PRIVATE3",
    "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
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

ZERO_TOL = 1e-12

# =============================================================================
# LOAD FROZEN X -> Z TRANSFORM
# =============================================================================

npz = np.load(TRANSFORM_FILE)

A_mean = np.asarray(npz["A_mean"], dtype=float).reshape(-1)
A_sd   = np.asarray(npz["A_sd"], dtype=float).reshape(-1)
G_mean = np.asarray(npz["G_mean"], dtype=float).reshape(-1)
G_sd   = np.asarray(npz["G_sd"], dtype=float).reshape(-1)
WA     = np.asarray(npz["WA"], dtype=float)
WG     = np.asarray(npz["WG"], dtype=float)
rho    = np.asarray(npz["rho"], dtype=float).reshape(-1)

shared_rank = int(np.asarray(npz["shared_rank"]).reshape(-1)[0])

if shared_rank != 2:
    raise ValueError(f"Expected shared_rank=2, got {shared_rank}")

# =============================================================================
# REBUILD J_f = dZ/dX
# =============================================================================

J_uA = WA.T @ np.diag(1.0 / A_sd)
J_vG = WG.T @ np.diag(1.0 / G_sd)

J_cx = np.zeros((7, 7))
J_cx[:4, :4] = J_uA
J_cx[4:, 4:] = J_vG

H = np.zeros((7, 7))

for j in range(2):
    shared_den = np.sqrt(2.0 * (1.0 + rho[j]))
    discord_den = np.sqrt(2.0 * (1.0 - rho[j]))

    H[j, j] = 1.0 / shared_den
    H[j, 4 + j] = 1.0 / shared_den

    H[2 + j, j] = 1.0 / discord_den
    H[2 + j, 4 + j] = -1.0 / discord_den

H[4, 2] = 1.0
H[5, 3] = 1.0
H[6, 6] = 1.0

J_f = H @ J_cx

if np.linalg.matrix_rank(J_f) != 7:
    raise RuntimeError("Frozen X->Z transform is not full rank.")

# =============================================================================
# LOAD Z -> PHENOTYPE COEFFICIENTS
# =============================================================================

coef = pd.read_csv(COEF_FILE)

required = ["period", "outcome", "term", "beta"]
missing = [c for c in required if c not in coef.columns]

if missing:
    raise ValueError(
        "Coefficient file missing columns: " + ", ".join(missing)
    )

coef = coef[coef["term"].isin(Z_TERMS)].copy()
coef["beta"] = pd.to_numeric(coef["beta"], errors="coerce")

if coef.empty:
    raise RuntimeError("No Script-72 physiological coefficients found.")

def state(v):
    return "ZERO" if abs(v) <= ZERO_TOL else "NONZERO"

# =============================================================================
# CHAIN RULE AUDIT
# =============================================================================

raw_rows = []
block_rows = []
path_rows = []
matrix_rows = []

for (period, outcome), g in coef.groupby(["period", "outcome"], dropna=False):

    beta_map = dict(zip(g["term"], g["beta"]))

    missing_terms = [t for t in Z_TERMS if t not in beta_map]

    if missing_terms:
        raise RuntimeError(
            f"{period} / {outcome}: missing terms "
            + ", ".join(missing_terms)
        )

    beta_z = np.array([beta_map[t] for t in Z_TERMS], dtype=float)

    # dP/dX = dP/dZ @ dZ/dX
    grad_x = beta_z @ J_f

    # Per +1 discovery-SD raw-variable change
    X_SD = np.r_[A_sd, G_sd]
    grad_x_std = grad_x * X_SD

    # Contribution through each latent component:
    # (dP/dz_i) * (dz_i/dx_j)
    contribution = beta_z[:, None] * J_f
    contribution_std = contribution * X_SD[None, :]

    for j, x_name in enumerate(X_NAMES):

        block = "A" if j < 4 else "G"

        raw_rows.append({
            "period": period,
            "outcome": outcome,
            "block": block,
            "raw_variable": x_name,
            "partial_derivative_dP_dx": float(grad_x[j]),
            "std_partial_derivative_per_1SD_x": float(grad_x_std[j]),
            "abs_std_partial_derivative": float(abs(grad_x_std[j])),
            "derivative_state": state(grad_x[j]),
            "direction": (
                "positive" if grad_x[j] > ZERO_TOL
                else "negative" if grad_x[j] < -ZERO_TOL
                else "zero"
            ),
        })

        for i, z_name in enumerate(Z_NAMES):
            path_rows.append({
                "period": period,
                "outcome": outcome,
                "block": block,
                "raw_variable": x_name,
                "latent_component": z_name,
                "dP_dz": float(beta_z[i]),
                "dz_dx": float(J_f[i, j]),
                "path_contribution_to_dP_dx": float(contribution[i, j]),
                "std_path_contribution_per_1SD_x": float(
                    contribution_std[i, j]
                ),
            })

    grad_A = grad_x[:4]
    grad_G = grad_x[4:]
    grad_A_std = grad_x_std[:4]
    grad_G_std = grad_x_std[4:]

    for block_name, names, grad, grad_std in [
        ("A", A_NAMES, grad_A, grad_A_std),
        ("G", G_NAMES, grad_G, grad_G_std),
    ]:

        nz = np.abs(grad) > ZERO_TOL
        strongest_idx = int(np.argmax(np.abs(grad_std)))

        block_rows.append({
            "period": period,
            "outcome": outcome,
            "block": block_name,
            "n_variables": len(names),
            "n_nonzero_derivatives": int(nz.sum()),
            "n_zero_derivatives": int((~nz).sum()),
            "block_gradient_state": (
                "ZERO_BLOCK" if np.all(~nz) else "NONZERO_BLOCK"
            ),
            "variables": "|".join(names),
            "raw_derivatives": "|".join(f"{v:+.10g}" for v in grad),
            "std_derivatives": "|".join(f"{v:+.10g}" for v in grad_std),
            "standardized_gradient_L2": float(
                np.linalg.norm(grad_std, ord=2)
            ),
            "standardized_gradient_L1": float(
                np.sum(np.abs(grad_std))
            ),
            "strongest_raw_variable": names[strongest_idx],
            "strongest_std_derivative": float(grad_std[strongest_idx]),
        })

    matrix_rows.append({
        "period": period,
        "outcome": outcome,
        **{
            f"dP_d_{name}": float(grad_x[j])
            for j, name in enumerate(X_NAMES)
        },
        **{
            f"std_dP_d_{name}": float(grad_x_std[j])
            for j, name in enumerate(X_NAMES)
        },
    })

raw_df = pd.DataFrame(raw_rows)
block_df = pd.DataFrame(block_rows)
path_df = pd.DataFrame(path_rows)
matrix_df = pd.DataFrame(matrix_rows)

# =============================================================================
# VERIFY CHAIN RULE
# =============================================================================

check = (
    path_df
    .groupby(
        ["period", "outcome", "raw_variable"],
        as_index=False
    )["path_contribution_to_dP_dx"]
    .sum()
    .rename(columns={
        "path_contribution_to_dP_dx":
            "sum_latent_path_contributions"
    })
)

check = check.merge(
    raw_df[
        [
            "period",
            "outcome",
            "raw_variable",
            "partial_derivative_dP_dx",
        ]
    ],
    on=["period", "outcome", "raw_variable"],
    how="left",
)

check["absolute_difference"] = np.abs(
    check["sum_latent_path_contributions"]
    - check["partial_derivative_dP_dx"]
)

max_chain_error = float(check["absolute_difference"].max())

# =============================================================================
# SAVE
# =============================================================================

raw_df.to_csv(
    RESULTS / "76_raw_AG_partial_derivatives.csv",
    index=False
)

block_df.to_csv(
    RESULTS / "76_AG_block_gradient_summary.csv",
    index=False
)

path_df.to_csv(
    RESULTS / "76_raw_to_Z_to_phenotype_path_contributions.csv",
    index=False
)

matrix_df.to_csv(
    RESULTS / "76_raw_AG_derivative_matrix.csv",
    index=False
)

check.to_csv(
    RESULTS / "76_chain_rule_verification.csv",
    index=False
)

# =============================================================================
# PRINT
# =============================================================================

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

print("=" * 110)
print("SCRIPT 76 — DIRECT A-BLOCK / G-BLOCK PHENOTYPE PARTIAL-DERIVATIVE AUDIT")
print("=" * 110)

print()
print("Chain rule:")
print("    dP/dX = dP/dZ @ dZ/dX")
print()
print("A = [Hb, RBC, MCV, RDW]")
print("G = [HbA1c, Glucose, logInsulin]")
print("No correlations are calculated.")

for (period, outcome), g in raw_df.groupby(
    ["period", "outcome"],
    dropna=False
):

    print()
    print("=" * 110)
    print(f"PERIOD: {period}    PHENOTYPE: {outcome}")
    print("=" * 110)

    for block_name in ["A", "G"]:

        b = g[g["block"].eq(block_name)]

        print()
        print(f"{block_name} BLOCK")

        for _, r in b.iterrows():
            print(
                f"  dP/d{r['raw_variable']:<11s} "
                f"= {r['partial_derivative_dP_dx']:+.10f}   "
                f"std(1SD)={r['std_partial_derivative_per_1SD_x']:+.6f}   "
                f"{r['derivative_state']}"
            )

        bs = block_df[
            block_df["period"].eq(period)
            & block_df["outcome"].eq(outcome)
            & block_df["block"].eq(block_name)
        ].iloc[0]

        print(
            f"  BLOCK RESULT: {bs['block_gradient_state']} "
            f"({int(bs['n_nonzero_derivatives'])}/"
            f"{int(bs['n_variables'])} raw-variable derivatives nonzero)"
        )

        print(
            f"  standardized gradient L2 = "
            f"{bs['standardized_gradient_L2']:.6f}"
        )

        print(
            f"  strongest direction = "
            f"{bs['strongest_raw_variable']} "
            f"({bs['strongest_std_derivative']:+.6f} per 1 SD)"
        )

print()
print("=" * 110)
print("BLOCK SUMMARY")
print("=" * 110)
print(block_df.to_string(index=False))

print()
print("=" * 110)
print("CHAIN-RULE VERIFICATION")
print("=" * 110)

print(
    "max absolute difference between direct derivative and "
    f"sum of latent path contributions = {max_chain_error:.3e}"
)

if max_chain_error < 1e-10:
    print("PASS — latent paths exactly reconstruct the raw derivative.")
else:
    print("CHECK — chain-rule reconstruction error is larger than expected.")

print()
print("=" * 110)
print("INTERPRETATION")
print("=" * 110)

print(r"""
For each phenotype P:

    grad_A P =
        [dP/dHb,
         dP/dRBC,
         dP/dMCV,
         dP/dRDW]

    grad_G P =
        [dP/dHbA1c,
         dP/dGlucose,
         dP/dlogInsulin]

If every entry of a block gradient were zero, the fitted phenotype function
would not depend on that physiological block through the frozen X->Z bridge.

If at least one entry is nonzero, that block contributes to the fitted
phenotype function.

The standardised derivative reports the fitted phenotype change associated
with a +1 discovery-SD change in the raw biomarker while the other raw
variables and model covariates are held fixed.

This is functional dependence in the fitted model, not a causal effect.
""")

print("=" * 110)
print("OUTPUTS")
print("=" * 110)

for name in [
    "76_raw_AG_partial_derivatives.csv",
    "76_AG_block_gradient_summary.csv",
    "76_raw_to_Z_to_phenotype_path_contributions.csv",
    "76_raw_AG_derivative_matrix.csv",
    "76_chain_rule_verification.csv",
]:
    print(RESULTS / name)

print("=" * 110)
