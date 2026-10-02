#!/usr/bin/env python
# =============================================================================
# 75_phenotype_partial_derivative_audit.py
#
# PROF. SAEED NOTEBOOK TEST — PHYSIOLOGICAL COMPONENTS -> PHENOTYPE
#
# Script 72 fits a survey-weighted Gaussian phenotype model of the form:
#
#   P_hat =
#       beta0
#       + beta_S1 * Shared1
#       + beta_S2 * Shared2
#       + beta_D1 * Discord1
#       + beta_D2 * Discord2
#       + beta_A3 * A_private3
#       + beta_A4 * A_private4
#       + beta_G3 * G_private3
#       + covariate terms
#
# For this linear phenotype function:
#
#       d P_hat / d z_i = beta_i
#
# Therefore this script directly audits, for every phenotype and period:
#
#       dP/dShared1
#       dP/dShared2
#       dP/dDiscord1
#       dP/dDiscord2
#       dP/dA_private3
#       dP/dA_private4
#       dP/dG_private3
#
# Exact zero derivative:
#       phenotype function does not directly depend on that component.
#
# Nonzero derivative:
#       phenotype function directly depends on that component.
#
# IMPORTANT:
# - This is NOT a correlation analysis.
# - This is NOT a causal analysis.
# - p-values / confidence intervals are retained only as inferential context.
# - "CI contains zero" is NOT the same statement as "derivative is zero".
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
# INPUT
# =============================================================================

COEF_CANDIDATES = [
    RESULTS / "72_shared_private_coefficients.csv",
    ROOT / "72_shared_private_coefficients.csv",
]

COEF_FILE = next(
    (p for p in COEF_CANDIDATES if p.exists()),
    None
)

if COEF_FILE is None:
    raise FileNotFoundError(
        "Could not find 72_shared_private_coefficients.csv"
    )

coef = pd.read_csv(COEF_FILE)


# =============================================================================
# EXPECTED SCRIPT-72 PHYSIOLOGICAL TERMS
# =============================================================================

TERMS = [
    "SP_SHARED1",
    "SP_SHARED2",
    "SP_DISCORD1",
    "SP_DISCORD2",
    "SP_A_PRIVATE3",
    "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]

PRETTY = {
    "SP_SHARED1": "Shared1",
    "SP_SHARED2": "Shared2",
    "SP_DISCORD1": "Discord1",
    "SP_DISCORD2": "Discord2",
    "SP_A_PRIVATE3": "A_private3",
    "SP_A_PRIVATE4": "A_private4",
    "SP_G_PRIVATE3": "G_private3",
}

BLOCK = {
    "SP_SHARED1": "AG_shared",
    "SP_SHARED2": "AG_shared",
    "SP_DISCORD1": "AG_discord",
    "SP_DISCORD2": "AG_discord",
    "SP_A_PRIVATE3": "A_private",
    "SP_A_PRIVATE4": "A_private",
    "SP_G_PRIVATE3": "G_private",
}


# =============================================================================
# VALIDATE INPUT
# =============================================================================

required_cols = [
    "period",
    "outcome",
    "model",
    "term",
    "beta",
    "se",
    "ci_low",
    "ci_high",
    "p",
]

missing_cols = [
    c for c in required_cols
    if c not in coef.columns
]

if missing_cols:
    raise ValueError(
        "72_shared_private_coefficients.csv is missing columns: "
        + ", ".join(missing_cols)
    )

coef = coef.copy()

for c in [
    "beta",
    "se",
    "ci_low",
    "ci_high",
    "p",
]:
    coef[c] = pd.to_numeric(
        coef[c],
        errors="coerce"
    )

coef = coef[
    coef["term"].isin(TERMS)
].copy()

if coef.empty:
    raise RuntimeError(
        "No expected Script-72 physiological terms were found."
    )


# =============================================================================
# PARTIAL DERIVATIVE AUDIT
#
# For the linear Gaussian phenotype model:
#
#       P_hat = beta0 + sum_i beta_i z_i + ...
#
#       dP_hat/dz_i = beta_i
#
# =============================================================================

ZERO_TOL = 1e-12

coef["component"] = coef["term"].map(PRETTY)
coef["block"] = coef["term"].map(BLOCK)

coef["partial_derivative_dP_dz"] = coef["beta"]

coef["abs_partial_derivative"] = np.abs(
    coef["partial_derivative_dP_dz"]
)

coef["derivative_state"] = np.where(
    coef["abs_partial_derivative"] <= ZERO_TOL,
    "ZERO",
    "NONZERO"
)

coef["direction"] = np.select(
    [
        coef["partial_derivative_dP_dz"] > ZERO_TOL,
        coef["partial_derivative_dP_dz"] < -ZERO_TOL,
    ],
    [
        "positive",
        "negative",
    ],
    default="zero"
)

coef["ci_contains_zero"] = (
    (coef["ci_low"] <= 0)
    & (coef["ci_high"] >= 0)
)

coef["p_lt_0_05"] = (
    coef["p"] < 0.05
)


# =============================================================================
# COMPLETE GRID CHECK
#
# Every period × outcome should have all seven derivatives.
# =============================================================================

grid_rows = []

for (period, outcome), g in coef.groupby(
    ["period", "outcome"],
    dropna=False
):

    present = set(
        g["term"].astype(str)
    )

    missing = [
        t for t in TERMS
        if t not in present
    ]

    grid_rows.append({
        "period": period,
        "outcome": outcome,
        "n_expected_derivatives": len(TERMS),
        "n_found_derivatives": len(
            present.intersection(TERMS)
        ),
        "complete": len(missing) == 0,
        "missing_terms": "|".join(missing),
    })

grid_df = pd.DataFrame(grid_rows)


# =============================================================================
# BLOCK-LEVEL AUDIT
#
# This answers the higher-level question:
#
#   Does the learned phenotype function depend on:
#       A-private?
#       G-private?
#       AG-shared?
#       AG-discordance?
#
# A block is marked:
#   ALL_ZERO     -> every derivative in the block is zero
#   SOME_NONZERO -> at least one is nonzero
#   ALL_NONZERO  -> every derivative is nonzero
# =============================================================================

block_rows = []

for (
    period,
    outcome,
    block
), g in coef.groupby(
    [
        "period",
        "outcome",
        "block",
    ],
    dropna=False
):

    vals = g[
        "partial_derivative_dP_dz"
    ].to_numpy(dtype=float)

    nz = np.abs(vals) > ZERO_TOL

    if np.all(~nz):
        block_state = "ALL_ZERO"
    elif np.all(nz):
        block_state = "ALL_NONZERO"
    else:
        block_state = "SOME_NONZERO"

    block_rows.append({
        "period": period,
        "outcome": outcome,
        "block": block,
        "n_components": len(g),
        "n_nonzero_derivatives": int(
            nz.sum()
        ),
        "n_zero_derivatives": int(
            (~nz).sum()
        ),
        "block_state": block_state,
        "components": "|".join(
            g["component"].astype(str)
        ),
        "derivatives": "|".join(
            f"{x:+.10g}"
            for x in vals
        ),
    })

block_df = pd.DataFrame(
    block_rows
)


# =============================================================================
# PHENOTYPE-LEVEL SUMMARY
# =============================================================================

summary_rows = []

for (
    period,
    outcome
), g in coef.groupby(
    ["period", "outcome"],
    dropna=False
):

    g = g.copy()

    zero = g[
        g["derivative_state"].eq(
            "ZERO"
        )
    ]

    nonzero = g[
        g["derivative_state"].eq(
            "NONZERO"
        )
    ]

    strongest_idx = (
        g["abs_partial_derivative"]
        .idxmax()
    )

    strongest = g.loc[
        strongest_idx
    ]

    summary_rows.append({
        "period": period,
        "outcome": outcome,

        "n_components_tested":
            len(g),

        "n_nonzero_derivatives":
            len(nonzero),

        "n_zero_derivatives":
            len(zero),

        "zero_components":
            "|".join(
                zero["component"]
                .astype(str)
            ),

        "nonzero_components":
            "|".join(
                nonzero["component"]
                .astype(str)
            ),

        "strongest_component":
            strongest[
                "component"
            ],

        "strongest_derivative":
            strongest[
                "partial_derivative_dP_dz"
            ],
    })

summary_df = pd.DataFrame(
    summary_rows
)


# =============================================================================
# DERIVATIVE MATRIX
#
# One matrix per period/outcome combination is also exported in long form.
# =============================================================================

matrix_df = coef.pivot_table(
    index=[
        "period",
        "outcome",
    ],
    columns="component",
    values="partial_derivative_dP_dz",
    aggfunc="first"
).reset_index()

ordered_component_cols = [
    "Shared1",
    "Shared2",
    "Discord1",
    "Discord2",
    "A_private3",
    "A_private4",
    "G_private3",
]

matrix_cols = [
    "period",
    "outcome",
] + [
    c for c in ordered_component_cols
    if c in matrix_df.columns
]

matrix_df = matrix_df[
    matrix_cols
]


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

detail_cols = [
    "period",
    "outcome",
    "model",
    "block",
    "component",
    "term",
    "partial_derivative_dP_dz",
    "abs_partial_derivative",
    "derivative_state",
    "direction",
    "se",
    "ci_low",
    "ci_high",
    "ci_contains_zero",
    "p",
    "p_lt_0_05",
]

coef[
    detail_cols
].to_csv(
    RESULTS
    / "75_phenotype_partial_derivatives.csv",
    index=False
)

block_df.to_csv(
    RESULTS
    / "75_phenotype_derivative_block_summary.csv",
    index=False
)

summary_df.to_csv(
    RESULTS
    / "75_phenotype_derivative_summary.csv",
    index=False
)

matrix_df.to_csv(
    RESULTS
    / "75_phenotype_derivative_matrix.csv",
    index=False
)

grid_df.to_csv(
    RESULTS
    / "75_phenotype_derivative_completeness.csv",
    index=False
)


# =============================================================================
# PRINT
# =============================================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

pd.set_option(
    "display.max_colwidth",
    120
)


print("=" * 105)
print(
    "SCRIPT 75 — PHENOTYPE PARTIAL-DERIVATIVE AUDIT"
)
print("=" * 105)

print()
print(
    "Notebook test: for every learned phenotype function, compute dP/dz_i."
)

print(
    "Because Script 72 uses a linear Gaussian svyglm, dP/dz_i = fitted beta_i."
)

print(
    "No correlation coefficients are used."
)


# =============================================================================
# PRINT EACH PHENOTYPE FUNCTION
# =============================================================================

for (
    period,
    outcome
), g in coef.groupby(
    ["period", "outcome"],
    dropna=False
):

    print()
    print("=" * 105)
    print(
        f"PERIOD: {period}    PHENOTYPE: {outcome}"
    )
    print("=" * 105)

    g = g.copy()

    order_map = {
        t: i
        for i, t in enumerate(
            TERMS
        )
    }

    g["_order"] = g[
        "term"
    ].map(order_map)

    g = g.sort_values(
        "_order"
    )

    for _, r in g.iterrows():

        print(
            f"dP/d{r['component']:<12s} "
            f"= {r['partial_derivative_dP_dz']:+.10f}   "
            f"{r['derivative_state']:<7s}   "
            f"95% CI [{r['ci_low']:+.10f}, {r['ci_high']:+.10f}]   "
            f"p={r['p']:.6g}"
        )

    zeros = g.loc[
        g["derivative_state"].eq(
            "ZERO"
        ),
        "component"
    ].tolist()

    print()

    if len(zeros) == 0:
        print(
            "ZERO DERIVATIVES: NONE"
        )
    else:
        print(
            "ZERO DERIVATIVES: "
            + ", ".join(zeros)
        )


# =============================================================================
# PRINT BLOCK SUMMARY
# =============================================================================

print()
print("=" * 105)
print(
    "BLOCK-LEVEL FUNCTIONAL DEPENDENCE"
)
print("=" * 105)

print(
    block_df.sort_values(
        [
            "period",
            "outcome",
            "block",
        ]
    ).to_string(
        index=False
    )
)


# =============================================================================
# PRINT OVERALL SUMMARY
# =============================================================================

print()
print("=" * 105)
print(
    "PHENOTYPE-LEVEL SUMMARY"
)
print("=" * 105)

print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# COMPLETENESS
# =============================================================================

print()
print("=" * 105)
print(
    "DERIVATIVE GRID COMPLETENESS"
)
print("=" * 105)

print(
    grid_df.to_string(
        index=False
    )
)


# =============================================================================
# INTERPRETATION
# =============================================================================

print()
print("=" * 105)
print(
    "INTERPRETATION RULE"
)
print("=" * 105)

print(
r"""
For the current Script-72 phenotype model:

    P_hat = beta0 + sum_i beta_i z_i + covariates

therefore:

    dP_hat/dz_i = beta_i

Hence:

    dP_hat/dz_i = 0
        -> the fitted phenotype function does not directly depend on z_i.

    dP_hat/dz_i != 0
        -> z_i directly enters the fitted phenotype function.

This is the Prof. Saeed notebook test at the PHENOTYPE FUNCTION level.

The confidence interval and p-value are reported separately because:

    "estimated derivative is nonzero"

and

    "population evidence excludes zero"

are NOT the same claim.

This script does not calculate Pearson or Spearman correlation.
"""
)


print("=" * 105)
print(
    "OUTPUTS"
)
print("=" * 105)

for name in [
    "75_phenotype_partial_derivatives.csv",
    "75_phenotype_derivative_block_summary.csv",
    "75_phenotype_derivative_summary.csv",
    "75_phenotype_derivative_matrix.csv",
    "75_phenotype_derivative_completeness.csv",
]:
    print(
        RESULTS / name
    )

print("=" * 105)
