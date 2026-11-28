
#!/usr/bin/env python3
"""
122 | Statistical robustness and influence audit

1. Weighted regression leverage
2. Cook's distance
3. DFBETAS
4. Residual and collinearity diagnostics
5. Influential-participant sensitivity
6. Stratified leave-one-PSU-out sensitivity
7. Physiological transformation stability
8. Consolidation of existing survey inference

Classical influence statistics are descriptive WLS diagnostics,
not design-adjusted hypothesis tests.
"""

from pathlib import Path
import argparse
import json
import ast

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.linalg import subspace_angles
from tqdm.auto import tqdm


# ============================================================
# Configuration
# ============================================================

EDA = Path(__file__).resolve().parent
ROOT = EDA.parent
R = EDA / "Results"

INPUT = R / "72_shared_private_model_input.csv"
TRANSFORM = R / "72_shared_private_frozen_transform.npz"

OUT = R / "122_statistical_robustness"

RAW = [
    "LBXHGB",
    "LBXRBCSI",
    "LBXMCVSI",
    "LBXRDW",
    "LBXGH",
    "LBXGLU",
    "LOG_IN",
]

Z = [
    "SP_SHARED1",
    "SP_SHARED2",
    "SP_DISCORD1",
    "SP_DISCORD2",
    "SP_A_PRIVATE3",
    "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]

GROUPS = {
    "Shared": [0, 1],
    "Discordant": [2, 3],
    "Haematology-private": [4, 5],
    "Glycaemia-private": [6],
}

COV_NUM = [
    "RIDAGEYR",
    "INDFMPIR",
    "BMXBMI",
    "EGFR_2021",
]

COV_CAT = [
    "RIAGENDR",
    "RACE",
    "EDUC3",
    "SMOKING3",
]

OUTCOMES = {
    "PHQ_total": "PHQ9_TOTAL",
    "Somatic": "SOMATIC_SCORE",
    "CogAff": "COGAFF_SUM",
}

PERIODS = [
    "2005-2008",
    "2009-2018",
    "2021-2023",
]

METHOD_COLORS = {
    "Scalar": "#AC583C",
    "ScalarInteraction": "#C4A16A",
    "Continuous2": "#92999E",
    "Raw7": "#282E33",
    "HG7": "#146F69",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.dpi": 600,
})


# ============================================================
# Mathematical utilities
# ============================================================

def wm(x, w):
    return np.average(x, axis=0, weights=w)


def mse(y, pred, w):
    return float(wm((y - pred) ** 2, w))


def weighted_corr(x, y, w):

    a = x - wm(x, w)
    b = y - wm(y, w)

    denominator = np.sqrt(
        wm(a ** 2, w) * wm(b ** 2, w)
    )

    if denominator <= 0:
        return np.nan

    return float(wm(a * b, w) / denominator)


def wquantile(x, w, q):

    order = np.argsort(x)

    x = np.asarray(x)[order]
    w = np.asarray(w)[order]

    cdf = (
        np.cumsum(w) - 0.5 * w
    ) / w.sum()

    return np.interp(q, cdf, x)


def weighted_standardize(x, w):

    mu = wm(x, w)

    sd = np.sqrt(
        wm((x - mu) ** 2, w)
    )

    if np.any(sd <= 1e-12):
        raise ValueError(
            "Constant predictor in regression design"
        )

    return (x - mu) / sd


def top_indices(x, fraction):

    k = max(
        1,
        int(np.ceil(len(x) * fraction)),
    )

    return np.argsort(
        -np.asarray(x),
        kind="stable",
    )[:k]


# ============================================================
# Input preparation
# ============================================================

def load_data():

    d = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        *RAW,
        *Z,
        *COV_NUM,
        *COV_CAT,
        *OUTCOMES.values(),
        "SURVEY_WT",
        "STRATUM",
        "PSU",
        "CYCLE",
        "PERIOD",
        "ELIGIBLE_ADULT_NONPREG",
    ]

    missing = set(required) - set(d.columns)

    if missing:
        raise KeyError(
            f"Missing columns: {sorted(missing)}"
        )

    numeric = [
        *RAW,
        *Z,
        *COV_NUM,
        *OUTCOMES.values(),
        "SURVEY_WT",
        "ELIGIBLE_ADULT_NONPREG",
        "RIAGENDR",
    ]

    for c in numeric:

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce",
        )

    d = d.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Operational burden definitions from the manuscript.
    hb_threshold = d.RIAGENDR.map({
        1: 13.0,
        2: 12.0,
    })

    d["SCALAR_A"] = np.maximum(
        0,
        hb_threshold - d.LBXHGB,
    )

    d["SCALAR_G"] = np.maximum(
        0,
        d.LBXGH - 5.7,
    )

    d["SCALAR_AG"] = (
        d.SCALAR_A * d.SCALAR_G
    )

    # Cycle-specific design identifiers prevent
    # accidental pooling of identically numbered PSUs.
    valid = d[
        ["CYCLE", "STRATUM", "PSU"]
    ].notna().all(axis=1)

    d["_STRATUM"] = (
        d.CYCLE.astype(str)
        + ":"
        + d.STRATUM.astype(str)
    ).where(valid)

    d["_PSU"] = (
        d["_STRATUM"]
        + ":"
        + d.PSU.astype(str)
    ).where(valid)

    return d


def matched_sample(d, period, outcome):

    full = d.loc[
        d.PERIOD.eq(period)
    ].copy()

    required = [
        *RAW,
        *Z,
        *COV_NUM,
        *COV_CAT,
        "SCALAR_A",
        "SCALAR_G",
        "SCALAR_AG",
        outcome,
        "SURVEY_WT",
        "_STRATUM",
        "_PSU",
    ]

    ok = (
        full.ELIGIBLE_ADULT_NONPREG.eq(1)
        & full.SURVEY_WT.gt(0)
        & full[required].notna().all(axis=1)
    )

    q = full.loc[ok].reset_index(drop=True)

    if len(q) < 100:
        raise RuntimeError(
            f"Insufficient matched sample: "
            f"{period}, {outcome}"
        )

    return q, REGRESSION_DESIGN_FRAMES.get(period, full)


def covariate_design(q, w):

    cat = COV_CAT.copy()

    if q.CYCLE.nunique() > 1:
        cat.append("CYCLE")

    dummy = pd.get_dummies(
        q[cat].astype(str),
        drop_first=True,
        dtype=float,
    )

    frame = pd.concat([
        q[COV_NUM],
        dummy,
    ], axis=1)

    values = weighted_standardize(
        frame.to_numpy(float),
        w,
    )

    return values, frame.columns.tolist()


def make_designs(q, w):

    cov, names = covariate_design(q, w)

    predictors = {
        "Scalar": [
            "SCALAR_A",
            "SCALAR_G",
        ],
        "ScalarInteraction": [
            "SCALAR_A",
            "SCALAR_G",
            "SCALAR_AG",
        ],
        "Continuous2": [
            "LBXHGB",
            "LBXGH",
        ],
        "Raw7": RAW,
        "HG7": Z,
    }

    designs = {}

    for method, columns in predictors.items():

        physiology = weighted_standardize(
            q[columns].to_numpy(float),
            w,
        )

        x = np.column_stack([
            np.ones(len(q)),
            cov,
            physiology,
        ])

        designs[method] = {
            "x": x,
            "names": [
                "Intercept",
                *names,
                *columns,
            ],
            "physiology": columns,
        }

    return designs


# ============================================================
# Weighted least squares and influence
# ============================================================

def fit_wls(x, y, w):

    keep = w > 0

    a = x[keep] * np.sqrt(
        w[keep, None]
    )

    b = y[keep] * np.sqrt(w[keep])

    beta, _, rank, _ = np.linalg.lstsq(
        a, b, rcond=1e-12
    )

    if rank != x.shape[1]:

        raise RuntimeError(
            f"Rank-deficient refit: "
            f"{rank}/{x.shape[1]}"
        )

    return beta


def influence(x, y, w):

    n, p = x.shape

    a = x * np.sqrt(w[:, None])

    beta = fit_wls(x, y, w)

    pred = x @ beta

    residual = y - pred

    rw = np.sqrt(w) * residual

    q, r = np.linalg.qr(
        a,
        mode="reduced",
    )

    leverage = np.sum(q ** 2, axis=1)

    np.testing.assert_allclose(
        leverage.sum(),
        p,
        atol=1e-7,
    )

    if np.any(leverage >= 1 - 1e-10):
        raise RuntimeError(
            "Unit leverage encountered; "
            "single-case deletion undefined."
        )

    r_inverse = np.linalg.inv(r)

    inverse = r_inverse @ r_inverse.T

    sse = np.sum(rw ** 2)

    sigma2 = sse / (n - p)

    one_minus_h = 1 - leverage

    student_internal = rw / np.sqrt(
        sigma2 * one_minus_h
    )

    # Exact WLS case-deletion coefficient change.
    delta = (
        a @ inverse
    ) * (
        rw / one_minus_h
    )[:, None]

    sse_deleted = (
        sse
        - rw ** 2 / one_minus_h
    )

    variance_deleted = (
        sse_deleted / (n - p - 1)
    )

    if np.any(variance_deleted <= 0):
        raise RuntimeError(
            "Invalid deleted residual variance"
        )

    student_external = rw / np.sqrt(
        variance_deleted * one_minus_h
    )

    cooks = (
        student_internal ** 2
        * leverage
        / (p * one_minus_h)
    )

    dfbetas = delta / np.sqrt(
        variance_deleted[:, None]
        * np.diag(inverse)[None, :]
    )

    # VIFs for weighted-standardized predictor columns.
    # Dummy-level VIFs are not generalized factor GVIFs.
    vif = (
        np.diag(a.T @ a)
        * np.diag(inverse)
    )

    # Verify the analytic deletion identity against
    # explicit refits for the most influential cases.
    check = top_indices(
        cooks,
        min(3 / n, 1),
    )

    for i in check:

        ww = w.copy()
        ww[i] = 0

        explicit = fit_wls(x, y, ww)

        np.testing.assert_allclose(
            beta - explicit,
            delta[i],
            atol=1e-7,
            rtol=1e-5,
        )

    return {
        "beta": beta,
        "pred": pred,
        "residual": residual,
        "leverage": leverage,
        "cooks": cooks,
        "dfbetas": dfbetas,
        "student": student_external,
        "vif": vif,
        "condition": np.linalg.cond(a),
        "sigma2": sigma2,
    }


# ============================================================
# Residual diagnostics
# ============================================================

def residual_summary(y, w, fit):

    r = fit["residual"]
    pred = fit["pred"]

    mu = wm(r, w)

    sd = np.sqrt(
        wm((r - mu) ** 2, w)
    )

    quantiles = wquantile(
        r,
        w,
        [0.025, 0.25, 0.5, 0.75, 0.975],
    )

    return {
        "weighted_residual_mean": mu,
        "weighted_residual_sd": sd,
        "weighted_residual_skewness":
            wm(((r - mu) / sd) ** 3, w),
        "weighted_residual_excess_kurtosis":
            wm(((r - mu) / sd) ** 4, w) - 3,
        "residual_q025": quantiles[0],
        "residual_q25": quantiles[1],
        "residual_median": quantiles[2],
        "residual_q75": quantiles[3],
        "residual_q975": quantiles[4],
        "corr_fitted_squared_residual":
            weighted_corr(pred, r ** 2, w),
        "corr_fitted_absolute_residual":
            weighted_corr(pred, np.abs(r), w),
        "fraction_predictions_below_zero":
            wm(pred < 0, w),
        "weighted_mse": mse(y, pred, w),
        "weighted_mae": wm(np.abs(r), w),
    }


# ============================================================
# Stratified PSU deletion weights
# ============================================================

def psu_deletions(q, full, w):

    # Use the available pre-domain survey frame
    # rather than deriving strata only from complete cases.
    frame = full[
        ["_STRATUM", "_PSU"]
    ].dropna().drop_duplicates()

    counts = frame.groupby(
        "_STRATUM"
    )["_PSU"].nunique()

    for stratum, cluster in frame.itertuples(
        index=False,
        name=None,
    ):

        m = int(counts.loc[stratum])

        if m < 2:

            yield stratum, cluster, None
            continue

        in_stratum = (
            q["_STRATUM"].to_numpy()
            == stratum
        )

        in_psu = (
            q["_PSU"].to_numpy()
            == cluster
        )

        ww = w.copy()

        ww[in_stratum] *= m / (m - 1)

        ww[in_psu] = 0

        yield stratum, cluster, ww


# ============================================================
# Refit sensitivity
# ============================================================

def sensitivity_record(
    x, y, w, ww, fit, names,
    physiology, metadata,
):

    try:

        beta = fit_wls(x, y, ww)

    except RuntimeError as exc:

        return {
            **metadata,
            "status": "rank_limited",
            "reason": str(exc),
        }, []

    pred = x @ beta

    base = fit["beta"]

    change = beta - base

    ix = [
        names.index(c)
        for c in physiology
    ]

    scale = np.sqrt(
        wm((y - wm(y, w)) ** 2, w)
    )

    result = {
        **metadata,
        "status": "completed",
        "n_retained": int(np.sum(ww > 0)),
        "original_sample_mse_change":
            mse(y, pred, w)
            - mse(y, fit["pred"], w),
        "prediction_shift_rmse":
            np.sqrt(
                wm(
                    (pred - fit["pred"]) ** 2,
                    w,
                )
            ),
        "max_physiological_beta_change":
            np.max(np.abs(change[ix])),
        "max_beta_change_in_outcome_SD":
            np.max(np.abs(change[ix])) / scale,
    }

    coefficients = []

    for i in ix:

        coefficients.append({
            **metadata,
            "term": names[i],
            "original_beta": base[i],
            "refitted_beta": beta[i],
            "delta_beta": change[i],
            "sign_reversal":
                bool(base[i] * beta[i] < 0),
        })

    return result, coefficients


# ============================================================
# Main regression audit
# ============================================================

REGRESSION_DESIGN_FRAMES = {}


def load_regression_data():

    import importlib.util
    import sys

    source = EDA / "103_jive_ajive_benchmark.py"

    spec = importlib.util.spec_from_file_location(
        "s103_for_122", source
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    d, _ = module.load_matched()
    d = d.copy()

    cycles = {
        "0506": "D",
        "0708": "E",
        "0910": "F",
        "1112": "G",
        "1314": "H",
        "1516": "I",
        "1718": "J",
        "2123": "L",
    }

    d["PERIOD"] = d.SPLIT.map({
        "train_2005-2008": "2005-2008",
        "test_2009-2018": "2009-2018",
        "validate_2021-2023": "2021-2023",
    })

    expected = {
        "2005-2008": 3815,
        "2009-2018": 9893,
        "2021-2023": 2247,
    }

    assert d.PERIOD.value_counts().to_dict() == expected

    frames = []

    for cycle, suffix in cycles.items():

        folder = ROOT / "Data" / (
            "NHANES"
            if cycle in ("0506", "0708")
            else "NHANES_Transfer"
        ) / cycle

        candidates = [
            folder / f"DEMO_{suffix}.XPT",
            folder / f"DEMO_{suffix}.xpt",
        ]

        path = next(
            (p for p in candidates if p.exists()),
            None,
        )

        if path is None:
            raise FileNotFoundError(candidates[0])

        demo = pd.read_sas(
            path, format="xport"
        )[["SEQN", "SDMVSTRA", "SDMVPSU"]].copy()

        if demo.isna().any().any():
            raise ValueError(
                f"Missing design identifiers: {cycle}"
            )

        demo["CYCLE"] = cycle

        demo["_STRATUM"] = (
            cycle + ":"
            + demo.SDMVSTRA.astype(int).astype(str)
        )

        demo["_PSU"] = (
            demo["_STRATUM"] + ":"
            + demo.SDMVPSU.astype(int).astype(str)
        )

        frames.append(demo)

    design = pd.concat(frames, ignore_index=True)

    d = d.merge(
        design,
        on=["SEQN", "CYCLE"],
        how="left",
        validate="one_to_one",
    )

    assert d[["_STRATUM", "_PSU"]].notna().all().all()

    d["STRATUM"] = d.SDMVSTRA
    d["PSU"] = d.SDMVPSU

    d["SCALAR_A"] = np.maximum(
        0,
        d.RIAGENDR.map({1: 13.0, 2: 12.0})
        - d.LBXHGB,
    )

    d["SCALAR_G"] = np.maximum(
        0, d.LBXGH - 5.7
    )

    d["SCALAR_AG"] = (
        d.SCALAR_A * d.SCALAR_G
    )

    for period in PERIODS:

        used_cycles = d.loc[
            d.PERIOD.eq(period), "CYCLE"
        ].unique()

        REGRESSION_DESIGN_FRAMES[period] = (
            design.loc[
                design.CYCLE.isin(used_cycles),
                ["_STRATUM", "_PSU"],
            ].drop_duplicates()
        )

    print("\nVerified regression cohort:")
    print(d.PERIOD.value_counts().to_string())

    return d

def regression_audit(d, fraction):

    diagnostics = []
    coefficient_rows = []
    residual_rows = []
    sensitivity_rows = []
    sensitivity_coefs = []
    invariance_rows = []
    profile_rows = []

    combinations = [
        (period, name, outcome)
        for period in PERIODS
        for name, outcome in OUTCOMES.items()
    ]

    for period, name, outcome in tqdm(
        combinations,
        desc="Regression audit",
    ):

        q, full = matched_sample(
            d, period, outcome
        )

        y = q[outcome].to_numpy(float)

        original_w = q.SURVEY_WT.to_numpy(float)

        # Normalization does not change WLS estimates.
        w = original_w / original_w.mean()

        n = len(q)

        designs = make_designs(q, w)

        fits = {}

        common_delete = {
            "high_leverage": set(),
            "high_cook": set(),
            "high_dfbeta": set(),
        }

        for method, design in designs.items():

            x = design["x"]

            f = influence(x, y, w)

            fits[method] = f

            p = x.shape[1]

            h = f["leverage"]
            cook = f["cooks"]

            max_dfbeta = np.max(
                np.abs(f["dfbetas"]),
                axis=1,
            )

            metadata = {
                "period": period,
                "outcome": name,
                "method": method,
            }

            diagnostics.append({
                **metadata,
                "n": n,
                "parameters": p,
                "max_leverage": h.max(),
                "mean_leverage": h.mean(),
                "n_leverage_above_2p_n":
                    int(np.sum(h > 2 * p / n)),
                "max_cooks_distance": cook.max(),
                "n_cook_above_4_n":
                    int(np.sum(cook > 4 / n)),
                "max_absolute_DFBETAS":
                    max_dfbeta.max(),
                "n_dfbeta_above_2_sqrt_n":
                    int(np.sum(
                        max_dfbeta > 2 / np.sqrt(n)
                    )),
                "n_studentized_residual_above_3":
                    int(np.sum(
                        np.abs(f["student"]) > 3
                    )),
                "design_condition_number":
                    f["condition"],
                "max_nonintercept_VIF":
                    f["vif"][1:].max(),
            })

            residual_rows.append({
                **metadata,
                **residual_summary(y, w, f),
            })

            for j, term in enumerate(
                design["names"]
            ):

                coefficient_rows.append({
                    **metadata,
                    "term": term,
                    "standardized_predictor_beta":
                        f["beta"][j],
                    "VIF": f["vif"][j]
                        if j else np.nan,
                    "maximum_absolute_DFBETAS":
                        np.abs(
                            f["dfbetas"][:, j]
                        ).max(),
                    "n_DFBETAS_above_threshold":
                        int(np.sum(
                            np.abs(
                                f["dfbetas"][:, j]
                            )
                            > 2 / np.sqrt(n)
                        )),
                })

            # Common deletion sets preserve comparability
            # across representations. Their union may
            # exceed the per-model deletion fraction.
            for key, value in [
                ("high_leverage", h),
                ("high_cook", cook),
                ("high_dfbeta", max_dfbeta),
            ]:

                common_delete[key].update(
                    top_indices(value, fraction)
                )

            # Aggregate residual profiles by fitted decile.
            order = np.argsort(f["pred"])

            bins = np.array_split(order, 10)

            for k, idx in enumerate(bins):

                profile_rows.append({
                    **metadata,
                    "fitted_decile": k + 1,
                    "weighted_fitted_mean":
                        wm(f["pred"][idx], w[idx]),
                    "weighted_residual_mean":
                        wm(f["residual"][idx], w[idx]),
                    "weighted_residual_RMSE":
                        np.sqrt(wm(
                            f["residual"][idx] ** 2,
                            w[idx],
                        )),
                })

        # Exact full-rank affine invariance.
        raw_fit = fits["Raw7"]
        hg_fit = fits["HG7"]

        checks = {}

        for metric in [
            "pred",
            "leverage",
            "cooks",
            "student",
        ]:

            diff = np.max(np.abs(
                raw_fit[metric]
                - hg_fit[metric]
            ))

            checks[metric] = float(diff)

            np.testing.assert_allclose(
                raw_fit[metric],
                hg_fit[metric],
                atol=1e-7,
                rtol=1e-6,
            )

        invariance_rows.append({
            "period": period,
            "outcome": name,
            **checks,
            "passed": True,
        })

        deletions = []

        for key, indices in common_delete.items():

            ww = w.copy()

            ww[list(indices)] = 0

            deletions.append((
                key,
                ww,
                {
                    "deleted_n": len(indices),
                    "stratum": "",
                    "psu": "",
                },
            ))

        for stratum, cluster, ww in psu_deletions(
            q, full, w
        ):

            if ww is None:

                sensitivity_rows.append({
                    "period": period,
                    "outcome": name,
                    "sensitivity": "delete_PSU",
                    "stratum": stratum,
                    "psu": cluster,
                    "status": "singleton_frame",
                })

                continue

            deletions.append((
                "delete_PSU",
                ww,
                {
                    "stratum": stratum,
                    "psu": cluster,
                    "deleted_n":
                        int(np.sum(ww == 0)),
                },
            ))

        for kind, ww, extra in tqdm(
            deletions,
            desc=f"{period} / {name}",
            leave=False,
        ):

            for method, design in designs.items():

                result, coeffs = sensitivity_record(
                    design["x"],
                    y,
                    w,
                    ww,
                    fits[method],
                    design["names"],
                    design["physiology"],
                    {
                        "period": period,
                        "outcome": name,
                        "method": method,
                        "sensitivity": kind,
                        **extra,
                    },
                )

                sensitivity_rows.append(result)

                sensitivity_coefs.extend(coeffs)

    return {
        "regression_diagnostics":
            pd.DataFrame(diagnostics),
        "coefficient_influence":
            pd.DataFrame(coefficient_rows),
        "residual_diagnostics":
            pd.DataFrame(residual_rows),
        "residual_profiles":
            pd.DataFrame(profile_rows),
        "deletion_sensitivity":
            pd.DataFrame(sensitivity_rows),
        "deletion_coefficients":
            pd.DataFrame(sensitivity_coefs),
        "full_rank_invariance":
            pd.DataFrame(invariance_rows),
    }


# ============================================================
# Reuse exact Script72 CCA implementation
# ============================================================

def load_cca_functions():

    source = (
        EDA /
        "72_shared_private_decomposition_and_phenotype.py"
    ).read_text(encoding="utf-8")

    tree = ast.parse(source)

    required = {
        "wnorm",
        "weighted_mean_sd",
        "wcov",
        "invsqrt",
        "fit_weighted_cca",
        "transform_block",
    }

    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in required
    ]

    found = {node.name for node in nodes}

    if found != required:

        raise RuntimeError(
            f"Missing Script72 functions: "
            f"{required - found}"
        )

    namespace = {
        "np": np,
        "pd": pd,
    }

    module = ast.Module(
        body=nodes,
        type_ignores=[],
    )

    exec(
        compile(
            module,
            "<script72_functions>",
            "exec",
        ),
        namespace,
    )

    return namespace


def similarity(a, b, w):

    a = (
        a - wm(a, w)
    ) * np.sqrt(w[:, None])

    b = (
        b - wm(b, w)
    ) * np.sqrt(w[:, None])

    if (
        np.linalg.matrix_rank(a) != a.shape[1]
        or np.linalg.matrix_rank(b) != b.shape[1]
    ):
        raise RuntimeError(
            "Rank-deficient coordinate group"
        )

    angles = subspace_angles(a, b)

    return float(
        np.mean(np.cos(angles) ** 2)
    )


# ============================================================
# Physiological transformation stability
# ============================================================

def structural_audit(d, fraction):

    full = d.loc[
        d.PERIOD.eq("2005-2008")
    ].copy()

    required = [
        *RAW,
        *Z,
        "_STRATUM",
        "_PSU",
        "SURVEY_WT",
    ]

    ok = (
        full.ELIGIBLE_ADULT_NONPREG.eq(1)
        & full.SURVEY_WT.gt(1e-12)
        & full[required].notna().all(axis=1)
    )

    q = full.loc[ok].reset_index(drop=True)

    x = q[RAW].to_numpy(float)
    z = q[Z].to_numpy(float)

    w = q.SURVEY_WT.to_numpy(float)

    w = w / w.mean()

    helpers = load_cca_functions()

    cca = helpers["fit_weighted_cca"]
    transform = helpers["transform_block"]

    frozen = dict(np.load(TRANSFORM))

    if int(
        frozen["shared_rank"].ravel()[0]
    ) != 2:

        raise RuntimeError(
            "Unexpected frozen shared rank"
        )

    baseline, _, _ = transform(
        x[:, :4],
        x[:, 4:],
        frozen,
        2,
    )

    np.testing.assert_allclose(
        baseline[Z].to_numpy(float),
        z,
        atol=1e-7,
        rtol=1e-6,
    )

    # Independently reproduce the original discovery fit.
    reconstructed_fit = cca(
        x[:, :4],
        x[:, 4:],
        w,
    )

    np.testing.assert_allclose(
        reconstructed_fit["rho"],
        frozen["rho"],
        atol=1e-7,
    )

    # Outcome-independent leverage in physiological space.
    standardized = weighted_standardize(x, w)

    a = np.column_stack([
        np.ones(len(q)),
        standardized,
    ]) * np.sqrt(w[:, None])

    qr, _ = np.linalg.qr(
        a,
        mode="reduced",
    )

    h = np.sum(qr ** 2, axis=1)

    perturbations = []

    ww = w.copy()

    ww[top_indices(h, fraction)] = 0

    perturbations.append((
        "high_physiological_leverage",
        "",
        "",
        ww,
    ))

    # Distinguish unusual physiology from large
    # survey weights as sources of weighted influence.
    ww = w.copy()

    ww[top_indices(w, fraction)] = 0

    perturbations.append((
        "high_survey_weight",
        "",
        "",
        ww,
    ))

    perturbations.extend([
        ("delete_PSU", s, p, ww)
        for s, p, ww in psu_deletions(
            q, full, w
        )
    ])

    records = []

    for kind, stratum, cluster, ww in tqdm(
        perturbations,
        desc="Structural stability",
    ):

        metadata = {
            "perturbation": kind,
            "stratum": stratum,
            "psu": cluster,
        }

        if ww is None:

            records.append({
                **metadata,
                "status": "singleton_frame",
            })

            continue

        keep = ww > 0

        try:

            fitted = cca(
                x[keep, :4],
                x[keep, 4:],
                ww[keep],
            )

            projected, _, _ = transform(
                x[:, :4],
                x[:, 4:],
                fitted,
                2,
            )

            zz = projected[Z].to_numpy(float)

            for group, indices in GROUPS.items():

                records.append({
                    **metadata,
                    "status": "completed",
                    "group": group,
                    "n_retained":
                        int(keep.sum()),
                    "subspace_similarity":
                        similarity(
                            z[:, indices],
                            zz[:, indices],
                            w,
                        ),
                    "rho1": fitted["rho"][0],
                    "rho2": fitted["rho"][1],
                    "rho3": fitted["rho"][2],
                    "rho1_change":
                        fitted["rho"][0]
                        - frozen["rho"][0],
                    "rho2_change":
                        fitted["rho"][1]
                        - frozen["rho"][1],
                    "rho3_change":
                        fitted["rho"][2]
                        - frozen["rho"][2],
                })

        except (ValueError, RuntimeError,
                np.linalg.LinAlgError) as exc:

            records.append({
                **metadata,
                "status": "failed",
                "reason": str(exc),
            })

    return pd.DataFrame(records)


# ============================================================
# Consolidate existing survey-adjusted evidence
# ============================================================

def collect_existing_inference():

    sources = {
        "scalar_coefficients":
            ROOT / "Results" /
            "26_design_based_coefficients.csv",

        "scalar_incremental_tests":
            ROOT / "Results" /
            "27_incremental_information_tests.csv",

        "comparator_wald_tests":
            ROOT / "Results" /
            "48_comparator_sensitivity_tests.csv",

        "HG_block_wald_tests":
            R / "72_shared_private_block_tests.csv",

        "HG_coefficients":
            R / "72_shared_private_coefficients.csv",

        "raw_gradient_wald":
            R / "77_raw_block_gradient_wald.csv",

        "component_ablation":
            R / "77_component_and_block_ablation.csv",

        "temporal_interactions":
            R / "86_component_time_interaction_audit.csv",
    }

    inventory = []

    for name, path in sources.items():

        exists = path.exists()

        inventory.append({
            "evidence": name,
            "source": str(path.relative_to(ROOT)),
            "available": exists,
        })

        if exists:

            table = pd.read_csv(path)

            table.to_csv(
                OUT / f"122_existing_{name}.csv",
                index=False,
            )

    return pd.DataFrame(inventory)


# ============================================================
# Supplementary publication figures
# ============================================================

def save_figure(fig, name):

    fig.savefig(
        OUT / f"{name}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def figures(results, structure):

    df = results["regression_diagnostics"]

    q = df.loc[
        df.outcome.eq("PHQ_total")
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.1, 2.8),
        layout="constrained",
    )

    metrics = [
        ("max_leverage", "Maximum leverage"),
        ("max_cooks_distance", "Maximum Cook's distance"),
        ("max_absolute_DFBETAS", "Maximum |DFBETAS|"),
    ]

    methods = list(METHOD_COLORS)

    for ax, (metric, label), letter in zip(
        axes,
        metrics,
        "abc",
    ):

        for j, method in enumerate(methods):

            sub = (
                q.loc[q.method.eq(method)]
                .set_index("period")
                .loc[PERIODS]
            )

            ax.plot(
                range(3),
                sub[metric],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                color=METHOD_COLORS[method],
                label=method,
            )

        ax.set_xticks(
            range(3),
            ["05–08", "09–18", "21–23"],
        )

        ax.set_title(
            f"{letter}  {label}",
            loc="left",
        )

        ax.set_xlabel("NHANES period")

    axes[0].legend(
        frameon=False,
        fontsize=6.5,
    )

    save_figure(
        fig,
        "122_influence_diagnostics",
    )

    # Residual profile: mean residual and RMS error
    # by fitted-score decile.
    profile = results["residual_profiles"]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.1, 4.6),
        layout="constrained",
    )

    for j, period in enumerate(PERIODS):

        data = profile.loc[
            profile.period.eq(period)
            & profile.outcome.eq("PHQ_total")
        ]

        for method in [
            "Scalar",
            "Raw7",
            "HG7",
        ]:

            sub = data.loc[
                data.method.eq(method)
            ].sort_values("fitted_decile")

            axes[0, j].plot(
                sub.fitted_decile,
                sub.weighted_residual_mean,
                color=METHOD_COLORS[method],
                marker="o",
                ms=2.5,
                label=method,
            )

            axes[1, j].plot(
                sub.fitted_decile,
                sub.weighted_residual_RMSE,
                color=METHOD_COLORS[method],
                marker="o",
                ms=2.5,
            )

        axes[0, j].axhline(
            0,
            color="#92999E",
            lw=0.7,
            ls="--",
        )

        axes[0, j].set_title(period)

        axes[1, j].set_xlabel(
            "Fitted-score decile"
        )

    axes[0, 0].set_ylabel("Mean residual")
    axes[1, 0].set_ylabel("Residual RMSE")

    axes[0, 0].legend(
        frameon=False,
        fontsize=7,
    )

    save_figure(
        fig,
        "122_residual_profiles",
    )

    # Structural robustness under PSU perturbation.
    sub = structure.loc[
        structure.status.eq("completed")
        & structure.perturbation.eq("delete_PSU")
    ]

    if not sub.empty:

        fig, ax = plt.subplots(
            figsize=(7.1, 2.8),
            layout="constrained",
        )

        colors = [
            "#146F69",
            "#AC583C",
            "#47677E",
            "#8A9D91",
        ]

        for i, (group, color) in enumerate(
            zip(GROUPS, colors)
        ):

            values = sub.loc[
                sub.group.eq(group),
                "subspace_similarity",
            ].to_numpy()

            if not len(values):
                continue

            ax.scatter(
                values,
                np.full(len(values), i),
                color=color,
                s=12,
                alpha=0.5,
            )

            ax.scatter(
                np.median(values),
                i,
                color=color,
                edgecolor="#282E33",
                marker="D",
                s=35,
                zorder=4,
            )

        ax.set_yticks(
            range(4),
            list(GROUPS),
        )

        ax.invert_yaxis()

        ax.set_xlim(0, 1.02)

        ax.set_xlabel(
            "Similarity to original discovery subspace"
        )

        ax.axvline(
            1,
            color="#92999E",
            lw=0.7,
            ls="--",
        )

        save_figure(
            fig,
            "122_structural_robustness",
        )


# ============================================================
# Readout and completion
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()

    if not 0 < args.fraction < 0.1:
        raise ValueError(
            "Deletion fraction must be between 0 and 0.1"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    completion = OUT / "122_completion.json"

    completion.write_text(
        json.dumps({"completed": False}, indent=2),
        encoding="utf-8",
    )

    print("Loading Script72 inputs...")

    d = load_data()

    results = regression_audit(
        load_regression_data(),
        args.fraction,
    )

    for name, frame in results.items():

        frame.to_csv(
            OUT / f"122_{name}.csv",
            index=False,
        )

    structure = structural_audit(
        d,
        args.fraction,
    )

    structure.to_csv(
        OUT / "122_structural_robustness.csv",
        index=False,
    )

    evidence = collect_existing_inference()

    evidence.to_csv(
        OUT / "122_existing_evidence_inventory.csv",
        index=False,
    )

    sensitivity = results["deletion_sensitivity"]

    completed = sensitivity.loc[
        sensitivity.status.eq("completed")
    ]

    summary = (
        completed.groupby([
            "period",
            "outcome",
            "method",
            "sensitivity",
        ])
        .agg(
            successful_refits=(
                "status", "size"
            ),
            maximum_prediction_shift=(
                "prediction_shift_rmse", "max"
            ),
            maximum_beta_change=(
                "max_physiological_beta_change",
                "max",
            ),
            maximum_original_sample_MSE_change=(
                "original_sample_mse_change",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT / "122_sensitivity_summary.csv",
        index=False,
    )

    valid = structure.loc[
        structure.status.eq("completed")
    ]

    structural_summary = (
        valid.groupby([
            "perturbation",
            "group",
        ])
        .subspace_similarity
        .agg(["count", "min", "median", "max"])
        .reset_index()
    )

    structural_summary.to_csv(
        OUT / "122_structural_summary.csv",
        index=False,
    )

    figures(results, structure)

    invariance = results["full_rank_invariance"]

    status = {
        "completed": True,
        "per_model_deletion_fraction": args.fraction,
        "regression_models":
            len(results["regression_diagnostics"]),
        "successful_refits": len(completed),
        "unsuccessful_or_limited_records":
            int(len(sensitivity) - len(completed)),
        "structural_failed_or_limited_records":
            int(structure.status.ne("completed").sum()),
        "full_rank_invariance_passed":
            bool(invariance.passed.all()),
        "existing_inference_files_found":
            int(evidence.available.sum()),
        "existing_inference_files_expected":
            len(evidence),
        "interpretation": [
            "Classical WLS influence statistics are descriptive.",
            "PSU deletions are sensitivity analyses, not confidence intervals.",
            "Regression coefficients use standardized predictors.",
            "Within-period fits are not frozen prediction models.",
            "Structural refits retain fixed shared rank two.",
            "Historical survey inference is copied, not recomputed.",
            "No causal or clinical robustness claim is established.",
        ],
    }

    completion.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    readout = [
        "# Script122 — Statistical robustness audit",
        "",
        "## Completion",
        "",
        "```text",
        json.dumps(status, indent=2),
        "```",
        "",
        "## Full-rank invariance",
        "",
        "```text",
        invariance.to_string(index=False),
        "```",
        "",
        "## Structural robustness",
        "",
        "```text",
        structural_summary.to_string(index=False),
        "```",
        "",
        "## Regression deletion sensitivity",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation boundaries",
        "",
        "Influence thresholds are screening heuristics.",
        "No observations are removed from the primary analysis.",
        "PSU perturbations use the available pre-domain frame.",
        "Residual diagnostics are descriptive and not formal",
        "survey-adjusted specification tests.",
    ]

    (OUT / "122_READOUT.md").write_text(
        "\n".join(readout),
        encoding="utf-8",
    )

    print()
    print("=" * 64)
    print("STATISTICAL ROBUSTNESS AUDIT COMPLETE")
    print("=" * 64)

    print(json.dumps(status, indent=2))

    print()
    print(structural_summary.to_string(index=False))

    print()
    print("Results:", OUT)


if __name__ == "__main__":

    main()
