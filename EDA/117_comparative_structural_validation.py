
#!/usr/bin/env python3
"""
117 | Comparative structural validation of the H/G coordinate engine.

Objectives
----------
1. Matched empirical structural rediscovery and conditioned nulls.
2. Controlled recovery under coupling, discordance, noise and shift.
3. Comparative Jacobian, conditioning and perturbation analysis.

Comparators
-----------
HG7
PCA7
BlockPCA7
JIVE7
AJIVE7_adaptive
RawX7

Design
------
- Identical samples and perturbations across methods.
- Common unweighted fitting for the comparative refit experiment.
- Original survey-weighted HG7 evidence remains unchanged.
- Existing native JIVE/AJIVE implementations are reused.
- Reduced-rank AJIVE fits are recorded, never padded.
- No phenotype information enters representation fitting.
- No biological, causal or clinical superiority claims.
- Existing research files are not overwritten.
"""

from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS"
):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import expm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.dont_write_bytecode = True

EDA = Path(__file__).resolve().parent
ROOT = EDA.parent

SEED = 117

METHODS = (
    "HG7",
    "PCA7",
    "BlockPCA7",
    "JIVE7",
    "AJIVE7_adaptive",
    "RawX7",
)

COHORTS = {
    "0506": ["0506"],
    "0708": ["0708"],
    "0918": ["0910", "1112", "1314", "1516", "1718"],
    "2123": ["2123"],
}

GROUP_DIMS = {
    "shared": 2,
    "discordant": 2,
    "H_private": 2,
    "G_private": 1,
}

SCENARIOS = {
    "reference": {
        "rho": 0.50,
        "noise": 0.20,
        "private": 1.0,
        "shift": "none",
    },
    "weak_coupling": {
        "rho": 0.15,
        "noise": 0.20,
        "private": 1.0,
        "shift": "none",
    },
    "strong_coupling": {
        "rho": 0.85,
        "noise": 0.20,
        "private": 1.0,
        "shift": "none",
    },
    "high_noise": {
        "rho": 0.50,
        "noise": 0.60,
        "private": 1.0,
        "shift": "none",
    },
    "private_dominant": {
        "rho": 0.50,
        "noise": 0.20,
        "private": 4.0,
        "shift": "none",
    },
    "population_shift": {
        "rho": 0.50,
        "noise": 0.20,
        "private": 1.0,
        "shift": "population",
    },
    "measurement_shift": {
        "rho": 0.50,
        "noise": 0.20,
        "private": 1.0,
        "shift": "measurement",
    },
    "no_sharing": {
        "rho": 0.0,
        "noise": 0.20,
        "private": 1.0,
        "shift": "none",
    },
}

NOISE_LEVELS = (0.01, 0.05, 0.10, 0.20)

REQUIRED_SCRIPTS = (
    "72_shared_private_decomposition_and_phenotype.py",
    "103_jive_ajive_benchmark.py",
    "104_representation_validation_gate.py",
    "105_specificity_and_transfer_audit.py",
    "112_finish_benchmarks.py",
    "114_structure_recovery.py",
    "116_7d_structural_validation.py",
)


# ============================================================================
# Utilities
# ============================================================================

def dump_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            default=lambda x: (
                x.item()
                if isinstance(x, np.generic)
                else str(x)
            ),
        ),
        encoding="utf-8",
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [EDA / name for name in REQUIRED_SCRIPTS]
    paths.append(Path(__file__))

    return {
        p.name: sha256(p)
        for p in paths
        if p.is_file()
    }


def append_csv(path, rows):
    if not rows:
        return

    frame = pd.DataFrame(rows)

    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def rng_for(*parts):
    return np.random.default_rng(
        np.random.SeedSequence([SEED, *parts])
    )


def affine_fit(x, y):
    xx = np.column_stack(
        [np.ones(len(x)), x]
    )

    return np.linalg.lstsq(
        xx,
        y,
        rcond=1e-12,
    )[0]


def affine_predict(x, coef):
    return coef[0] + x @ coef[1:]


def r2_matrix(y, pred, center=None):
    if center is None:
        center = y.mean(axis=0)

    numerator = np.sum((y - pred) ** 2)
    denominator = np.sum((y - center) ** 2)

    if denominator <= 1e-14:
        return np.nan

    return float(1.0 - numerator / denominator)


def heldout_r2(z_train, z_test, y_train, y_test):
    coef = affine_fit(z_train, y_train)

    return r2_matrix(
        y_test,
        affine_predict(z_test, coef),
    )


def paired_interval(values, rng, n_boot=1999):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan, np.nan

    indices = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    means = values[indices].mean(axis=1)

    return tuple(
        np.percentile(means, [2.5, 97.5])
    )


def subspace_basis(x):
    x = np.asarray(x, dtype=float)

    if x.shape[1] == 0:
        raise ValueError("Empty subspace")

    u, s, _ = np.linalg.svd(
        x,
        full_matrices=False,
    )

    tolerance = (
        max(x.shape)
        * np.finfo(float).eps
        * s[0]
    )

    rank = int(np.sum(s > tolerance))

    if rank != x.shape[1]:
        raise ValueError("Rank-deficient subspace")

    return u[:, :rank]


def similarity(a, b):
    if a.shape != b.shape:
        raise ValueError("Unmatched group dimensions")

    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)

    qa = subspace_basis(a)
    qb = subspace_basis(b)

    s = np.linalg.svd(
        qa.T @ qb,
        compute_uv=False,
    )

    return float(
        np.mean(np.clip(s, 0, 1) ** 2)
    )


def log_failure(out, phase, condition, rep, method):
    append_csv(
        out / "117_failures.csv",
        [{
            "phase": phase,
            "condition": condition,
            "replicate": rep,
            "method": method,
            "error": traceback.format_exc(),
        }],
    )


# ============================================================================
# Existing implementation adapters
# ============================================================================

def load_dependencies():

    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(
            name,
            path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    missing = [
        name for name in REQUIRED_SCRIPTS
        if not (EDA / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required scripts missing: "
            + ", ".join(missing)
        )

    s114 = load(
        "s114_for_117",
        EDA / "114_structure_recovery.py",
    )

    s104 = load(
        "s104_for_117",
        EDA / "104_representation_validation_gate.py",
    )

    s105 = load(
        "s105_for_117",
        EDA / "105_specificity_and_transfer_audit.py",
    )

    s112 = load(
        "s112_for_117",
        EDA / "112_finish_benchmarks.py",
    )

    s116 = load(
        "s116_for_117",
        EDA / "116_7d_structural_validation.py",
    )

    benchmark = s104.get_103()

    helpers = s114.cca_functions(
        EDA / "72_shared_private_decomposition_and_phenotype.py"
    )

    return {
        "s114": s114,
        "s104": s104,
        "s105": s105,
        "s112": s112,
        "s116": s116,
        "benchmark": benchmark,
        "helpers": helpers,
    }


# ============================================================================
# Representation fitting
# ============================================================================

def fit_representation(method, x, deps, args):

    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).all():
        raise ValueError("Nonfinite input measurements")

    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)

    groups = {}

    if method == "HG7":

        # Existing Script104 implementation:
        # fixed two-pair, full-dimensional H/G construction.
        # Unweighted here to match the comparator fitting protocol.

        matrix, mean, rho = deps["s104"].fit_hg_cca(
            x,
            np.ones(len(x)),
        )

        scores = (x - mean) @ matrix

        groups = {
            "shared": scores[:, :2],
            "discordant": scores[:, 2:4],
            "H_private": scores[:, 4:6],
            "G_private": scores[:, 6:7],
        }

    elif method == "PCA7":

        _, _, vt = np.linalg.svd(
            xs,
            full_matrices=False,
        )

        scores = xs @ vt.T

        # Leading-variance comparator only.
        groups = {
            "shared": scores[:, :2],
        }

    elif method == "BlockPCA7":

        _, _, vh = np.linalg.svd(
            xs[:, :4],
            full_matrices=False,
        )

        _, _, vg = np.linalg.svd(
            xs[:, 4:],
            full_matrices=False,
        )

        zh = xs[:, :4] @ vh.T
        zg = xs[:, 4:] @ vg.T

        scores = np.column_stack([zh, zg])

        groups = {
            "H_private": zh[:, :2],
            "G_private": zg[:, :1],
        }

    elif method == "RawX7":

        scores = xs

        # Raw markers provide an information and
        # sensitivity control. No arbitrary latent
        # group labels are assigned.

    elif method in ("JIVE7", "AJIVE7_adaptive"):

        fitted = deps["s114"].fit_method(
            method,
            x,
            deps["helpers"],
            deps["benchmark"],
            deps["s112"],
            args.rscript,
            args.timeout,
        )

        transformed = fitted(x)
        scores = transformed["all"]

        for name, dimension in GROUP_DIMS.items():

            if name not in transformed:
                continue

            candidate = transformed[name]

            if candidate.shape[1] >= dimension:
                groups[name] = candidate[:, :dimension]

    else:
        raise KeyError(method)

    actual_dim = int(scores.shape[1])

    if actual_dim == 0:
        raise RuntimeError("No score coordinates")

    maps = {
        "all": affine_fit(x, scores),
    }

    for name, values in groups.items():
        maps[name] = affine_fit(x, values)

    full_rank = np.linalg.matrix_rank(
        maps["all"][1:]
    )

    eligible = (
        actual_dim == 7
        and full_rank == 7
    )

    if method != "AJIVE7_adaptive" and not eligible:
        raise RuntimeError(
            f"{method} failed requested 7D budget"
        )

    for name, coef in maps.items():

        expected = (
            scores
            if name == "all"
            else groups[name]
        )

        obtained = affine_predict(x, coef)

        relative_error = (
            np.linalg.norm(obtained - expected)
            / max(
                np.linalg.norm(
                    expected - expected.mean(axis=0)
                ),
                1e-12,
            )
        )

        if relative_error > 1e-7:
            raise RuntimeError(
                f"Affine adapter error: {relative_error}"
            )

    return {
        "method": method,
        "maps": maps,
        "actual_dim": actual_dim,
        "rank": int(full_rank),
        "eligible_7D": bool(eligible),
    }


def transform(fit, x, group="all"):
    return affine_predict(
        x,
        fit["maps"][group],
    )


# ============================================================================
# Jacobian and perturbation diagnostics
# ============================================================================

def normalized_jacobian(x, fit, group):

    """
    Input x is expressed in a common reference-standardized
    measurement space.

    Each score is standardized using fitting observations
    to avoid arbitrary differences in coordinate scale.
    """

    coef = fit["maps"][group]

    z = affine_predict(x, coef)
    scale = z.std(axis=0)

    if np.any(scale <= 1e-12):
        raise ValueError("Degenerate score variance")

    return coef[1:] / scale[None, :]


def leakage_fraction(jacobian, group):

    """
    Basis-invariant cross-domain support.

    An orthonormal basis of the loading subspace is used,
    avoiding dependence on arbitrary score scaling or
    rotations within the designated group.
    """

    q = subspace_basis(jacobian)

    if group == "H_private":
        cross = q[4:, :]

    elif group == "G_private":
        cross = q[:4, :]

    else:
        return np.nan

    return float(
        np.linalg.norm(cross, ord="fro") ** 2
        / q.shape[1]
    )


def jacobian_diagnostics(
    x_train,
    x_test,
    fit,
    perturbations,
    phase,
    condition,
    rep,
):

    rows = []

    for group in fit["maps"]:

        if not fit["eligible_7D"]:
            continue

        j = normalized_jacobian(
            x_train,
            fit,
            group,
        )

        z = transform(
            fit,
            x_test,
            group,
        )

        sd = transform(
            fit,
            x_train,
            group,
        ).std(axis=0)

        leakage = leakage_fraction(
            j,
            group,
        )

        # All methods evaluated in identical measurement units.
        # Conditioning is reported, not assumed to favour HG7.

        singular_values = np.linalg.svd(
            j,
            compute_uv=False,
        )

        condition_number = (
            singular_values[0]
            / singular_values[-1]
        )

        inverse_error = np.nan

        if group == "all" and j.shape == (7, 7):
            inverse_error = float(
                np.max(
                    np.abs(
                        j @ np.linalg.inv(j)
                        - np.eye(7)
                    )
                )
            )

        for noise, epsilon in perturbations.items():

            perturbed = transform(
                fit,
                x_test + epsilon,
                group,
            )

            standardized_change = (
                perturbed - z
            ) / sd

            actual_rms = float(
                np.sqrt(
                    np.mean(standardized_change ** 2)
                )
            )

            predicted_change = epsilon @ j

            agreement_error = float(
                np.max(
                    np.abs(
                        predicted_change
                        - standardized_change
                    )
                )
            )

            input_rms = float(
                np.sqrt(np.mean(epsilon ** 2))
            )

            rows.append({
                "phase": phase,
                "condition": condition,
                "replicate": rep,
                "method": fit["method"],
                "group": group,
                "group_dim": j.shape[1],
                "noise_sd": noise,
                "input_rms": input_rms,
                "score_rms_change": actual_rms,
                "noise_amplification":
                    actual_rms / input_rms,
                "condition_number": condition_number,
                "jacobian_frobenius":
                    np.linalg.norm(j, ord="fro"),
                "cross_domain_leakage": leakage,
                "inverse_identity_error": inverse_error,
                "analytic_perturbation_error":
                    agreement_error,
            })

    return rows


# ============================================================================
# Empirical structural specificity
# ============================================================================

def conditional_shuffle(x, df, rng):

    """
    Permute complete glycaemic rows within survey cycle,
    sex and broad age bands.

    The null preserves within-panel glycaemic relationships
    and coarse demographic composition.

    This is an exploratory conditional placebo, not a
    design-based NHANES independence test.
    """

    age = df["RIDAGEYR"].to_numpy(float)
    sex = df["RIAGENDR"].to_numpy(int)

    cycle = df["CYCLE"].astype(str).to_numpy()

    age_band = np.digitize(
        age,
        [35, 50, 65],
    )

    strata = pd.DataFrame({
        "cycle": cycle,
        "sex": sex,
        "age_band": age_band,
    })

    result = x.copy()
    movable = 0

    for _, ids in strata.groupby(
        ["cycle", "sex", "age_band"]
    ).indices.items():

        ids = np.asarray(ids)

        if len(ids) >= 2:

            result[ids, 4:] = x[
                rng.permutation(ids),
                4:
            ]

            movable += len(ids)

    return result, movable / len(x)


def matched_group_similarity(reference, candidate, anchor):

    values = []

    if not (
        reference["eligible_7D"]
        and candidate["eligible_7D"]
    ):
        return values

    for group in GROUP_DIMS:

        if (
            group not in reference["maps"]
            or group not in candidate["maps"]
        ):
            continue

        a = transform(reference, anchor, group)
        b = transform(candidate, anchor, group)

        if a.shape[1] != b.shape[1]:
            continue

        values.append({
            "group": group,
            "group_dim": a.shape[1],
            "similarity": similarity(a, b),
        })

    return values


def empirical_experiment(out, deps, args):

    d, _ = deps["benchmark"].load_matched()

    d = d.copy()
    d["CYCLE"] = (
        d["CYCLE"]
        .astype(str)
        .str.zfill(4)
    )

    cols = deps["benchmark"].X_COLS

    cohorts = {}

    for index, (name, cycles) in enumerate(
        COHORTS.items()
    ):

        subset = d.loc[
            d["CYCLE"].isin(cycles)
        ].copy()

        if args.cap and len(subset) > args.cap:

            rng = rng_for(10, index)

            indices = rng.choice(
                len(subset),
                args.cap,
                replace=False,
            )

            subset = subset.iloc[
                np.sort(indices)
            ]

        subset = subset.reset_index(drop=True)

        if len(subset) < 150:
            raise RuntimeError(
                f"Insufficient empirical sample: {name}"
            )

        cohorts[name] = subset

    # Reserve a common independent reference anchor.
    # No method is fitted on these anchor participants.

    reference = cohorts["0506"]

    rng = rng_for(11)

    order = rng.permutation(len(reference))

    n_anchor = max(
        100,
        int(0.25 * len(reference)),
    )

    anchor_df = reference.iloc[
        order[:n_anchor]
    ]

    reference_df = reference.iloc[
        order[n_anchor:]
    ]

    if len(reference_df) < 100:
        raise ValueError("Reference fit sample too small")

    common_scaler = StandardScaler().fit(
        reference_df[cols].to_numpy(float)
    )

    x_ref = common_scaler.transform(
        reference_df[cols].to_numpy(float)
    )

    anchor = common_scaler.transform(
        anchor_df[cols].to_numpy(float)
    )

    target_x = {
        name: common_scaler.transform(
            frame[cols].to_numpy(float)
        )
        for name, frame in cohorts.items()
        if name != "0506"
    }

    dump_json(
        out / "117_empirical_sample.json",
        {
            "reference_fit_n": len(x_ref),
            "independent_anchor_n": len(anchor),
            "target_sizes": {
                k: len(v)
                for k, v in target_x.items()
            },
            "markers": cols,
            "common_fit_weights": "unweighted",
        },
    )

    ref_fits = {}

    perturb_rng = rng_for(12)

    base_noise = perturb_rng.normal(
        size=anchor.shape
    )

    perturbations = {
        level: level * base_noise
        for level in NOISE_LEVELS
    }

    for method in args.methods:

        try:

            fit = fit_representation(
                method,
                x_ref,
                deps,
                args,
            )

            ref_fits[method] = fit

            append_csv(
                out / "117_fit_status.csv",
                [{
                    "phase": "empirical_reference",
                    "condition": "0506",
                    "replicate": 0,
                    "method": method,
                    "actual_dim": fit["actual_dim"],
                    "rank": fit["rank"],
                    "eligible_7D": fit["eligible_7D"],
                }],
            )

            append_csv(
                out / "117_jacobian.csv",
                jacobian_diagnostics(
                    x_ref,
                    anchor,
                    fit,
                    perturbations,
                    "empirical_reference",
                    "0506",
                    0,
                ),
            )

        except Exception:

            log_failure(
                out,
                "empirical_reference",
                "0506",
                0,
                method,
            )

    if "HG7" not in ref_fits:
        raise RuntimeError("HG7 reference fit failed")

    structural_methods = [
        m for m in args.methods
        if m != "RawX7"
        and m in ref_fits
    ]

    total = (
        len(target_x)
        * (1 + args.null + args.bootstrap)
        * len(structural_methods)
    )

    progress = tqdm(
        total=total,
        desc="Empirical structural comparisons",
    )

    for cohort_index, (name, xx) in enumerate(
        target_x.items()
    ):

        df = cohorts[name]

        tasks = (
            [("observed", 0)]
            + [
                ("conditional_null", i)
                for i in range(args.null)
            ]
            + [
                ("target_bootstrap", i)
                for i in range(args.bootstrap)
            ]
        )

        for kind, rep in tasks:

            task_rng = rng_for(
                20,
                cohort_index,
                rep,
                {
                    "observed": 0,
                    "conditional_null": 1,
                    "target_bootstrap": 2,
                }[kind],
            )

            movable = np.nan

            if kind == "conditional_null":

                fit_x, movable = conditional_shuffle(
                    xx,
                    df,
                    task_rng,
                )

            elif kind == "target_bootstrap":

                # Same case-bootstrap rows for every method.
                # Stratified by survey cycle; descriptive only.

                indices = []

                for _, ids in df.groupby(
                    "CYCLE"
                ).indices.items():

                    ids = np.asarray(ids)

                    indices.extend(
                        task_rng.choice(
                            ids,
                            len(ids),
                            replace=True,
                        )
                    )

                fit_x = xx[
                    np.asarray(indices)
                ]

            else:
                fit_x = xx

            for method in structural_methods:

                try:

                    fit = fit_representation(
                        method,
                        fit_x,
                        deps,
                        args,
                    )

                    append_csv(
                        out / "117_fit_status.csv",
                        [{
                            "phase": kind,
                            "condition": name,
                            "replicate": rep,
                            "method": method,
                            "actual_dim":
                                fit["actual_dim"],
                            "rank": fit["rank"],
                            "eligible_7D":
                                fit["eligible_7D"],
                        }],
                    )

                    rows = matched_group_similarity(
                        ref_fits[method],
                        fit,
                        anchor,
                    )

                    append_csv(
                        out / "117_empirical_metrics.csv",
                        [
                            {
                                "cohort": name,
                                "kind": kind,
                                "replicate": rep,
                                "method": method,
                                "movable_fraction":
                                    movable,
                                **row,
                            }
                            for row in rows
                        ],
                    )

                except Exception:

                    log_failure(
                        out,
                        kind,
                        name,
                        rep,
                        method,
                    )

                finally:
                    progress.update(1)

    progress.close()


# ============================================================================
# Synthetic operating conditions
# ============================================================================

def random_rotation(rng, dimension):

    q, r = np.linalg.qr(
        rng.normal(
            size=(dimension, dimension)
        )
    )

    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1

    return q * signs


def rotation_shift(rng, dimension, magnitude=0.35):

    a = rng.normal(
        size=(dimension, dimension)
    )

    skew = a - a.T

    norm = np.linalg.norm(
        skew,
        ord=2,
    )

    if norm > 0:
        skew = skew / norm

    return expm(magnitude * skew)


def generate_data(
    rng,
    n,
    scenario,
    settings,
    rotations,
    is_test=False,
):

    s = rng.normal(size=(n, 2))
    d = rng.normal(size=(n, 2))

    h = rng.normal(size=(n, 2))
    g = rng.normal(size=(n, 1))

    private_scale = settings["private"]

    if (
        is_test
        and settings["shift"] == "population"
    ):
        s = 1.25 * s + 0.50
        h = 1.50 * h
        g = 0.75 * g

    rho = settings["rho"]

    # Before rotation and measurement noise:
    # Var(U) = Var(V) = 1
    # Corr(U,V) = rho
    # Shared and discordant amplitudes vary continuously.

    shared_amp = np.sqrt((1.0 + rho) / 2.0)
    discord_amp = np.sqrt((1.0 - rho) / 2.0)

    if scenario == "no_sharing":

        u = rng.normal(size=(n, 2))
        v = rng.normal(size=(n, 2))

        truth = {
            "H_private": h,
            "G_private": g,
        }

    else:

        u = shared_amp * s + discord_amp * d
        v = shared_amp * s - discord_amp * d

        truth = {
            "shared": s,
            "discordant": d,
            "H_private": h,
            "G_private": g,
        }

    xh = np.column_stack([
        u,
        private_scale * h,
    ])

    xg = np.column_stack([
        v,
        private_scale * g,
    ])

    xh = xh @ rotations[0]
    xg = xg @ rotations[1]

    x = np.column_stack([xh, xg])

    x += rng.normal(
        scale=settings["noise"],
        size=x.shape,
    )

    return x, truth


def synthetic_experiment(out, deps, args):

    total = (
        len(SCENARIOS)
        * args.reps
        * len(args.methods)
    )

    progress = tqdm(
        total=total,
        desc="Synthetic operating conditions",
    )

    for scenario_index, (
        scenario,
        settings,
    ) in enumerate(SCENARIOS.items()):

        for rep in range(args.reps):

            rng = rng_for(
                30,
                scenario_index,
                rep,
            )

            rotations = [
                random_rotation(rng, 4),
                random_rotation(rng, 3),
            ]

            test_rotations = [
                r.copy() for r in rotations
            ]

            if settings["shift"] == "measurement":

                test_rotations = [
                    rotations[0] @ rotation_shift(
                        rng, 4
                    ),
                    rotations[1] @ rotation_shift(
                        rng, 3
                    ),
                ]

            train, y_train = generate_data(
                rng,
                args.n_train,
                scenario,
                settings,
                rotations,
                is_test=False,
            )

            test, y_test = generate_data(
                rng,
                args.n_test,
                scenario,
                settings,
                test_rotations,
                is_test=True,
            )

            # Common discovery measurement scale.
            common_scaler = StandardScaler().fit(train)

            train = common_scaler.transform(train)
            test = common_scaler.transform(test)

            gate_rng = rng_for(
                31,
                scenario_index,
                rep,
            )

            gate = deps["s116"].gate(
                train,
                args.permutations,
                gate_rng,
                0.05,
                0.10,
            )

            append_csv(
                out / "117_rank_gate.csv",
                [{
                    "scenario": scenario,
                    "replicate": rep,
                    **gate,
                }],
            )

            # Fixed-rank HG7 remains fixed for matched
            # representation comparison. Gate results
            # are reported separately.

            epsilon = rng.normal(size=test.shape)

            perturbations = {
                level: level * epsilon
                for level in NOISE_LEVELS
            }

            for method in args.methods:

                try:

                    fit = fit_representation(
                        method,
                        train,
                        deps,
                        args,
                    )

                    append_csv(
                        out / "117_fit_status.csv",
                        [{
                            "phase": "synthetic",
                            "condition": scenario,
                            "replicate": rep,
                            "method": method,
                            "actual_dim":
                                fit["actual_dim"],
                            "rank": fit["rank"],
                            "eligible_7D":
                                fit["eligible_7D"],
                        }],
                    )

                    if not fit["eligible_7D"]:
                        continue

                    z_train = transform(fit, train)
                    z_test = transform(fit, test)

                    decoder = affine_fit(
                        z_train,
                        train,
                    )

                    reconstruction = r2_matrix(
                        test,
                        affine_predict(
                            z_test,
                            decoder,
                        ),
                        center=train.mean(axis=0),
                    )

                    if abs(reconstruction - 1) > 1e-7:
                        raise RuntimeError(
                            "7D reconstruction invariant failed"
                        )

                    rows = []

                    for group in fit["maps"]:

                        train_group = transform(
                            fit,
                            train,
                            group,
                        )

                        test_group = transform(
                            fit,
                            test,
                            group,
                        )

                        for target in y_train:

                            score = heldout_r2(
                                train_group,
                                test_group,
                                y_train[target],
                                y_test[target],
                            )

                            rows.append({
                                "scenario": scenario,
                                "replicate": rep,
                                "method": method,
                                "group": group,
                                "group_dim":
                                    train_group.shape[1],
                                "target": target,
                                "heldout_R2": score,
                                "reconstruction_R2":
                                    reconstruction,
                            })

                    append_csv(
                        out / "117_synthetic_metrics.csv",
                        rows,
                    )

                    append_csv(
                        out / "117_jacobian.csv",
                        jacobian_diagnostics(
                            train,
                            test,
                            fit,
                            perturbations,
                            "synthetic",
                            scenario,
                            rep,
                        ),
                    )

                except Exception:

                    log_failure(
                        out,
                        "synthetic",
                        scenario,
                        rep,
                        method,
                    )

                finally:
                    progress.update(1)

    progress.close()


# ============================================================================
# Empirical summaries
# ============================================================================

def summarize_empirical(out, args):

    data = read_csv(
        out / "117_empirical_metrics.csv"
    )

    if data.empty:
        return

    observed = data.loc[
        data.kind.eq("observed")
    ]

    rows = []
    paired = []

    for row in observed.itertuples():

        subset = data.loc[
            data.cohort.eq(row.cohort)
            & data.method.eq(row.method)
            & data.group.eq(row.group)
        ]

        null = subset.loc[
            subset.kind.eq("conditional_null")
        ]

        boot = subset.loc[
            subset.kind.eq("target_bootstrap")
        ]

        enough_null = (
            len(null) >= np.ceil(0.80 * args.null)
        )

        enough_boot = (
            len(boot) >= np.ceil(
                0.80 * args.bootstrap
            )
        )

        q95 = (
            float(np.quantile(null.similarity, 0.95))
            if enough_null
            else np.nan
        )

        lo, hi = (
            np.percentile(
                boot.similarity,
                [2.5, 97.5],
            )
            if enough_boot and len(boot) > 1
            else (np.nan, np.nan)
        )

        rows.append({
            "cohort": row.cohort,
            "method": row.method,
            "group": row.group,
            "group_dim": row.group_dim,
            "observed_similarity":
                row.similarity,
            "conditional_null_q95": q95,
            "specificity_margin":
                row.similarity - q95,
            "descriptive_bootstrap_low": lo,
            "descriptive_bootstrap_high": hi,
            "valid_null_reps": len(null),
            "requested_null_reps": args.null,
            "valid_bootstrap_reps": len(boot),
            "requested_bootstrap_reps":
                args.bootstrap,
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        out / "117_empirical_summary.csv",
        index=False,
    )

    hg = observed.loc[
        observed.method.eq("HG7")
    ]

    for method in args.methods:

        if method == "HG7":
            continue

        for h in hg.itertuples():

            candidate = observed.loc[
                observed.method.eq(method)
                & observed.cohort.eq(h.cohort)
                & observed.group.eq(h.group)
                & observed.group_dim.eq(h.group_dim)
            ]

            if len(candidate) != 1:
                continue

            c = candidate.iloc[0]

            subset = data.loc[
                data.cohort.eq(h.cohort)
                & data.group.eq(h.group)
                & data.group_dim.eq(h.group_dim)
            ]

            def paired_draws(kind):

                aa = subset.loc[
                    subset.kind.eq(kind)
                    & subset.method.eq("HG7"),
                    ["replicate", "similarity"],
                ]

                bb = subset.loc[
                    subset.kind.eq(kind)
                    & subset.method.eq(method),
                    ["replicate", "similarity"],
                ]

                return aa.merge(
                    bb,
                    on="replicate",
                    suffixes=("_HG", "_ref"),
                    validate="one_to_one",
                )

            null = paired_draws(
                "conditional_null"
            )

            boot = paired_draws(
                "target_bootstrap"
            )

            margin_delta = np.nan
            lo = np.nan
            hi = np.nan

            if len(null) >= np.ceil(
                0.80 * args.null
            ):

                hg_margin = (
                    h.similarity
                    - np.quantile(
                        null.similarity_HG,
                        0.95,
                    )
                )

                ref_margin = (
                    c.similarity
                    - np.quantile(
                        null.similarity_ref,
                        0.95,
                    )
                )

                margin_delta = (
                    hg_margin - ref_margin
                )

            if (
                len(boot) >= np.ceil(
                    0.80 * args.bootstrap
                )
                and len(boot) > 1
            ):

                delta = (
                    boot.similarity_HG
                    - boot.similarity_ref
                )

                lo, hi = np.percentile(
                    delta,
                    [2.5, 97.5],
                )

            paired.append({
                "cohort": h.cohort,
                "group": h.group,
                "group_dim": h.group_dim,
                "comparator": method,
                "observed_similarity_delta":
                    h.similarity - c.similarity,
                "matched_null_margin_delta":
                    margin_delta,
                "paired_target_boot_low": lo,
                "paired_target_boot_high": hi,
                "valid_paired_null_reps":
                    len(null),
                "valid_paired_bootstrap_reps":
                    len(boot),
            })

    pd.DataFrame(paired).to_csv(
        out / "117_empirical_paired.csv",
        index=False,
    )


# ============================================================================
# Synthetic summaries
# ============================================================================

def summarize_synthetic(out, args):

    data = read_csv(
        out / "117_synthetic_metrics.csv"
    )

    if data.empty:
        return

    summary = data.groupby([
        "scenario",
        "method",
        "group",
        "group_dim",
        "target",
    ]).agg(
        successful_reps=("heldout_R2", "count"),
        mean_R2=("heldout_R2", "mean"),
        sd_R2=("heldout_R2", "std"),
        mean_reconstruction_R2=(
            "reconstruction_R2",
            "mean",
        ),
    ).reset_index()

    summary.to_csv(
        out / "117_synthetic_summary.csv",
        index=False,
    )

    rows = []

    hg = data.loc[
        data.method.eq("HG7")
    ]

    for method_index, method in enumerate(
        args.methods
    ):

        if method == "HG7":
            continue

        other = data.loc[
            data.method.eq(method)
        ]

        merged = hg.merge(
            other,
            on=[
                "scenario",
                "replicate",
                "group",
                "group_dim",
                "target",
            ],
            suffixes=("_HG", "_ref"),
            validate="one_to_one",
        )

        for index, (key, block) in enumerate(
            merged.groupby([
                "scenario",
                "group",
                "group_dim",
                "target",
            ])
        ):

            scenario, group, dimension, target = key

            delta = (
                block.heldout_R2_HG
                - block.heldout_R2_ref
            ).to_numpy(float)

            lo, hi = paired_interval(
                delta,
                rng_for(40, method_index, index),
            )

            adequate = (
                len(delta)
                >= np.ceil(0.80 * args.reps)
            )

            if not adequate:
                decision = "insufficient_matched_fits"

            elif not (
                np.isfinite(lo)
                and np.isfinite(hi)
            ):
                decision = "insufficient_uncertainty"

            elif group == "all":
                decision = "information_control"

            elif lo > 0:
                decision = "favours_HG7"

            elif hi < 0:
                decision = "favours_comparator"

            else:
                decision = "inconclusive"

            rows.append({
                "scenario": scenario,
                "group": group,
                "group_dim": dimension,
                "target": target,
                "comparator": method,
                "paired_reps": len(delta),
                "requested_reps": args.reps,
                "mean_delta_R2":
                    float(delta.mean()),
                "paired_bootstrap_low95": lo,
                "paired_bootstrap_high95": hi,
                "decision": decision,
            })

    pd.DataFrame(rows).to_csv(
        out / "117_synthetic_paired.csv",
        index=False,
    )

    # Within-method target selectivity:
    # designated target recovery minus the best
    # non-designated target recovered by that group.

    selectivity = []

    for key, block in data.groupby([
        "scenario",
        "replicate",
        "method",
        "group",
    ]):

        scenario, rep, method, group = key

        if group == "all":
            continue

        own = block.loc[
            block.target.eq(group),
            "heldout_R2",
        ]

        other = block.loc[
            block.target.ne(group),
            "heldout_R2",
        ]

        if len(own) != 1 or other.empty:
            continue

        selectivity.append({
            "scenario": scenario,
            "replicate": rep,
            "method": method,
            "group": group,
            "target_R2": own.iloc[0],
            "largest_off_target_R2":
                other.max(),
            "selectivity_margin":
                own.iloc[0] - other.max(),
        })

    pd.DataFrame(selectivity).to_csv(
        out / "117_group_selectivity.csv",
        index=False,
    )


def summarize_jacobians(out):

    data = read_csv(
        out / "117_jacobian.csv"
    )

    if data.empty:
        return

    summary = data.groupby([
        "phase",
        "condition",
        "method",
        "group",
        "group_dim",
        "noise_sd",
    ]).agg(
        successful_reps=(
            "noise_amplification",
            "count",
        ),
        mean_noise_amplification=(
            "noise_amplification",
            "mean",
        ),
        median_condition_number=(
            "condition_number",
            "median",
        ),
        mean_cross_domain_leakage=(
            "cross_domain_leakage",
            "mean",
        ),
        max_inverse_identity_error=(
            "inverse_identity_error",
            "max",
        ),
        max_analytic_perturbation_error=(
            "analytic_perturbation_error",
            "max",
        ),
    ).reset_index()

    summary.to_csv(
        out / "117_jacobian_summary.csv",
        index=False,
    )


# ============================================================================
# Publication-oriented figures
# ============================================================================

def make_figures(out):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    colors = {
        "HG7": "#176D69",
        "PCA7": "#697680",
        "BlockPCA7": "#7C8799",
        "JIVE7": "#A55A37",
        "AJIVE7_adaptive": "#946E9B",
        "RawX7": "#333333",
    }

    empirical = read_csv(
        out / "117_empirical_summary.csv"
    )

    synthetic = read_csv(
        out / "117_synthetic_summary.csv"
    )

    jacobian = read_csv(
        out / "117_jacobian_summary.csv"
    )

    # Figure A: empirical specificity margins.

    if not empirical.empty:

        groups = [
            "shared",
            "discordant",
            "H_private",
            "G_private",
        ]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 6),
            sharey=True,
        )

        periods = ["0708", "0918", "2123"]

        for ax, group in zip(
            axes.flat,
            groups,
        ):

            panel = empirical.loc[
                empirical.group.eq(group)
            ]

            for method, block in panel.groupby(
                "method"
            ):

                block = (
                    block
                    .set_index("cohort")
                    .reindex(periods)
                )

                ax.plot(
                    range(3),
                    block.specificity_margin,
                    marker="o",
                    color=colors.get(method),
                    label=method,
                )

            ax.axhline(
                0,
                color="black",
                linestyle=":",
                linewidth=0.8,
            )

            ax.set_title(group.replace("_", " "))
            ax.set_xticks(
                range(3),
                ["2007–08", "2009–18", "2021–23"],
            )

        handles = {}
        for ax in axes.flat:
            h, labels = ax.get_legend_handles_labels()
            handles.update(zip(labels, h))

        fig.legend(
            handles.values(),
            handles.keys(),
            loc="lower center",
            ncol=3,
            frameon=False,
        )

        fig.supylabel(
            "Observed similarity minus null 95th percentile"
        )

        fig.tight_layout(
            rect=[0.03, 0.10, 1, 1]
        )

        fig.savefig(
            out / "117_structural_specificity.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            out / "117_structural_specificity.png",
            dpi=400,
            bbox_inches="tight",
        )

        plt.close(fig)

    # Figure B: designated synthetic target recovery.

    if not synthetic.empty:

        own = synthetic.loc[
            synthetic.group.eq(synthetic.target)
            & synthetic.group.ne("all")
        ]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11, 7),
        )

        scenarios = list(SCENARIOS)

        for ax, group in zip(
            axes.flat,
            GROUP_DIMS,
        ):

            panel = own.loc[
                own.group.eq(group)
            ]

            for method, block in panel.groupby(
                "method"
            ):

                block = (
                    block
                    .set_index("scenario")
                    .reindex(scenarios)
                )

                adequate = (
                    block.successful_reps
                    >= np.ceil(
                        0.80
                        * synthetic.successful_reps.max()
                    )
                )

                values = block.mean_R2.where(
                    adequate
                )

                ax.plot(
                    range(len(scenarios)),
                    values,
                    marker="o",
                    markersize=3.5,
                    color=colors.get(method),
                    label=method,
                )

            ax.axhline(
                0,
                color="black",
                linestyle=":",
                linewidth=0.8,
            )

            ax.set_title(group.replace("_", " "))

            ax.set_xticks(
                range(len(scenarios)),
                [
                    name.replace("_", " ")
                    for name in scenarios
                ],
                rotation=40,
                ha="right",
            )

        handles = {}
        for ax in axes.flat:
            h, labels = ax.get_legend_handles_labels()
            handles.update(zip(labels, h))

        fig.legend(
            handles.values(),
            handles.keys(),
            loc="lower center",
            ncol=3,
            frameon=False,
        )

        fig.supylabel("Held-out synthetic target R²")

        fig.tight_layout(
            rect=[0.03, 0.08, 1, 1]
        )

        fig.savefig(
            out / "117_operating_conditions.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            out / "117_operating_conditions.png",
            dpi=400,
            bbox_inches="tight",
        )

        plt.close(fig)

    # Figure C: empirical perturbation response.

    if not jacobian.empty:

        panel = jacobian.loc[
            jacobian.phase.eq("empirical_reference")
            & jacobian.group.eq("all")
        ]

        if not panel.empty:

            fig, ax = plt.subplots(
                figsize=(7, 4.5)
            )

            for method, block in panel.groupby(
                "method"
            ):

                block = block.sort_values("noise_sd")

                ax.plot(
                    block.noise_sd,
                    block.mean_noise_amplification,
                    marker="o",
                    color=colors.get(method),
                    label=method,
                )

            ax.set_xlabel(
                "Measurement perturbation (reference SD)"
            )

            ax.set_ylabel(
                "Standardized score noise amplification"
            )

            ax.legend(frameon=False)

            fig.tight_layout()

            fig.savefig(
                out / "117_perturbation_response.pdf",
                bbox_inches="tight",
            )

            fig.savefig(
                out / "117_perturbation_response.png",
                dpi=400,
                bbox_inches="tight",
            )

            plt.close(fig)


# ============================================================================
# Completion and evidence boundaries
# ============================================================================

def write_readout(out, args):

    status = read_csv(
        out / "117_fit_status.csv"
    )

    failures = read_csv(
        out / "117_failures.csv"
    )

    empirical = read_csv(
        out / "117_empirical_summary.csv"
    )

    paired = read_csv(
        out / "117_synthetic_paired.csv"
    )

    gates = read_csv(
        out / "117_rank_gate.csv"
    )

    lines = [
        "# Script 117 — Comparative structural validation",
        "",
        "## Interpretation",
        "",
        "- Full-rank reconstruction is an information control.",
        "- The comparative refits use common unweighted fitting.",
        "- Empirical nulls preserve survey cycle, sex and broad age bands.",
        "- Empirical bootstrap intervals condition on the reference fit.",
        "- Empirical intervals are descriptive, not survey-design CIs.",
        "- Native score groups are dimension matched, not semantically identical.",
        "- PCA leading components are variance controls.",
        "- Block PCA components are domain-restricted controls.",
        "- No artificial discordance groups are assigned to comparators.",
        "- Rank-deficient AJIVE fits are excluded from strict comparisons.",
        "- Synthetic recovery depends on the stipulated generator.",
        "- Jacobian restrictions do not identify biological sources.",
        "- All intervals are exploratory and unadjusted for multiplicity.",
        "",
    ]

    if not status.empty:

        counts = status.groupby([
            "phase",
            "condition",
            "method",
        ]).agg(
            recorded_fits=("rank", "count"),
            eligible_7D_fits=("eligible_7D", "sum"),
        ).reset_index()

        counts.to_csv(
            out / "117_fit_counts.csv",
            index=False,
        )

        lines.extend([
            "## Actual fitted dimensions",
            "",
            "```text",
            counts.to_string(index=False),
            "```",
            "",
        ])

    if not empirical.empty:

        lines.extend([
            "## Empirical structural specificity",
            "",
            "```text",
            empirical.round(5).to_string(index=False),
            "```",
            "",
        ])

    if not paired.empty:

        primary = paired.loc[
            paired.group.eq(paired.target)
            & paired.group.ne("all")
        ]

        lines.extend([
            "## Matched designated-group recovery",
            "",
            "```text",
            primary.round(5).to_string(index=False),
            "```",
            "",
        ])

    if not gates.empty:

        gate_summary = gates.groupby(
            "scenario"
        ).agg(
            mean_selected_rank=(
                "detected_rank",
                "mean",
            ),
            fraction_detecting_coupling=(
                "any_detected",
                "mean",
            ),
            repetitions=(
                "detected_rank",
                "count",
            ),
        ).reset_index()

        gate_summary.to_csv(
            out / "117_gate_summary.csv",
            index=False,
        )

        lines.extend([
            "## Rank-selection sensitivity",
            "",
            "```text",
            gate_summary.to_string(index=False),
            "```",
            "",
        ])

    lines.extend([
        "## Recorded fitting exceptions",
        "",
        str(len(failures)),
        "",
        "No universal method ranking is inferred.",
        "",
    ])

    (out / "117_READOUT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    dump_json(
        out / "117_completion.json",
        {
            "recorded_successful_fits": len(status),
            "recorded_fitting_exceptions":
                len(failures),
            "eligible_7D_fits": (
                int(status.eligible_7D.sum())
                if not status.empty
                else 0
            ),
            "completed": True,
            "qualification":
                "Completed execution does not imply "
                "every comparator met its rank budget.",
        },
    )


# ============================================================================
# Numerical self-tests
# ============================================================================

def self_test():

    rng = rng_for(99)

    x = rng.normal(size=(500, 7))

    a = random_rotation(rng, 7)

    z = x @ a

    assert np.linalg.matrix_rank(a) == 7

    assert abs(
        similarity(
            z[:, :2],
            -z[:, :2],
        ) - 1.0
    ) < 1e-10

    coef = affine_fit(z, x)

    assert np.max(
        np.abs(
            affine_predict(z, coef) - x
        )
    ) < 1e-10

    h_map = np.zeros((7, 2))
    h_map[:4] = rng.normal(size=(4, 2))

    assert leakage_fraction(
        h_map,
        "H_private",
    ) < 1e-12

    g_map = np.zeros((7, 1))
    g_map[4:] = rng.normal(size=(3, 1))

    assert leakage_fraction(
        g_map,
        "G_private",
    ) < 1e-12

    rotations = [
        np.eye(4),
        np.eye(3),
    ]

    simulated, truth = generate_data(
        rng,
        2000,
        "reference",
        {
            "rho": 0.50,
            "noise": 0.0,
            "private": 1.0,
            "shift": "none",
        },
        rotations,
    )

    assert simulated.shape == (2000, 7)

    assert set(truth) == {
        "shared",
        "discordant",
        "H_private",
        "G_private",
    }

    observed_corr = np.corrcoef(
        simulated[:, 0],
        simulated[:, 4],
    )[0, 1]

    assert abs(observed_corr - 0.50) < 0.10

    print("PASS: affine inversion")
    print("PASS: subspace invariance")
    print("PASS: cross-domain leakage")
    print("PASS: synthetic coupling generator")


# ============================================================================
# Entry point
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--null",
        type=int,
        default=199,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=99,
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=199,
    )

    parser.add_argument(
        "--n-train",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--cap",
        type=int,
        default=1800,
        help="Maximum real rows per period; zero uses all.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )

    parser.add_argument(
        "--run-name",
        default="117_comparative_structural_validation",
    )

    parser.add_argument(
        "--phase",
        choices=("all", "empirical", "synthetic"),
        default="all",
    )

    parser.add_argument(
        "--rscript",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    args.methods = list(
        dict.fromkeys(args.methods)
    )

    if "HG7" not in args.methods:
        parser.error("HG7 is required")

    if min(
        args.reps,
        args.null,
        args.bootstrap,
    ) < 5:

        parser.error(
            "Require at least five replicates"
        )

    if args.permutations < 19:
        parser.error(
            "Require at least 19 permutations"
        )

    if min(args.n_train, args.n_test) < 100:
        parser.error("Simulation samples too small")

    if args.cap < 0:
        parser.error("Invalid sample cap")

    if not all(
        c.isalnum() or c in "_-"
        for c in args.run_name
    ):
        parser.error("Invalid run name")

    if args.rscript is None:

        args.rscript = (
            Path(sys.prefix)
            / "Lib/R/bin/x64/Rscript.exe"
        )

        if not args.rscript.is_file():

            detected = shutil.which("Rscript")

            if detected:
                args.rscript = Path(detected)

    if (
        "JIVE7" in args.methods
        and not args.rscript.is_file()
    ):
        raise FileNotFoundError(
            "Rscript not found; pass --rscript explicitly"
        )

    deps = load_dependencies()

    out = EDA / "Results" / args.run_name

    if out.exists():
        raise FileExistsError(
            "Output directory already exists. "
            "Use a new --run-name."
        )

    out.mkdir(parents=True)

    dump_json(
        out / "117_protocol.json",
        {
            "script": Path(__file__).name,
            "seed": SEED,
            "settings": vars(args),
            "scenarios": SCENARIOS,
            "noise_levels": NOISE_LEVELS,
            "source_hashes": source_hashes(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "group_dimensions": GROUP_DIMS,
            "strict_rank": 7,
            "minimum_valid_pair_fraction": 0.80,
            "primary_comparison":
                "Designated target recovery and "
                "conditioned structural specificity.",
            "native_group_matching":
                "First k native scores; "
                "no truth-guided selection.",
            "empirical_fit_weights": "unweighted",
            "empirical_bootstrap":
                "Cycle-stratified cases; "
                "reference fit fixed.",
            "external_validation": False,
            "confirmatory_inference": False,
        },
    )

    started = time.monotonic()

    if args.phase in ("all", "empirical"):
        empirical_experiment(out, deps, args)
        summarize_empirical(out, args)

    if args.phase in ("all", "synthetic"):
        synthetic_experiment(out, deps, args)
        summarize_synthetic(out, args)

    summarize_jacobians(out)

    if args.plot:
        make_figures(out)

    write_readout(out, args)

    print()
    print("=" * 72)
    print("SCRIPT 117 COMPLETE")
    print(f"Elapsed: {time.monotonic() - started:.1f}s")
    print(f"Results: {out}")
    print(f"Readout: {out / '117_READOUT.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
