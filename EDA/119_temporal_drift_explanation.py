
#!/usr/bin/env python3
"""
119 | Explaining temporal transfer failure.

Reuses the frozen experimental protocol of Script118.

Questions
---------
1. Does deteriorating prediction arise from calibration
   drift or loss of association?
2. Which HG coordinate groups contribute to the excess
   error of the full laboratory representation?
3. Which measured physiological distributions changed?
4. Does the shared pair retain its advantage after
   descriptive oracle recalibration?

No outcome-driven representation modification.
No clinical or causal identification.
No participant-level exports.
"""

from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tqdm import tqdm

sys.dont_write_bytecode = True

EDA = Path(__file__).resolve().parent

SEED = 119

CYCLES = ("0506", "0708", "2123")
HOLDOUTS = ("0708", "2123")

PRIMARY = "HG_Shared2"

VAE_SEEDS = (
    118, 119, 120, 121, 122
)

COMPARATORS = (
    "RawX7",
    "H4",
    "PCA2",
    "H_PCA2",
    "G_PCA2",
    "JIVE_Joint2",
    "AJIVE_Joint2",
    "VAE2_mean_loss",
    "Threshold2",
    "Hb_HbA1c",
    "Hb_RDW",
    "HG_Discordant2",
    "HG_HPrivate2",
    "HG_Weighted_Shared2",
    "Covariates",
)

ERROR_COMPONENTS = (
    "mse",
    "mean_bias_error",
    "slope_error",
    "calibration_error",
    "residual_error",
)

GROUP_NAMES = {
    "intercept": [0],
    "covariates": [1, 2, 3],
    "shared": [4, 5],
    "discordant": [6, 7],
    "H_private": [8, 9],
    "G_private": [10],
}


# ============================================================
# Utilities
# ============================================================

def load_module(name, path):

    spec = importlib.util.spec_from_file_location(
        name, path
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def dump_json(path, obj):

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def save_table(out, name, rows):

    pd.DataFrame(rows).to_csv(
        out / f"119_{name}.csv",
        index=False,
    )


def interval(values):

    lo, hi = np.percentile(
        values, [2.5, 97.5]
    )

    return float(lo), float(hi)


def summarize_draws(values):

    lo, hi = interval(values[1:])

    return {
        "estimate": float(values[0]),
        "psu_low95": lo,
        "psu_high95": hi,
    }


def wmean(x, w):

    return np.average(
        x,
        axis=0,
        weights=w,
    )


def weighted_sd(x, w):

    mu = wmean(x, w)

    return np.sqrt(
        wmean((x - mu) ** 2, w)
    )


# ============================================================
# Exact prediction error decomposition
# ============================================================

def decompose_error(y, pred, weights):

    """
    weights: (number of replicates, participants)

    Row 0 contains original normalized survey weights.
    Remaining rows contain paired bootstrap weights.

    Affine calibration is estimated descriptively on the
    evaluation sample. It is NOT prospective performance.
    """

    y = np.asarray(y, float)
    pred = np.asarray(pred, float)

    mu_y = weights @ y
    mu_p = weights @ pred

    var_y = np.maximum(
        weights @ (y ** 2) - mu_y ** 2,
        0.0,
    )

    var_p = np.maximum(
        weights @ (pred ** 2) - mu_p ** 2,
        0.0,
    )

    cov = (
        weights @ (y * pred)
        - mu_y * mu_p
    )

    slope = np.divide(
        cov,
        var_p,
        out=np.zeros_like(cov),
        where=var_p > 1e-14,
    )

    intercept = mu_y - slope * mu_p

    sd_product = np.sqrt(var_y * var_p)

    corr = np.divide(
        cov,
        sd_product,
        out=np.full_like(cov, np.nan),
        where=sd_product > 1e-14,
    )

    bias = mu_y - mu_p

    mse = weights @ (
        (y - pred) ** 2
    )

    mean_bias_error = bias ** 2

    slope_error = (
        var_p * (slope - 1.0) ** 2
    )

    residual_error = np.maximum(
        var_y - slope * cov,
        0.0,
    )

    calibration_error = (
        mean_bias_error + slope_error
    )

    identity_error = np.max(
        np.abs(
            mse
            - calibration_error
            - residual_error
        )
    )

    if identity_error > 1e-8:

        raise RuntimeError(
            f"MSE decomposition failed: "
            f"{identity_error}"
        )

    return {
        "mse": mse,
        "observed_mean": mu_y,
        "predicted_mean": mu_p,
        "observed_sd": np.sqrt(var_y),
        "predicted_sd": np.sqrt(var_p),
        "signed_mean_bias": bias,
        "mean_bias_error": mean_bias_error,
        "slope_error": slope_error,
        "calibration_error": calibration_error,
        "residual_error": residual_error,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "pearson_r": corr,
        "oracle_affine_MSE": residual_error,
        "oracle_intercept_only_MSE":
            mse - mean_bias_error,
    }


# ============================================================
# Reproduce Script118 exactly
# ============================================================

def reproduce_118(args, out):

    reference_dir = args.reference_run.resolve()

    protocol = json.loads(
        (reference_dir / "118_protocol.json")
        .read_text(encoding="utf-8")
    )

    completion = json.loads(
        (reference_dir / "118_completion.json")
        .read_text(encoding="utf-8")
    )

    if not completion.get("execution_completed"):

        raise RuntimeError(
            "Reference Script118 run incomplete"
        )

    source = (
        EDA / "118_biomedical_utility_benchmark.py"
    )

    s118 = load_module(
        "s118_for_119",
        source,
    )

    config = protocol["settings"]

    rscript = Path(config["rscript"])

    if not rscript.is_file():

        candidate = shutil.which("Rscript")

        if candidate:
            rscript = Path(candidate)

    if not rscript.is_file():

        raise FileNotFoundError(
            "Rscript not found"
        )

    # Reuse the same fitting settings.
    inherited = SimpleNamespace(
        epochs=int(config["epochs"]),
        beta=float(config["beta"]),
        fetch_missing=False,
        skip_native=False,
        rscript=rscript,
        timeout=int(config.get("timeout", 1200)),
    )

    staging = out / "118_reproduction"

    staging.mkdir()

    data = s118.load_inputs(
        inherited,
        staging,
    )

    representations, failures = (
        s118.build_representations(
            data,
            inherited,
            staging,
        )
    )

    predictions, diagnostics = (
        s118.fit_ferritin_models(
            data,
            representations,
            failures,
        )
    )

    pd.DataFrame(
        failures,
        columns=["method", "stage", "error"],
    ).to_csv(
        staging / "118_failures.csv",
        index=False,
    )

    pd.DataFrame(diagnostics).to_csv(
        staging / "118_downstream_diagnostics.csv",
        index=False,
    )

    if failures:

        raise RuntimeError(
            f"Script118 reproduction contains "
            f"{len(failures)} failures"
        )

    required = [
        PRIMARY,
        "RawX7",
        "HG7",
        "PCA7",
        *[
            name for name in COMPARATORS
            if name != "VAE2_mean_loss"
        ],
        *[
            f"VAE2_seed{s}"
            for s in VAE_SEEDS
        ],
    ]

    missing = set(required) - set(predictions)

    if missing:

        raise RuntimeError(
            f"Missing reproduced predictions: "
            f"{sorted(missing)}"
        )

    # Check against the archived original performance.
    old = pd.read_csv(
        reference_dir /
        "118_ferritin_performance.csv",
        dtype={"cohort": str},
    )

    old["cohort"] = (
        old["cohort"].str.zfill(4)
    )

    audit = []

    for cy in CYCLES:

        y = data["anchor"][cy]["y"]
        w = data["anchor"][cy]["w"]

        mse = {
            name: float(
                wmean(
                    (y - pred[cy]) ** 2,
                    w,
                )
            )
            for name, pred in predictions.items()
        }

        mse["VAE2_mean_loss"] = float(
            np.mean([
                mse[f"VAE2_seed{s}"]
                for s in VAE_SEEDS
            ])
        )

        for name, value in mse.items():

            reference = old.loc[
                old.cohort.eq(cy)
                & old.method.eq(name)
            ]

            if len(reference) != 1:

                raise RuntimeError(
                    f"Missing original result: "
                    f"{cy}, {name}"
                )

            original = float(
                reference.weighted_mse.iloc[0]
            )

            difference = value - original

            audit.append({
                "cohort": cy,
                "method": name,
                "original_MSE": original,
                "reproduced_MSE": value,
                "difference": difference,
                "pass":
                    abs(difference) < args.tolerance,
            })

    audit = pd.DataFrame(audit)

    audit.to_csv(
        out / "119_reproduction_audit.csv",
        index=False,
    )

    if not audit["pass"].all():

        raise RuntimeError(
            "Original Script118 performance was not "
            "reproduced. Inspect reproduction audit "
            "before interpreting new diagnostics."
        )

    return (
        data,
        representations,
        predictions,
        protocol,
        source,
    )


# ============================================================
# Paired survey resampling
# ============================================================

def bootstrap_weights(data, cy, reps):

    q = data["anchor"][cy]

    original = q["w"]

    rng = np.random.default_rng(
        np.random.SeedSequence([
            SEED, int(cy)
        ])
    )

    weights = [
        original / original.sum()
    ]

    for _ in tqdm(
        range(reps),
        desc=f"Survey replicates {cy}",
    ):

        w = data["s106"].rescaled_psu_weights(
            q["frame"],
            data["frames"][cy],
            rng,
        )

        weights.append(w / w.sum())

    return np.asarray(weights)


# ============================================================
# Explain predictive deterioration
# ============================================================

def calibration_analysis(
    data, predictions, weights, out
):

    results = {}
    rows = []
    gap_rows = []
    changes = []

    for cy in HOLDOUTS:

        y = data["anchor"][cy]["y"]
        w = weights[cy]

        decomposed = {
            name: decompose_error(
                y, pred[cy], w
            )
            for name, pred in predictions.items()
        }

        # Average decompositions across individual
        # fixed seeds; NOT prediction ensembling.
        names = [
            f"VAE2_seed{s}"
            for s in VAE_SEEDS
        ]

        decomposed["VAE2_mean_loss"] = {
            metric: np.mean([
                decomposed[name][metric]
                for name in names
            ], axis=0)
            for metric in decomposed[names[0]]
        }

        results[cy] = decomposed

        for method, metrics in decomposed.items():

            for metric, draws in metrics.items():

                if np.isfinite(draws).all():

                    rows.append({
                        "cohort": cy,
                        "method": method,
                        "metric": metric,
                        **summarize_draws(draws),
                    })

        for reference in COMPARATORS:

            for metric in ERROR_COMPONENTS:

                # Positive difference favours HG shared2.
                gap = (
                    decomposed[reference][metric]
                    - decomposed[PRIMARY][metric]
                )

                gap_rows.append({
                    "cohort": cy,
                    "reference": reference,
                    "candidate": PRIMARY,
                    "metric": metric,
                    **summarize_draws(gap),
                })

    # Independently resampled survey cycles.
    # Positive change means a larger HG advantage
    # in the distant period than the earlier period.
    for reference in COMPARATORS:

        for metric in ERROR_COMPONENTS:

            early = (
                results["0708"][reference][metric]
                - results["0708"][PRIMARY][metric]
            )

            late = (
                results["2123"][reference][metric]
                - results["2123"][PRIMARY][metric]
            )

            changes.append({
                "reference": reference,
                "metric": metric,
                **summarize_draws(late - early),
            })

    save_table(
        out,
        "calibration",
        rows,
    )

    save_table(
        out,
        "error_gap",
        gap_rows,
    )

    save_table(
        out,
        "error_gap_change",
        changes,
    )

    return (
        results,
        pd.DataFrame(gap_rows),
        pd.DataFrame(changes),
    )


# ============================================================
# Trace raw-marker excess error through HG coordinates
# ============================================================

def build_common_design(data, representations):

    """
    Use one common reference-centred basis containing:

      Intercept + age + BMI + income
      + S1,S2 + D1,D2 + H1,H2 + G1.

    This is a mathematical re-expression of existing
    predictions, not a new ferritin model.
    """

    features = {}

    for cy in CYCLES:

        cov = data["anchor"][cy]["cov"]

        z = representations["HG7"][cy][
            data["masks"][cy]
        ]

        features[cy] = np.column_stack(
            [cov, z]
        )

    w0 = data["anchor"]["0506"]["w"]

    mu = wmean(
        features["0506"],
        w0,
    )

    sd = weighted_sd(
        features["0506"],
        w0,
    )

    if np.any(sd <= 1e-12):

        raise RuntimeError(
            "Degenerate diagnostic design"
        )

    return {
        cy: np.column_stack([
            np.ones(len(features[cy])),
            (features[cy] - mu) / sd,
        ])
        for cy in CYCLES
    }


def affine_prediction_coefficients(
    design, predictions, method
):

    coefficients = np.linalg.lstsq(
        design["0506"],
        predictions[method]["0506"],
        rcond=1e-12,
    )[0]

    # An equivalent linear parameterization should
    # reproduce frozen predictions in EVERY cohort.
    for cy in CYCLES:

        error = np.max(
            np.abs(
                design[cy] @ coefficients
                - predictions[method][cy]
            )
        )

        if error > 1e-6:

            raise RuntimeError(
                f"Prediction re-expression failed: "
                f"{method}, {cy}, error={error}"
            )

    return coefficients


def coordinate_error_analysis(
    data,
    representations,
    predictions,
    weights,
    out,
):

    design = build_common_design(
        data,
        representations,
    )

    raw_coef = affine_prediction_coefficients(
        design,
        predictions,
        "RawX7",
    )

    shared_coef = affine_prediction_coefficients(
        design,
        predictions,
        PRIMARY,
    )

    # Shared model must not directly depend on
    # discordant or private coordinates.
    if np.max(
        np.abs(shared_coef[6:])
    ) > 1e-6:

        raise RuntimeError(
            "Shared model contains unexpected "
            "non-shared coefficient support"
        )

    delta_coef = raw_coef - shared_coef

    names = [
        "intercept",
        "age",
        "BMI",
        "income",
        "S1",
        "S2",
        "D1",
        "D2",
        "H1",
        "H2",
        "G1",
    ]

    save_table(
        out,
        "reexpressed_coefficients",
        [
            {
                "coordinate": name,
                "raw_model_coefficient":
                    raw_coef[j],
                "shared_model_coefficient":
                    shared_coef[j],
                "coefficient_difference":
                    delta_coef[j],
                "units":
                    "log-ferritin per reference SD",
            }
            for j, name in enumerate(names)
        ],
    )

    rows = []
    shift_rows = []
    draws_by_cycle = {}

    for cy in CYCLES:

        raw_pred = predictions["RawX7"][cy]
        shared_pred = predictions[PRIMARY][cy]

        y = data["anchor"][cy]["y"]
        ww = data["anchor"][cy]["w"]

        # Exact difference-of-squares identity:
        #
        # e_raw² - e_shared²
        # = (p_raw - p_shared)
        #   * (p_raw + p_shared - 2y)
        #
        # The first factor is decomposed by coordinate
        # group using the equivalent affine models.

        midpoint_factor = (
            raw_pred + shared_pred - 2 * y
        )

        group_errors = {}
        group_prediction_diffs = {}

        for group, indices in GROUP_NAMES.items():

            idx = np.asarray(indices)

            component = (
                design[cy][:, idx]
                @ delta_coef[idx]
            )

            group_prediction_diffs[group] = (
                component
            )

            group_errors[group] = (
                component * midpoint_factor
            )

            shift_rows.append({
                "cohort": cy,
                "group": group,
                "mean_raw_minus_shared_prediction":
                    float(wmean(component, ww)),
                "sd_prediction_difference":
                    float(weighted_sd(component, ww)),
            })

        observed_gap = (
            (raw_pred - y) ** 2
            - (shared_pred - y) ** 2
        )

        reconstructed_gap = np.sum(
            list(group_errors.values()),
            axis=0,
        )

        if np.max(
            np.abs(
                observed_gap - reconstructed_gap
            )
        ) > 1e-6:

            raise RuntimeError(
                "Group error decomposition failed"
            )

        if cy not in HOLDOUTS:
            continue

        draws_by_cycle[cy] = {
            group: weights[cy] @ error
            for group, error in group_errors.items()
        }

        draws_by_cycle[cy]["total"] = (
            weights[cy] @ observed_gap
        )

        for group, values in (
            draws_by_cycle[cy].items()
        ):

            rows.append({
                "cohort": cy,
                "group": group,
                **summarize_draws(values),
            })

    changes = []

    for group in draws_by_cycle["0708"]:

        values = (
            draws_by_cycle["2123"][group]
            - draws_by_cycle["0708"][group]
        )

        changes.append({
            "group": group,
            **summarize_draws(values),
        })

    save_table(
        out,
        "coordinate_error_contributions",
        rows,
    )

    save_table(
        out,
        "coordinate_error_change",
        changes,
    )

    save_table(
        out,
        "coordinate_prediction_shifts",
        shift_rows,
    )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(changes),
    )


# ============================================================
# Describe distribution shift
# ============================================================

def measurement_shift(
    data, representations, out
):

    profiles = {}

    names = [
        *data["s103"].X_COLS,
        "S1",
        "S2",
        "D1",
        "D2",
        "H1",
        "H2",
        "G1",
        "age",
        "BMI",
        "income",
        "log_ferritin",
    ]

    for cy in CYCLES:

        raw = data["raw"][cy][
            data["masks"][cy]
        ]

        z = representations["HG7"][cy][
            data["masks"][cy]
        ]

        q = data["anchor"][cy]

        matrix = np.column_stack([
            raw,
            z,
            q["cov"],
            q["y"],
        ])

        profiles[cy] = {
            "mean": wmean(matrix, q["w"]),
            "sd": weighted_sd(matrix, q["w"]),
        }

    mu0 = profiles["0506"]["mean"]
    sd0 = profiles["0506"]["sd"]

    rows = []

    for cy in CYCLES:

        mu = profiles[cy]["mean"]
        sd = profiles[cy]["sd"]

        for j, name in enumerate(names):

            rows.append({
                "cohort": cy,
                "variable": name,
                "weighted_mean": mu[j],
                "weighted_sd": sd[j],
                "mean_shift_in_reference_SD":
                    (mu[j] - mu0[j]) / sd0[j],
                "sd_ratio_to_reference":
                    sd[j] / sd0[j],
            })

    save_table(
        out,
        "distribution_shift",
        rows,
    )

    return pd.DataFrame(rows)


# ============================================================
# Scientific figure
# ============================================================

def make_figure(gaps, components, out):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    references = [
        "RawX7",
        "PCA2",
        "H_PCA2",
        "JIVE_Joint2",
        "AJIVE_Joint2",
        "VAE2_mean_loss",
        "Threshold2",
        "Hb_HbA1c",
    ]

    labels = [
        "Raw markers",
        "PCA",
        "Haematology PCA",
        "JIVE",
        "AJIVE",
        "Multi-decoder VAE",
        "Threshold burdens",
        "Hb + HbA1c",
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5.5),
    )

    # Positive components increase comparator MSE
    # relative to the shared-pair model.
    parts = [
        "mean_bias_error",
        "slope_error",
        "residual_error",
    ]

    colors = {
        "mean_bias_error": "#BD7852",
        "slope_error": "#D1B06F",
        "residual_error": "#367D80",
    }

    part_labels = {
        "mean_bias_error": "Mean bias",
        "slope_error": "Slope calibration",
        "residual_error": "Residual variation",
    }

    for ax, cy, panel_title in zip(
        axes[:2],
        HOLDOUTS,
        ["a   2007–2008", "b   2021–2023"],
    ):

        df = gaps.loc[
            gaps.cohort.eq(cy)
        ]

        y = np.arange(len(references))

        positive = np.zeros(len(y))
        negative = np.zeros(len(y))

        for metric in parts:

            values = (
                df.loc[df.metric.eq(metric)]
                .set_index("reference")
                .reindex(references)
                .estimate.to_numpy(float)
            )

            left = np.where(
                values >= 0,
                positive,
                negative,
            )

            ax.barh(
                y,
                values,
                left=left,
                height=0.55,
                color=colors[metric],
                label=part_labels[metric],
            )

            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)

        total = (
            df.loc[df.metric.eq("mse")]
            .set_index("reference")
            .reindex(references)
        )

        ax.scatter(
            total.estimate,
            y,
            color="black",
            s=20,
            zorder=5,
            label="Total MSE gap",
        )

        ax.axvline(
            0,
            color="black",
            linewidth=0.7,
        )

        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(panel_title, loc="left")
        ax.set_xlabel("Comparator minus HG shared MSE")

    # Third panel: exact decomposition of the
    # increase in raw-marker disadvantage.
    groups = [
        "intercept",
        "covariates",
        "shared",
        "discordant",
        "H_private",
        "G_private",
        "total",
    ]

    group_labels = [
        "Intercept",
        "Covariates",
        "Shared",
        "Discordant",
        "Haematology-private",
        "Glycaemia-private",
        "Total",
    ]

    df = (
        components
        .set_index("group")
        .reindex(groups)
    )

    ax = axes[2]

    y = np.arange(len(groups))

    ax.hlines(
        y,
        df.psu_low95,
        df.psu_high95,
        color="#777777",
    )

    ax.scatter(
        df.estimate,
        y,
        color="#176D69",
        s=25,
        zorder=3,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_yticks(y, group_labels)
    ax.invert_yaxis()
    ax.set_title(
        "c   Change in raw-marker error gap",
        loc="left",
    )

    ax.set_xlabel("2021–23 minus 2007–08")

    for ax in axes:

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    fig.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    fig.savefig(
        out / "119_drift_explanation.png",
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        out / "119_drift_explanation.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Self-test
# ============================================================

def self_test():

    rng = np.random.default_rng(SEED)

    y = rng.normal(size=1000)
    p = 2.5 * y + 1.8

    w = np.ones((1, len(y))) / len(y)

    metrics = decompose_error(y, p, w)

    # Perfect association, severely wrong calibration.
    assert abs(
        metrics["pearson_r"][0] - 1.0
    ) < 1e-10

    assert (
        metrics["residual_error"][0]
        < 1e-10
    )

    assert (
        metrics["calibration_error"][0]
        > 1.0
    )

    assert abs(
        metrics["mse"][0]
        - metrics["calibration_error"][0]
    ) < 1e-10

    # Independent predictions: substantial residual
    # variation remains after affine recalibration.
    p2 = rng.normal(size=len(y))

    metrics2 = decompose_error(y, p2, w)

    assert (
        metrics2["residual_error"][0]
        > 0.8
    )

    a = rng.normal(size=len(y))
    b = rng.normal(size=len(y))

    assert np.max(
        np.abs(
            (a - y) ** 2
            - (b - y) ** 2
            - (a - b) * (a + b - 2 * y)
        )
    ) < 1e-10

    print("PASS: exact MSE decomposition")
    print("PASS: calibration failure control")
    print("PASS: association failure control")
    print("PASS: coordinate error identity")


# ============================================================
# Entry point
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=999,
    )

    parser.add_argument(
        "--reference-run",
        type=Path,
        default=(
            EDA / "Results" /
            "118_biomedical_utility"
        ),
    )

    parser.add_argument(
        "--run-name",
        default="119_temporal_drift_explanation",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:

        self_test()
        return

    if args.replicates < 39:

        parser.error(
            "At least 39 replicates required"
        )

    if args.tolerance <= 0:

        parser.error("Invalid tolerance")

    if not args.run_name or not all(
        c.isalnum() or c in "_-"
        for c in args.run_name
    ):

        parser.error("Invalid run name")

    out = (
        EDA / "Results" / args.run_name
    )

    if out.exists():

        raise FileExistsError(
            f"{out} already exists. "
            "Choose another --run-name."
        )

    out.mkdir(parents=True)

    print("Reproducing Script118", flush=True)

    (
        data,
        representations,
        predictions,
        protocol,
        source,
    ) = reproduce_118(args, out)

    dump_json(
        out / "119_protocol.json",
        {
            "seed": SEED,
            "settings": vars(args),
            "inherited_protocol": protocol,
            "118_source_sha256":
                hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
            "primary_question":
                "Calibration versus residual "
                "association failure",
            "coordinate_attribution":
                "Exact algebraic decomposition of "
                "the difference of squared errors",
            "new_representation_fit":
                "Reproduction of Script118 only",
            "uncertainty":
                "Paired holdout PSU bootstrap; "
                "conditional on reproduced fits",
            "oracle_recalibration":
                "Descriptive diagnostic only",
            "multiplicity_adjusted": False,
            "causal_identification": False,
        },
    )

    weights = {
        cy: bootstrap_weights(
            data, cy, args.replicates
        )
        for cy in HOLDOUTS
    }

    print("Decomposing prediction error", flush=True)

    calibration, gaps, changes = (
        calibration_analysis(
            data,
            predictions,
            weights,
            out,
        )
    )

    print("Tracing coordinate contributions", flush=True)

    contributions, contribution_changes = (
        coordinate_error_analysis(
            data,
            representations,
            predictions,
            weights,
            out,
        )
    )

    shifts = measurement_shift(
        data,
        representations,
        out,
    )

    make_figure(
        gaps,
        contribution_changes,
        out,
    )

    focus = [
        "RawX7",
        "VAE2_mean_loss",
        "H_PCA2",
        "PCA2",
        "AJIVE_Joint2",
        "JIVE_Joint2",
        "Threshold2",
    ]

    selected = gaps.loc[
        gaps.reference.isin(focus)
    ]

    change_focus = changes.loc[
        changes.reference.isin(focus)
    ]

    readout = [
        "# Script119 — Temporal drift explanation",
        "",
        "## Reproduction",
        "",
        "Original Script118 weighted MSE results reproduced.",
        "",
        "## Error decomposition",
        "",
        "Positive differences favour HG shared2.",
        "",
        "```text",
        selected.round(5).to_string(index=False),
        "```",
        "",
        "## Change in relative performance across periods",
        "",
        "```text",
        change_focus.round(5).to_string(index=False),
        "```",
        "",
        "## Raw-marker error contribution by coordinate group",
        "",
        "```text",
        contributions.round(5).to_string(index=False),
        "```",
        "",
        "## Change in coordinate-level error contribution",
        "",
        "```text",
        contribution_changes.round(5).to_string(index=False),
        "```",
        "",
        "## Distribution shift",
        "",
        "```text",
        shifts.loc[
            shifts.cohort.eq("2123")
        ].round(5).to_string(index=False),
        "```",
        "",
        "## Interpretation boundaries",
        "",
        "- Affine recalibration is fitted descriptively "
        "on the evaluation distribution; it is not a "
        "prospective correction.",
        "- Residual error is the remaining variance "
        "after optimal affine recalibration.",
        "- Lower residual error does not imply causality.",
        "- Coordinate contributions depend on the "
        "specified reference parameterization.",
        "- Contributions are signed algebraic terms, "
        "not independent biological effects.",
        "- VAE metrics average the five fixed-seed "
        "decompositions, not predictions.",
        "- Population shift, assay changes and selection "
        "cannot be causally separated by this analysis.",
        "- PSU intervals condition on frozen model fits.",
        "- Exploratory comparisons are unadjusted "
        "for multiplicity.",
        "",
    ]

    (out / "119_READOUT.md").write_text(
        "\n".join(readout),
        encoding="utf-8",
    )

    dump_json(
        out / "119_completion.json",
        {
            "completed": True,
            "118_reproduction_passed": True,
            "mse_identity_verified": True,
            "coordinate_identity_verified": True,
            "replicates": args.replicates,
            "methods": list(predictions),
        },
    )

    print()
    print("=" * 70)
    print("SCRIPT 119 COMPLETE")
    print("=" * 70)

    print(
        change_focus.round(5)
        .to_string(index=False)
    )

    print()
    print(
        contribution_changes.round(5)
        .to_string(index=False)
    )

    print()
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
