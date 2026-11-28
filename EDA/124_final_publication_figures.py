
#!/usr/bin/env python3
"""
124 | Final HG7 publication figure suite

Uses saved experimental results only.
No model fitting or statistical recalculation.
Preserves Script121 figures.
Exports PNG exclusively.
"""

from pathlib import Path
import shutil
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

EDA = Path(__file__).resolve().parent
R = EDA / "Results"

R121 = R / "121_manuscript_figures"
R122 = R / "122_statistical_robustness"
R123 = R / "123_biological_specificity"

OUT = R / "124_final_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

INK = "#282E33"
SHARED = "#146F69"
DISCORD = "#AC583C"
HEM = "#47677E"
GLYC = "#8A9D91"
GREY = "#92999E"
GOLD = "#C4A16A"
WHITE = "#FFFFFF"

GROUPS = ["Shared", "Discordant", "H_private", "G_private"]
GLABELS = ["Shared", "Discordant", "H-private", "G-private"]
GCOLORS = [SHARED, DISCORD, HEM, GLYC]

PERIODS = ["2005-2008", "2009-2018", "2021-2023"]
PLABELS = ["2005–08", "2009–18", "2021–23"]

CYCLES = ["0506", "0708", "2123"]
CLABELS = ["2005–06", "2007–08", "2021–23"]

ANCHORS = ["Ferritin", "sTfR", "CRP", "OGTT"]
OUTCOMES = ["PHQ_total", "Somatic", "CogAff"]
OLABELS = ["PHQ total", "Somatic", "Cognitive-affective"]

METHODS = [
    "Scalar", "ScalarInteraction",
    "Continuous2", "Raw7", "HG7"
]

MLABELS = {
    "Scalar": "Thresholds",
    "ScalarInteraction": "Thresholds + interaction",
    "Continuous2": "Hb + HbA1c",
    "Raw7": "Raw7",
    "HG7": "HG7",
}

MCOLORS = {
    "Scalar": GREY,
    "ScalarInteraction": DISCORD,
    "Continuous2": GOLD,
    "Raw7": INK,
    "HG7": SHARED,
}

CMAP = LinearSegmentedColormap.from_list(
    "HG_sequential",
    [WHITE, "#DCEAE6", "#91BCB3", SHARED]
)

DIVERGING = LinearSegmentedColormap.from_list(
    "HG_diverging",
    [DISCORD, "#F7F7F5", SHARED]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Liberation Sans", "DejaVu Sans"
    ],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.5,
    "axes.titlelocation": "left",
    "axes.linewidth": 0.65,
    "axes.edgecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.facecolor": WHITE,
    "savefig.dpi": 600,
})


# ============================================================
# Utilities
# ============================================================

def read(folder, filename):
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(path)

    d = pd.read_csv(path, dtype={"cycle": str})

    if "cycle" in d:
        d["cycle"] = d["cycle"].str.zfill(4)

    return d


def figure(rows, cols, height, **kwargs):
    return plt.subplots(
        rows, cols,
        figsize=(180 * MM, height * MM),
        layout="constrained",
        **kwargs
    )


def panel(ax, letter, title):
    ax.set_title(
        rf"$\bf{{{letter}}}$  {title}",
        loc="left",
        pad=7
    )


def save(fig, filename):
    fig.savefig(
        OUT / f"{filename}.png",
        dpi=600,
        facecolor=WHITE
    )
    plt.close(fig)
    print("Saved:", filename)


def zero(ax):
    ax.axvline(
        0, color=GREY,
        ls="--", lw=0.7, zorder=0
    )


def period_line(ax, d, metric, methods):
    for method in methods:

        q = (
            d.loc[d.method.eq(method)]
            .set_index("period")
            .loc[PERIODS]
        )

        ax.plot(
            range(3), q[metric],
            marker="o",
            ms=4,
            lw=1.1,
            color=MCOLORS[method],
            label=MLABELS[method]
        )

    ax.set_xticks(range(3), PLABELS)


def matrix(ax, values, labels=None,
           cmap=CMAP, vmin=0, vmax=None,
           fmt=".2f", suffix=None):

    a = np.asarray(values, float)

    cm = cmap.copy()
    cm.set_bad("#ECEFEE")

    im = ax.imshow(
        a, cmap=cm,
        vmin=vmin, vmax=vmax,
        aspect="auto",
        interpolation="nearest"
    )

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):

            v = a[i, j]

            if not np.isfinite(v):
                text = "–"
                color = GREY
            else:
                text = format(v, fmt)

                if suffix is not None:
                    text += suffix[i, j]

                rgba = im.cmap(im.norm(v))

                lum = (
                    .2126 * rgba[0]
                    + .7152 * rgba[1]
                    + .0722 * rgba[2]
                )

                color = WHITE if lum < .48 else INK

            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=7,
                color=color
            )

    for s in ax.spines.values():
        s.set_visible(False)

    ax.tick_params(length=0)

    return im


# ============================================================
# Retain approved Script121 figures
# ============================================================

def preserve_previous_figures():

    mapping = {
        "02_representation_and_recovery":
            "Fig02_representation_and_recovery",

        "04_structure_and_transport":
            "Fig06_structure_and_transport",

        "05_biomedical_utility":
            "Fig05_biomedical_utility",

        "03_information_and_prediction":
            "FigS01_information_and_prediction",

        "S01_physiological_shift":
            "FigS02_physiological_shift",

        "06_temporal_error_attribution":
            "FigS03_temporal_error_attribution",
    }

    for source, target in mapping.items():

        path = R121 / f"{source}.png"

        if not path.is_file():
            raise FileNotFoundError(path)

        shutil.copy2(
            path, OUT / f"{target}.png"
        )

        print("Retained:", target)


# ============================================================
# Load final audits
# ============================================================

diag = read(R122, "122_regression_diagnostics.csv")
sens = read(R122, "122_sensitivity_summary.csv")
struct = read(R122, "122_structural_robustness.csv")
resid = read(R122, "122_residual_profiles.csv")

wald = read(R123, "123_conditional_wald.csv")
rotate = read(R123, "123_random_rotation_controls.csv")
paired = read(R123, "123_frozen_paired_contrasts.csv")
cv = read(R123, "123_cv_summary.csv")

primary = wald.loc[
    wald.model.eq("Primary")
].copy()

assert len(primary) == 44


# ============================================================
# FIGURE 3
# Numerical and structural robustness
# ============================================================

def fig_robustness():

    fig, axes = figure(2, 2, 125)

    q = diag.loc[diag.outcome.eq("PHQ_total")]

    # A: Maximum leverage
    ax = axes[0, 0]

    period_line(
        ax, q, "max_leverage",
        ["Scalar", "ScalarInteraction",
         "Continuous2", "Raw7"]
    )

    panel(ax, "a", "Regression leverage")
    ax.set_ylabel("Maximum leverage")

    handles, labels = ax.get_legend_handles_labels()
    labels[-1] = "Raw7 = HG7"

    ax.legend(
        handles, labels,
        frameon=False,
        fontsize=6.5
    )

    # B: Coordinate conditioning
    ax = axes[0, 1]

    for j, method in enumerate(["Raw7", "HG7"]):

        sub = (
            q.loc[q.method.eq(method)]
            .set_index("period")
            .loc[PERIODS]
        )

        ax.bar(
            np.arange(3) + (j - .5) * .28,
            sub.max_nonintercept_VIF,
            width=.28,
            color=MCOLORS[method],
            label=method
        )

    ax.set_xticks(range(3), PLABELS)
    ax.set_ylabel("Maximum non-intercept VIF")
    panel(ax, "b", "Parameterisation conditioning")
    ax.legend(frameon=False)

    # C: Regression perturbation sensitivity
    ax = axes[1, 0]

    for method in ["Scalar", "HG7"]:

        for kind, marker, style in [
            ("delete_PSU", "o", "-"),
            ("high_cook", "s", "--"),
        ]:

            sub = sens.loc[
                sens.method.eq(method)
                & sens.outcome.eq("PHQ_total")
                & sens.sensitivity.eq(kind)
            ].set_index("period").loc[PERIODS]

            ax.plot(
                range(3),
                sub.maximum_prediction_shift,
                color=MCOLORS[method],
                marker=marker,
                ls=style,
                lw=1.1,
                ms=4,
                label=(
                    f"{MLABELS[method]} / "
                    + ("PSU" if kind == "delete_PSU"
                       else "Cook")
                )
            )

    panel(ax, "c", "Sensitivity of fitted values")
    ax.set_xticks(range(3), PLABELS)
    ax.set_ylabel("Prediction-shift RMSE\n(PHQ points)")
    ax.legend(frameon=False, fontsize=6.3)

    # D: Stability of designated subspaces
    ax = axes[1, 1]

    groups = [
        "Shared",
        "Discordant",
        "Haematology-private",
        "Glycaemia-private",
    ]

    for i, group in enumerate(groups):

        for kind, marker, offset in [
            ("delete_PSU", "o", -.19),
            ("high_physiological_leverage", "D", 0),
            ("high_survey_weight", "s", .19),
        ]:

            sub = struct.loc[
                struct.group.eq(group)
                & struct.perturbation.eq(kind)
                & struct.status.eq("completed"),
                "subspace_similarity"
            ].to_numpy()

            if not len(sub):
                continue

            ax.scatter(
                sub, np.full(len(sub), i + offset),
                s=9 if kind == "delete_PSU" else 24,
                marker=marker,
                color=GCOLORS[i],
                alpha=.35 if kind == "delete_PSU" else 1
            )

            if kind == "delete_PSU":
                ax.scatter(
                    np.median(sub), i + offset,
                    color=GCOLORS[i],
                    edgecolor=INK,
                    s=24, zorder=3
                )

    ax.set_yticks(range(4), GLABELS)
    ax.invert_yaxis()
    ax.set_xlim(.980, 1.001)
    ax.set_xlabel("Subspace similarity")
    ax.axvline(1, color=GREY, lw=.7, ls="--")
    panel(ax, "d", "Physiological structure stability")

    ax.legend(
        handles=[
            Line2D([], [], marker=m, color=INK,
                   ls="none", label=l, ms=4)
            for m, l in [
                ("o", "PSU deletion"),
                ("D", "High physiological leverage"),
                ("s", "High survey weight"),
            ]
        ],
        frameon=False,
        fontsize=6.3
    )

    save(fig, "Fig03_numerical_structural_robustness")


# ============================================================
# FIGURE 4
# Independent biological construct validation
# ============================================================

def fig_biological():

    fig = plt.figure(
        figsize=(180 * MM, 135 * MM),
        layout="constrained"
    )

    gs = fig.add_gridspec(
        2, 6, height_ratios=[1, 1.25]
    )

    axes = [
        fig.add_subplot(gs[0, 2*j:2*j+2])
        for j in range(3)
    ]

    vmax = 100 * primary.partial_R2.max()

    for j, (ax, cycle) in enumerate(zip(axes, CYCLES)):

        sub = primary.loc[primary.cycle.eq(cycle)]

        values = sub.pivot(
            index="anchor", columns="group",
            values="partial_R2"
        ).reindex(index=ANCHORS, columns=GROUPS)

        pvals = sub.pivot(
            index="anchor", columns="group",
            values="p_holm"
        ).reindex(index=ANCHORS, columns=GROUPS)

        stars = np.where(
            pvals.to_numpy() < .05, "*", ""
        )

        im = matrix(
            ax, 100 * values.to_numpy(),
            vmax=vmax, fmt=".1f", suffix=stars
        )

        ax.set_xticks(range(4), ["S", "D", "H", "G"])
        ax.set_yticks(range(4), ANCHORS if j == 0 else [])
        panel(ax, "abc"[j], CLABELS[j])

    fig.colorbar(
        im, ax=axes,
        location="bottom",
        shrink=.7, aspect=35,
        label="Conditional partial R² (%)"
    )

    # D: Replication of iron-associated signatures
    ax = fig.add_subplot(gs[1, :3])

    for anchor, marker in [
        ("Ferritin", "o"), ("sTfR", "s")
    ]:
        for group, color in [
            ("Shared", SHARED),
            ("Discordant", DISCORD)
        ]:

            q = primary.loc[
                primary.anchor.eq(anchor)
                & primary.group.eq(group)
            ].set_index("cycle").loc[CYCLES]

            ax.plot(
                range(3),
                100 * q.partial_R2,
                color=color,
                marker=marker,
                lw=1.2, ms=4,
                label=f"{anchor}: {group}"
            )

    panel(ax, "d", "Iron-associated signatures")
    ax.set_xticks(range(3), CLABELS)
    ax.set_ylabel("Conditional partial R² (%)")
    ax.legend(frameon=False, fontsize=6.5)

    # E: Glycaemic versus haematological specificity
    ax = fig.add_subplot(gs[1, 3:])

    for i, (group, color) in enumerate(
        zip(GROUPS, GCOLORS)
    ):

        q = primary.loc[
            primary.anchor.eq("OGTT")
            & primary.group.eq(group)
        ].set_index("cycle").loc[["0506", "0708"]]

        ax.plot(
            [0, 1], 100 * q.partial_R2,
            marker="o", color=color,
            lw=1.3, ms=4,
            label=GLABELS[i]
        )

    panel(ax, "e", "Glucose-regulation specificity")
    ax.set_xticks([0, 1], CLABELS[:2])
    ax.set_ylabel("Conditional partial R² (%)")
    ax.legend(frameon=False, fontsize=6.5)

    save(fig, "Fig04_biological_construct_validation")


# ============================================================
# SUPPLEMENT S4
# Complete influence screening
# ============================================================

def fig_influence():

    fig, axes = figure(3, 3, 150)

    metrics = [
        ("n_leverage_above_2p_n", "Leverage"),
        ("n_cook_above_4_n", "Cook's distance"),
        ("n_dfbeta_above_2_sqrt_n", "DFBETAS"),
    ]

    for j, (metric, label) in enumerate(metrics):

        d = diag.copy()
        d["rate"] = 100 * d[metric] / d.n

        vmax = max(d.rate.max(), .01)

        for i, outcome in enumerate(OUTCOMES):

            ax = axes[i, j]

            q = d.loc[d.outcome.eq(outcome)]

            table = q.pivot(
                index="method", columns="period",
                values="rate"
            ).reindex(index=METHODS, columns=PERIODS)

            im = matrix(
                ax, table, vmax=vmax, fmt=".1f"
            )

            ax.set_xticks(
                range(3), ["05–08", "09–18", "21–23"]
            )

            ax.set_yticks(
                range(5),
                [MLABELS[m] for m in METHODS]
                if j == 0 else []
            )

            panel(
                ax, "abcdefghi"[3*i+j],
                f"{OLABELS[i]} / {label}"
            )

        fig.colorbar(
            im, ax=axes[:, j].tolist(),
            location="bottom",
            shrink=.8,
            label="Flagged participants (%)"
        )

    save(fig, "FigS04_influence_screening")


# ============================================================
# SUPPLEMENT S5–S6
# Residual profiles
# ============================================================

def fig_residuals(metric, ylabel, filename):

    fig, axes = figure(3, 3, 150, sharex=True)

    for i, outcome in enumerate(OUTCOMES):

        for j, period in enumerate(PERIODS):

            ax = axes[i, j]

            for method in [
                "Scalar", "ScalarInteraction", "Raw7"
            ]:

                q = resid.loc[
                    resid.outcome.eq(outcome)
                    & resid.period.eq(period)
                    & resid.method.eq(method)
                ].sort_values("fitted_decile")

                ax.plot(
                    q.fitted_decile, q[metric],
                    color=MCOLORS[method],
                    lw=1.1, marker="o", ms=2.5,
                    label=(
                        "Raw7 = HG7"
                        if method == "Raw7"
                        else MLABELS[method]
                    )
                )

            if metric == "weighted_residual_mean":
                ax.axhline(
                    0, color=GREY, ls="--", lw=.7
                )

            panel(
                ax, "abcdefghi"[3*i+j],
                f"{OLABELS[i]} / {PLABELS[j]}"
            )

            if j == 0:
                ax.set_ylabel(ylabel)

            if i == 2:
                ax.set_xlabel("Fitted-value decile")

    axes[0, 0].legend(
        frameon=False, fontsize=6
    )

    save(fig, filename)


# ============================================================
# SUPPLEMENT S7
# Random-rotation specificity
# ============================================================

def fig_rotations():

    row_keys = [
        (a, c)
        for a in ANCHORS
        for c in ["0708", "2123"]
        if not (a == "OGTT" and c == "2123")
    ]

    fig, axes = figure(1, 2, 88)

    for j, kind in enumerate(
        ["marginal", "conditional"]
    ):

        ax = axes[j]
        values = []

        for anchor, cycle in row_keys:

            q = rotate.loc[
                rotate.anchor.eq(anchor)
                & rotate.cycle.eq(cycle)
                & rotate.comparison.eq(kind)
            ].set_index("group").reindex(GROUPS)

            values.append(q.HG_percentile.to_numpy())

        im = matrix(
            ax, values,
            cmap=DIVERGING,
            vmin=0, vmax=100,
            fmt=".0f"
        )

        ax.set_xticks(range(4), ["S", "D", "H", "G"])

        ax.set_yticks(
            range(len(row_keys)),
            [f"{a} / {c}" for a, c in row_keys]
            if j == 0 else []
        )

        panel(
            ax, "ab"[j],
            kind.capitalize() + " contribution"
        )

    fig.colorbar(
        im, ax=axes.tolist(),
        location="bottom", shrink=.75,
        label="Percentile among 199 random subspaces"
    )

    save(fig, "FigS07_random_rotation_specificity")


# ============================================================
# SUPPLEMENT S8–S11
# Complete frozen assay comparisons
# ============================================================

def fig_anchor_forest(anchor, number):

    fig, axes = figure(1, 3, 75)

    comparisons = [
        ("group_vs_covariates", "Against covariates"),
        ("group_vs_PCA", "Against matched PCA"),
        ("conditional_group_contribution",
         "Conditional contribution"),
    ]

    for j, (comparison, title) in enumerate(comparisons):

        ax = axes[j]
        zero(ax)

        for i, (group, color) in enumerate(
            zip(GROUPS, GCOLORS)
        ):

            for cycle, marker, offset in [
                ("0708", "o", -.13),
                ("2123", "s", .13)
            ]:

                q = paired.loc[
                    paired.anchor.eq(anchor)
                    & paired.cycle.eq(cycle)
                    & paired.group.eq(group)
                    & paired.comparison.eq(comparison)
                ]

                if q.empty:
                    continue

                row = q.iloc[0]
                yy = i + offset

                ax.hlines(
                    yy, row.low95, row.high95,
                    color=color, lw=1.1
                )

                ax.scatter(
                    row.gain, yy,
                    marker=marker,
                    s=25,
                    facecolor=(
                        WHITE if cycle == "0708" else color
                    ),
                    edgecolor=color,
                    linewidth=.9,
                    zorder=3
                )

        ax.set_yticks(
            range(4), GLABELS if j == 0 else []
        )

        ax.invert_yaxis()
        ax.set_xlabel("MSE reduction")
        panel(ax, "abc"[j], title)

    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", ls="none",
                   mfc=WHITE, mec=INK,
                   label="2007–08", ms=4),
            Line2D([], [], marker="s", ls="none",
                   mfc=INK, mec=INK,
                   label="2021–23", ms=4),
        ],
        frameon=False,
        fontsize=6.5
    )

    fig.suptitle(anchor)

    save(
        fig,
        f"FigS{number:02d}_{anchor}_frozen_contrasts"
    )


# ============================================================
# SUPPLEMENT S12
# Measured-confounder sensitivity
# ============================================================

def fig_adjustment():

    keys = ["anchor", "cycle", "group"]

    base = wald.loc[
        wald.model.eq("Sensitivity_base")
    ]

    expanded = wald.loc[
        wald.model.eq("Sensitivity_expanded")
    ]

    q = base.merge(
        expanded,
        on=keys,
        suffixes=("_base", "_expanded"),
        validate="one_to_one"
    )

    q = q.dropna(
        subset=[
            "partial_R2_base",
            "partial_R2_expanded"
        ]
    )

    if q.empty:
        raise RuntimeError(
            "No matched adjustment comparisons available"
        )

    fig, axes = figure(2, 2, 120)

    bound = 100 * max(
        q.partial_R2_base.max(),
        q.partial_R2_expanded.max()
    )

    for i, anchor in enumerate(ANCHORS):

        ax = axes.flat[i]

        ax.plot(
            [0, bound], [0, bound],
            color=GREY, lw=.8, ls="--"
        )

        for group, color in zip(GROUPS, GCOLORS):

            sub = q.loc[
                q.anchor.eq(anchor)
                & q.group.eq(group)
            ]

            ax.scatter(
                100 * sub.partial_R2_base,
                100 * sub.partial_R2_expanded,
                color=color,
                s=28,
                label=group
            )

        ax.set_xlim(0, bound * 1.05)
        ax.set_ylim(0, bound * 1.05)

        ax.set_xlabel("Base adjustment, partial R² (%)")
        ax.set_ylabel("Expanded adjustment, partial R² (%)")
        panel(ax, "abcd"[i], anchor)

    axes[0, 0].legend(
        frameon=False, fontsize=6.5
    )

    save(fig, "FigS12_confounder_adjustment")


# ============================================================
# SUPPLEMENT S13
# Fold-specific representation leakage sensitivity
# ============================================================

def fig_cv():

    fig, ax = plt.subplots(
        figsize=(180 * MM, 65 * MM),
        layout="constrained"
    )

    for i, outcome in enumerate(OUTCOMES):

        q = cv.loc[
            cv.outcome.eq(outcome)
        ].set_index("method")

        reference = q.loc[
            "DiscoveryWide_HG7",
            "validation_MSE"
        ]

        for method, offset, color, marker in [
            ("FoldRefitted_HG7", -.13, SHARED, "o"),
            ("Raw7", .13, INK, "s"),
        ]:

            value = (
                q.loc[method, "validation_MSE"]
                - reference
            ) * 1e4

            yy = i + offset

            ax.scatter(
                value, yy,
                color=color, marker=marker,
                s=30
            )

            ax.annotate(
                f"{value:+.2f}",
                (value, yy),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7
            )

    zero(ax)
    ax.set_yticks(range(3), OLABELS)
    ax.invert_yaxis()
    ax.margins(x=.18)

    ax.set_xlabel(
        r"CV MSE difference from discovery-wide HG7"
        r" ($\times 10^{-4}$)"
    )

    panel(ax, "a", "Fold-specific representation fitting")

    ax.legend(
        handles=[
            Line2D([], [], color=SHARED, marker="o",
                   ls="none", label="Fold-refitted HG7"),
            Line2D([], [], color=INK, marker="s",
                   ls="none", label="Raw7"),
        ],
        frameon=False
    )

    save(fig, "FigS13_fold_specific_leakage_sensitivity")


# ============================================================
# Execute
# ============================================================

if __name__ == "__main__":

    preserve_previous_figures()

    fig_robustness()
    fig_biological()
    fig_influence()

    fig_residuals(
        "weighted_residual_mean",
        "Mean residual",
        "FigS05_residual_mean_profiles"
    )

    fig_residuals(
        "weighted_residual_RMSE",
        "Residual RMSE",
        "FigS06_residual_variance_profiles"
    )

    fig_rotations()

    for number, anchor in enumerate(
        ANCHORS, start=8
    ):
        fig_anchor_forest(anchor, number)

    fig_adjustment()
    fig_cv()

    files = sorted(
        p.name for p in OUT.glob("*.png")
    )

    (OUT / "124_figure_manifest.json").write_text(
        json.dumps({
            "figures": files,
            "figure_01": "Existing methodology schematic",
            "format": "PNG",
            "dpi": 600,
            "new_model_fitting": False,
            "source_scripts": [121, 122, 123],
        }, indent=2),
        encoding="utf-8"
    )

    print("\nFINAL PUBLICATION FIGURES COMPLETE")
    print("PNG files:", len(files))
    print("Output:", OUT)
