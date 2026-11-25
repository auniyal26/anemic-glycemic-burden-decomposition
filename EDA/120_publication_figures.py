
#!/usr/bin/env python3
"""
120 | Publication figures — redesigned

Data-driven visualisations from Scripts 118–119.
No model refitting. No approximated results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

from matplotlib.ticker import MaxNLocator


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent / "Results"

R118 = ROOT / "118_biomedical_utility"
R119 = ROOT / "119_temporal_drift_explanation"

OUT = ROOT / "120_publication_figures" / "revised"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#087F8C"
RUST = "#C15C3D"
BLUE = "#426C95"
PURPLE = "#8064A2"
GOLD = "#C99C42"
GREY = "#92999E"
INK = "#23292D"
PALE = "#F1F3F4"

MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
})

PERIODS = ["0708", "2123"]

LABELS = {
    "HG_Shared2": "HG shared",
    "Threshold2": "Threshold burdens",
    "Hb_HbA1c": "Hb + HbA1c",
    "Hb_RDW": "Hb + RDW",
    "PCA2": "PCA",
    "H_PCA2": "Haematology PCA",
    "G_PCA2": "Glycaemia PCA",
    "JIVE_Joint2": "JIVE joint",
    "AJIVE_Joint2": "AJIVE joint",
    "VAE2_mean_loss": "Multi-decoder VAE",
    "RawX7": "Raw markers",
}


# ============================================================
# Utilities
# ============================================================

def read(path):
    df = pd.read_csv(path, dtype={"cohort": str})

    if "cohort" in df:
        df["cohort"] = df["cohort"].str.zfill(4)

    return df


def save(fig, name):

    for ext in ("png", "pdf", "svg"):

        fig.savefig(
            OUT / f"{name}.{ext}",
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )

    plt.close(fig)

    print(f"Saved {name}")


def panel(ax, letter, title):

    ax.set_title(
        rf"$\bf{{{letter}}}$   {title}",
        loc="left",
        pad=10,
    )


def clean(ax):

    ax.spines[["top", "right"]].set_visible(False)

    ax.tick_params(
        direction="out",
        length=3,
        width=0.65,
    )


def row(df, **conditions):

    mask = np.ones(len(df), dtype=bool)

    for key, value in conditions.items():
        mask &= df[key].eq(value).to_numpy()

    selected = df.loc[mask]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {conditions}, "
            f"found {len(selected)}"
        )

    return selected.iloc[0]


def signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "−")


def ci_text(lo, hi, digits=2):
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]".replace("-", "−")


# ============================================================
# FIGURE 6
# Calibration-discrimination trade-off
# ============================================================

def figure_biomedical():

    perf = read(R118 / "118_ferritin_performance.csv")

    pairs = read(R118 / "118_paired_comparisons.csv")

    pairs = pairs.loc[
        pairs.candidate.eq("HG_Shared2")
    ]

    # Main methods shown in the trade-off maps.
    styles = {
        "HG_Shared2": (TEAL, "D", 62),
        "RawX7": (INK, "s", 44),
        "VAE2_mean_loss": (PURPLE, "^", 48),
        "JIVE_Joint2": (RUST, "o", 44),
        "AJIVE_Joint2": (GOLD, "P", 48),
        "Threshold2": (GREY, "X", 44),
    }

    methods = list(styles)

    comparisons = [
        "Threshold2",
        "Hb_HbA1c",
        "Hb_RDW",
        "PCA2",
        "H_PCA2",
        "G_PCA2",
        "JIVE_Joint2",
        "AJIVE_Joint2",
        "VAE2_mean_loss",
        "RawX7",
    ]

    fig = plt.figure(
        figsize=(180 * MM, 170 * MM)
    )

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1.5],
        hspace=0.52,
        wspace=0.24,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]

    focus = perf.loc[
        perf.method.isin(methods)
        & perf.cohort.isin(PERIODS)
    ]

    x_min = focus.weighted_mse.min() - 0.025
    x_max = focus.weighted_mse.max() + 0.025

    y_min = focus.low_ferritin_weighted_AUC.min() - 0.025
    y_max = focus.low_ferritin_weighted_AUC.max() + 0.025

    # A-B: each point represents quantitative prediction
    # accuracy AND low-ferritin discrimination.
    for ax, cy, letter, period in zip(
        axes,
        PERIODS,
        ["a", "b"],
        ["2007–2008", "2021–2023"],
    ):

        for method, (color, marker, size) in styles.items():

            r = row(
                perf,
                cohort=cy,
                method=method,
            )

            ax.scatter(
                r.weighted_mse,
                r.low_ferritin_weighted_AUC,
                marker=marker,
                s=size,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

        # Reference lines through HG shared performance.
        hg = row(
            perf,
            cohort=cy,
            method="HG_Shared2",
        )

        ax.axvline(
            hg.weighted_mse,
            color=TEAL,
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
        )

        ax.axhline(
            hg.low_ferritin_weighted_AUC,
            color=TEAL,
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Weighted log-ferritin MSE")

        ax.xaxis.set_major_locator(
            MaxNLocator(4)
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(5)
        )

        ax.grid(
            alpha=0.12,
            linewidth=0.6,
        )

        panel(ax, letter, period)
        clean(ax)

    axes[0].set_ylabel(
        "Low-ferritin AUC"
    )

    axes[1].tick_params(labelleft=False)

    handles = [
        Line2D(
            [], [],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=7,
            label=LABELS[method],
        )
        for method, (color, marker, _) in styles.items()
    ]

    # C: annotated comparison matrix.
    ax = fig.add_subplot(gs[1, :])

    n = len(comparisons)

    ax.set_xlim(0, 2)
    ax.set_ylim(n + 0.35, -0.45)

    for i, method in enumerate(comparisons):

        for j, cy in enumerate(PERIODS):

            r = row(
                pairs,
                cohort=cy,
                reference=method,
            )

            value = r.delta_mse_positive_candidate

            lo = r.psu_low95
            hi = r.psu_high95

            # Highlight only directionally supported
            # differences. Other cells remain neutral.
            color = (
                TEAL if lo > 0
                else RUST if hi < 0
                else GREY
            )

            supported = lo > 0 or hi < 0

            alpha = (
                0.13 + 0.32 * min(abs(value) / 0.25, 1)
                if supported
                else 0.09
            )

            ax.add_patch(Rectangle(
                (j + 0.035, i + 0.035),
                0.93,
                0.91,
                facecolor=mcolors.to_rgba(color, alpha),
                edgecolor="none",
            ))

            ax.text(
                j + 0.5,
                i + 0.35,
                signed(value),
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=color if supported else INK,
            )

            ax.text(
                j + 0.5,
                i + 0.69,
                ci_text(lo, hi, 3),
                ha="center",
                va="center",
                fontsize=6.8,
                color=INK,
            )

    ax.set_yticks(
        np.arange(n) + 0.5,
        [LABELS[m] for m in comparisons],
    )

    ax.set_xticks(
        [0.5, 1.5],
        ["2007–2008", "2021–2023"],
    )

    ax.xaxis.tick_top()

    ax.tick_params(
        length=0,
        pad=5,
    )

    ax.axhline(
        n - 1,
        color=INK,
        linewidth=0.7,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    panel(
        ax,
        "c",
        "Paired MSE differences with 95% intervals",
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.27,
        right=0.98,
        top=0.86,
        bottom=0.08,
    )

    fig.text(
        0.62, 0.025,
        "Positive differences favour HG shared. "
        "Raw markers use 7D.",
        ha="center",
        fontsize=6.8,
        color=INK,
    )

    save(fig, "06_biomedical_utility")


# ============================================================
# Waterfall utility
# ============================================================

def waterfall(
    ax,
    start,
    changes,
    end,
    labels,
    colors,
    intervals=None,
):

    changes = np.asarray(changes, float)

    np.testing.assert_allclose(
        start + changes.sum(),
        end,
        atol=1e-7,
    )

    vals = np.r_[
        start,
        start + np.cumsum(changes),
        end,
    ]

    x = np.arange(len(vals))
    width = 0.62

    # Endpoint bars originate at zero.
    for i in [0, len(vals) - 1]:

        ax.bar(
            i,
            abs(vals[i]),
            bottom=min(0, vals[i]),
            width=width,
            color=INK,
            zorder=3,
        )

    # Intermediate bars represent increments,
    # not independent absolute effects.
    for i, change in enumerate(changes, start=1):

        previous = vals[i - 1]
        current = vals[i]

        ax.bar(
            i,
            abs(change),
            bottom=min(previous, current),
            width=width,
            color=colors[i - 1],
            zorder=3,
        )

    for i in range(len(vals) - 1):

        ax.plot(
            [i + width / 2, i + 1 - width / 2],
            [vals[i], vals[i]],
            color=GREY,
            linewidth=0.8,
            linestyle="--",
            zorder=2,
        )

    ax.axhline(
        0,
        color=INK,
        linewidth=0.7,
    )

    # Place annotations outside the bars.
    for i, v in enumerate(vals):

        if i == 0 or i == len(vals) - 1:

            label = signed(v)

            upper = max(v, 0)

        else:

            label = signed(changes[i - 1])

            upper = max(
                vals[i - 1], v
            )

        ax.annotate(
            label,
            (i, upper),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            fontweight="medium",
        )

    tick_labels = labels.copy()

    if intervals is not None:

        tick_labels = [
            f"{name}\n{ci_text(lo, hi)}"
            for name, (lo, hi) in zip(
                labels, intervals
            )
        ]

    ax.set_xticks(
        x,
        tick_labels,
        fontsize=6.6,
    )

    lower = min(0, vals.min())
    upper = max(0, vals.max())

    span = max(upper - lower, 0.05)

    ax.set_ylim(
        lower - 0.15 * span,
        upper + 0.25 * span,
    )

    ax.set_xlim(
        -0.65,
        len(vals) - 0.35,
    )

    ax.tick_params(axis="x", length=0)

    ax.yaxis.set_major_locator(
        MaxNLocator(5)
    )

    clean(ax)

    return vals


# ============================================================
# FIGURE 7
# Anatomy of the performance reversal
# ============================================================

def figure_drift():

    gaps = read(R119 / "119_error_gap.csv")

    changes = read(
        R119 / "119_error_gap_change.csv"
    )

    components = read(
        R119 / "119_coordinate_error_change.csv"
    )

    fig = plt.figure(
        figsize=(180 * MM, 145 * MM)
    )

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1.05],
        hspace=0.66,
        wspace=0.26,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    metrics = [
        "mean_bias_error",
        "slope_error",
        "residual_error",
    ]

    top_colors = [
        RUST, GOLD, TEAL,
    ]

    # A-B: explain the comparative error reversal
    # for the raw-marker and VAE representations.
    for ax, method, letter, title in [
        (ax_a, "RawX7", "a", "Raw markers"),
        (ax_b, "VAE2_mean_loss", "b", "Multi-decoder VAE"),
    ]:

        start = row(
            gaps,
            cohort="0708",
            reference=method,
            metric="mse",
        ).estimate

        end = row(
            gaps,
            cohort="2123",
            reference=method,
            metric="mse",
        ).estimate

        increments = [
            row(
                changes,
                reference=method,
                metric=metric,
            ).estimate
            for metric in metrics
        ]

        waterfall(
            ax,
            start,
            increments,
            end,
            [
                "2007–08",
                "Mean\nbias",
                "Slope\nerror",
                "Residual",
                "2021–23",
            ],
            top_colors,
        )

        total = row(
            changes,
            reference=method,
            metric="mse",
        )

        cal = row(
            changes,
            reference=method,
            metric="calibration_error",
        ).estimate

        fraction = (
            100 * cal / total.estimate
            if abs(total.estimate) > 1e-12
            else np.nan
        )

        panel(ax, letter, title)

        ax.text(
            0.02, 0.96,
            f"Calibration share: {fraction:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color=INK,
        )

    # Common y-scale for both comparative panels.
    lo = min(
        ax_a.get_ylim()[0],
        ax_b.get_ylim()[0],
    )

    hi = max(
        ax_a.get_ylim()[1],
        ax_b.get_ylim()[1],
    )

    ax_a.set_ylim(lo, hi)
    ax_b.set_ylim(lo, hi)

    ax_a.set_ylabel(
        "Comparator − HG shared MSE"
    )

    # C: exact coordinate-level accounting
    # for the change in the raw-marker error gap.
    groups = [
        "intercept",
        "covariates",
        "shared",
        "discordant",
        "H_private",
        "G_private",
    ]

    group_colors = [
        GREY,
        GREY,
        TEAL,
        RUST,
        BLUE,
        PURPLE,
    ]

    increments = [
        row(
            components,
            group=group,
        ).estimate
        for group in groups
    ]

    early = row(
        gaps,
        cohort="0708",
        reference="RawX7",
        metric="mse",
    )

    late = row(
        gaps,
        cohort="2123",
        reference="RawX7",
        metric="mse",
    )

    intervals = [
        (early.psu_low95, early.psu_high95),
        *[
            (
                row(components, group=g).psu_low95,
                row(components, group=g).psu_high95,
            )
            for g in groups
        ],
        (late.psu_low95, late.psu_high95),
    ]

    waterfall(
        ax_c,
        early.estimate,
        increments,
        late.estimate,
        [
            "2007–08",
            "Intercept",
            "Covariates",
            "Shared",
            "Discordant",
            "Haematology\nprivate",
            "Glycaemia\nprivate",
            "2021–23",
        ],
        group_colors,
        intervals=intervals,
    )

    panel(
        ax_c,
        "c",
        "Coordinate-level accounting of the raw-marker reversal",
    )

    ax_c.set_ylabel(
        "Raw-marker − HG shared MSE"
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.93,
        bottom=0.13,
    )

    fig.text(
        0.55, 0.02,
        "Intervals in c refer to individual signed "
        "contributions or endpoints, not cumulative sums.",
        ha="center",
        fontsize=6.6,
        color=INK,
    )

    save(fig, "07_temporal_error_reversal")


# ============================================================
# FIGURE S1
# Temporal physiological shift atlas
# ============================================================

def figure_shift_atlas():

    df = read(
        R119 / "119_distribution_shift.csv"
    )

    groups = [
        (
            "Haematology",
            BLUE,
            [
                ("LBXHGB", "Haemoglobin"),
                ("LBXRBCSI", "RBC count"),
                ("LBXMCVSI", "MCV"),
                ("LBXRDW", "RDW"),
            ],
        ),
        (
            "Glycaemia",
            PURPLE,
            [
                ("LBXGH", "HbA1c"),
                ("LBXGLU", "Fasting glucose"),
                ("LOG_IN", "Log insulin"),
            ],
        ),
        (
            "Shared",
            TEAL,
            [("S1", "S1"), ("S2", "S2")],
        ),
        (
            "Discordant",
            RUST,
            [("D1", "D1"), ("D2", "D2")],
        ),
        (
            "Haematology-private",
            BLUE,
            [("H1", "H1"), ("H2", "H2")],
        ),
        (
            "Glycaemia-private",
            PURPLE,
            [("G1", "G1")],
        ),
        (
            "Anchor",
            INK,
            [("log_ferritin", "Log ferritin")],
        ),
    ]

    features = []

    for group, color, variables in groups:

        for variable, label in variables:

            features.append({
                "variable": variable,
                "label": label,
                "group": group,
                "color": color,
            })

    mean = np.array([
        [
            row(
                df,
                variable=f["variable"],
                cohort=cy,
            ).mean_shift_in_reference_SD
            for cy in PERIODS
        ]
        for f in features
    ])

    ratio = np.array([
        [
            row(
                df,
                variable=f["variable"],
                cohort=cy,
            ).sd_ratio_to_reference
            for cy in PERIODS
        ]
        for f in features
    ])

    if np.any(ratio <= 0):
        raise ValueError("Nonpositive SD ratio")

    log_ratio = np.log2(ratio)

    mean_limit = max(
        0.1,
        np.max(np.abs(mean)),
    )

    ratio_limit = max(
        0.1,
        np.max(np.abs(log_ratio)),
    )

    mean_norm = mcolors.TwoSlopeNorm(
        vmin=-mean_limit,
        vcenter=0,
        vmax=mean_limit,
    )

    ratio_norm = mcolors.TwoSlopeNorm(
        vmin=-ratio_limit,
        vcenter=0,
        vmax=ratio_limit,
    )

    mean_cmap = mcolors.LinearSegmentedColormap.from_list(
        "mean_shift",
        ["#426C95", "#F8F8F6", "#C15C3D"],
    )

    ratio_cmap = mcolors.LinearSegmentedColormap.from_list(
        "dispersion",
        ["#8064A2", "#F8F8F6", "#087F8C"],
    )

    fig, ax = plt.subplots(
        figsize=(180 * MM, 135 * MM)
    )

    n = len(features)

    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(n, -0.2)

    positions = [
        0, 1, 2.15, 3.15
    ]

    def text_color(rgba):

        rgb = np.array(rgba[:3])

        luminance = np.dot(
            rgb,
            [0.2126, 0.7152, 0.0722],
        )

        return (
            "white"
            if luminance < 0.55
            else INK
        )

    for i, f in enumerate(features):

        # Group membership stripe.
        ax.add_patch(Rectangle(
            (-0.22, i + 0.035),
            0.07,
            0.92,
            facecolor=f["color"],
            linewidth=0,
        ))

        for j in range(4):

            if j < 2:

                value = mean[i, j]

                color = mean_cmap(
                    mean_norm(value)
                )

                label = signed(value, 2)

            else:

                value = ratio[i, j - 2]

                color = ratio_cmap(
                    ratio_norm(
                        log_ratio[i, j - 2]
                    )
                )

                label = f"{value:.2f}×"

            ax.add_patch(Rectangle(
                (
                    positions[j] + 0.02,
                    i + 0.035,
                ),
                0.95,
                0.92,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            ))

            ax.text(
                positions[j] + 0.5,
                i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color(color),
            )

        if i > 0:

            if f["group"] != features[i - 1]["group"]:

                ax.axhline(
                    i,
                    color=INK,
                    linewidth=0.5,
                    alpha=0.45,
                )

    ax.set_yticks(
        np.arange(n) + 0.5,
        [f["label"] for f in features],
    )

    ax.set_xticks(
        np.asarray(positions) + 0.5,
        ["2007–08", "2021–23"] * 2,
    )

    ax.xaxis.tick_top()

    ax.tick_params(
        axis="both",
        length=0,
        pad=5,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        1.0, -1.25,
        "Mean displacement",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    ax.text(
        3.15, -1.25,
        "Relative dispersion",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.25,
        right=0.98,
        top=0.85,
        bottom=0.22,
    )

    # Two separate physical scales.
    cax1 = fig.add_axes([
        0.28, 0.13, 0.29, 0.018
    ])

    cax2 = fig.add_axes([
        0.65, 0.13, 0.29, 0.018
    ])

    cb1 = fig.colorbar(
        ScalarMappable(
            norm=mean_norm,
            cmap=mean_cmap,
        ),
        cax=cax1,
        orientation="horizontal",
    )

    cb1.set_ticks([
        -mean_limit,
        0,
        mean_limit,
    ])

    cb1.set_label(
        "Discovery standard deviations",
        fontsize=7,
    )

    cb2 = fig.colorbar(
        ScalarMappable(
            norm=ratio_norm,
            cmap=ratio_cmap,
        ),
        cax=cax2,
        orientation="horizontal",
    )

    cb2.set_ticks([
        -ratio_limit,
        0,
        ratio_limit,
    ])

    cb2.set_ticklabels([
        f"{2 ** (-ratio_limit):.2f}×",
        "1×",
        f"{2 ** ratio_limit:.2f}×",
    ])

    cb2.set_label(
        "Standard deviation ratio",
        fontsize=7,
    )

    handles = [
        Patch(facecolor=BLUE, label="Haematology"),
        Patch(facecolor=PURPLE, label="Glycaemia"),
        Patch(facecolor=TEAL, label="Shared"),
        Patch(facecolor=RUST, label="Discordant"),
        Patch(facecolor=INK, label="Anchor"),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.59, 0.005),
    )

    save(fig, "S01_physiological_shift_atlas")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    figure_biomedical()
    figure_drift()
    figure_shift_atlas()

    print()
    print(f"Saved redesigned figures to:\n{OUT}")
