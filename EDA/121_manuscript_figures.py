
#!/usr/bin/env python3
"""
121 | Unified manuscript figure suite

Figures derived exclusively from saved numerical outputs.

02  Physiological representation and structural selectivity
03  Information preservation and predictive performance
04  Structural stability and phenotype transport
05  Biomedical utility and comparative anchoring
06  Explanation of temporal prediction failure
S01 Physiological distribution shift

No refitting. No simulated replacement data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import (
    LinearSegmentedColormap,
    TwoSlopeNorm,
    PowerNorm,
    Normalize,
)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable


# ============================================================
# GLOBAL DESIGN SYSTEM
# ============================================================

EDA = Path(__file__).resolve().parent
R = EDA / "Results"

R106 = R / "106_survey_ferritin_anchor"
R112 = R / "112_closure"
R116 = R / "116_7d_structural_validation"
R118 = R / "118_biomedical_utility"
R119 = R / "119_temporal_drift_explanation"

OUT = R / "121_manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

INK = "#282E33"
SHARED = "#146F69"
DISCORD = "#AC583C"
HEM = "#47677E"
GLYC = "#8A9D91"

GREY = "#92999E"
LIGHT = "#E7EBEB"
PALE = "#F4F6F5"
GOLD = "#C4A16A"
WHITE = "#FFFFFF"

GROUP_COLORS = {
    "shared": SHARED,
    "discordant": DISCORD,
    "H_private": HEM,
    "G_private": GLYC,
}

GROUPS = [
    "shared",
    "discordant",
    "H_private",
    "G_private",
]

GROUP_LABELS = [
    "Shared",
    "Discordant",
    "H-private",
    "G-private",
]

PERIODS = ("0708", "2123")

METHODS = {
    "Covariates": "Covariates",
    "RawX7": "Raw markers",
    "PCA2": "PCA",
    "H_PCA2": "H-PCA",
    "G_PCA2": "G-PCA",
    "Threshold2": "Thresholds",
    "Hb_HbA1c": "Hb + HbA1c",
    "Hb_RDW": "Hb + RDW",
    "JIVE_Joint2": "JIVE",
    "AJIVE_Joint2": "AJIVE",
    "VAE2_mean_loss": "VAE",
    "HG_Shared2": "HG shared",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
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
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.fontsize": 7,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
})

DENSITY = LinearSegmentedColormap.from_list(
    "density",
    [
        WHITE,
        "#E1EBE8",
        "#A7C4BD",
        "#5C9690",
        SHARED,
        "#163C39",
    ],
)

DIVERGING = LinearSegmentedColormap.from_list(
    "diverging",
    [HEM, "#F5F5F2", DISCORD],
)

RECOVERY = LinearSegmentedColormap.from_list(
    "recovery",
    [WHITE, "#BCD4CE", SHARED],
)


# ============================================================
# SHARED UTILITIES
# ============================================================

def read(path):

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        dtype={
            "cohort": str,
            "reference": str,
            "target": str,
        },
    )

    for col in (
        "cohort",
        "reference",
        "target",
    ):
        if col in df.columns:

            df[col] = df[col].map(
                lambda v:
                    str(v).zfill(4)
                    if str(v).isdigit()
                    else str(v)
            )

    return df


def row(df, **conditions):

    mask = np.ones(len(df), dtype=bool)

    for name, value in conditions.items():
        mask &= df[name].eq(value).to_numpy()

    selected = df.loc[mask]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {conditions}; "
            f"found {len(selected)}"
        )

    return selected.iloc[0]


def figure(height):

    fig = plt.figure(
        figsize=(180 * MM, height * MM),
        layout="constrained",
    )

    fig.get_layout_engine().set(
        w_pad=2.0 * MM,
        h_pad=2.5 * MM,
        wspace=0.04,
        hspace=0.06,
    )

    return fig


def panel(ax, letter, title):

    ax.set_title(
        rf"$\bf{{{letter}}}$  {title}",
        loc="left",
        pad=8,
    )


def save(fig, stem):

    for ext in ("pdf", "svg", "png"):

        fig.savefig(
            OUT / f"{stem}.{ext}",
            facecolor=WHITE,
            dpi=600,
        )

    plt.close(fig)

    print(f"Saved: {stem}")


def clean(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
    )


def zero(ax, axis="x"):

    if axis == "x":
        ax.axvline(
            0,
            color=GREY,
            linestyle=(0, (3, 3)),
            linewidth=0.8,
            zorder=0,
        )
    else:
        ax.axhline(
            0,
            color=GREY,
            linestyle=(0, (3, 3)),
            linewidth=0.8,
            zorder=0,
        )


def signed(x, digits=2):

    if abs(x) < 0.5 * 10 ** (-digits):
        return f"{0:.{digits}f}"

    return (
        f"{x:+.{digits}f}"
        .replace("-", "−")
    )


def legend_axis(ax, handles, ncol=3):

    ax.set_axis_off()

    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=ncol,
        handletextpad=0.5,
        columnspacing=1.3,
    )


def period_handles(
    first="2007–2008",
    second="2021–2023",
):

    return [
        Line2D(
            [], [],
            marker="o",
            markerfacecolor=WHITE,
            markeredgecolor=INK,
            linestyle="none",
            markersize=4.5,
            label=first,
        ),
        Line2D(
            [], [],
            marker="s",
            markerfacecolor=INK,
            markeredgecolor=INK,
            linestyle="none",
            markersize=4.5,
            label=second,
        ),
    ]


def point_interval(
    ax,
    y,
    estimate,
    low,
    high,
    color=INK,
    filled=True,
    marker="o",
    size=23,
):

    if not np.isfinite(
        [estimate, low, high]
    ).all():
        raise ValueError("Invalid interval")

    ax.hlines(
        y,
        low,
        high,
        color=color,
        linewidth=1.1,
        zorder=2,
    )

    ax.scatter(
        estimate,
        y,
        s=size,
        marker=marker,
        facecolor=color if filled else WHITE,
        edgecolor=color,
        linewidth=0.85,
        zorder=3,
    )


def finish_forest(ax, labels):

    ax.set_yticks(
        range(len(labels)),
        labels,
    )

    ax.set_ylim(
        len(labels) - 0.45,
        -0.55,
    )

    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
        pad=4,
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(4)
    )

    clean(ax)


def matrix_plot(
    ax,
    values,
    xlabels,
    ylabels,
    cmap,
    norm,
    formatter,
    x_top=False,
):

    image = ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    for i in range(values.shape[0]):

        for j in range(values.shape[1]):

            v = values[i, j]
            rgb = np.asarray(
                cmap(norm(v))[:3]
            )

            luminance = rgb @ np.array([
                0.2126,
                0.7152,
                0.0722,
            ])

            ax.text(
                j, i,
                formatter(v),
                ha="center",
                va="center",
                color=(
                    WHITE
                    if luminance < 0.5
                    else INK
                ),
                fontsize=7,
            )

    ax.set_xticks(
        range(len(xlabels)),
        xlabels,
    )

    ax.set_yticks(
        range(len(ylabels)),
        ylabels,
    )

    ax.set_xticks(
        np.arange(-0.5, len(xlabels), 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, len(ylabels), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color=WHITE,
        linewidth=0.8,
    )

    ax.tick_params(
        which="both",
        length=0,
    )

    if x_top:
        ax.xaxis.tick_top()

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


# ============================================================
# FIGURE 2
# From measured physiology to selective coordinates
# ============================================================

def figure_representation():

    data = read(
        R / "72_shared_private_model_input.csv"
    )

    cca = read(
        R / "72_shared_private_cca_summary.csv"
    )

    required = [
        "SURVEY_WT",
        *[
            f"{p}{k}"
            for k in (1, 2)
            for p in (
                "CCA_A",
                "CCA_G",
                "SP_SHARED",
                "SP_DISCORD",
            )
        ],
    ]

    d = (
        data.loc[
            data.PERIOD.eq("2005-2008"),
            required,
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    d = d.loc[d.SURVEY_WT.gt(0)]

    w = d.SURVEY_WT.to_numpy(float)

    edges = np.linspace(-3.5, 3.5, 57)

    fields = []

    for k in (1, 2):

        rho = float(
            cca.loc[
                cca.pair.eq(k),
                "discovery_canonical_corr",
            ].iloc[0]
        )

        u = d[f"CCA_A{k}"].to_numpy()
        v = d[f"CCA_G{k}"].to_numpy()

        s = d[f"SP_SHARED{k}"].to_numpy()
        q = d[f"SP_DISCORD{k}"].to_numpy()

        np.testing.assert_allclose(
            (u + v) / np.sqrt(2 * (1 + rho)),
            s,
            atol=1e-7,
        )

        np.testing.assert_allclose(
            (u - v) / np.sqrt(2 * (1 - rho)),
            q,
            atol=1e-7,
        )

        for x, y in [(u, v), (s, q)]:

            hist = np.histogram2d(
                x,
                y,
                bins=(edges, edges),
                weights=w,
            )[0].T

            fields.append(
                100 * hist / w.sum()
            )

    norm = PowerNorm(
        gamma=0.55,
        vmin=0,
        vmax=max(h.max() for h in fields),
    )

    fig = figure(153)

    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1.12, 1],
    )

    left = outer[:, 0].subgridspec(
        3, 2,
        height_ratios=[1, 1, 0.11],
    )

    for i, hist in enumerate(fields):

        ax = fig.add_subplot(
            left[i // 2, i % 2]
        )

        pair = i // 2 + 1

        mesh = ax.pcolormesh(
            edges,
            edges,
            hist,
            cmap=DENSITY,
            norm=norm,
            shading="flat",
            rasterized=True,
        )

        ax.set(
            xlim=(-3.5, 3.5),
            ylim=(-3.5, 3.5),
            xticks=[-3, 0, 3],
            yticks=[-3, 0, 3],
            aspect="equal",
        )

        ax.axhline(
            0, color=GREY, lw=0.5
        )

        ax.axvline(
            0, color=GREY, lw=0.5
        )

        if i % 2 == 0:

            ax.plot(
                [-3.5, 3.5],
                [-3.5, 3.5],
                "--",
                color=SHARED,
                lw=0.8,
            )

            ax.plot(
                [-3.5, 3.5],
                [3.5, -3.5],
                "--",
                color=DISCORD,
                lw=0.8,
            )

            ax.set_xlabel(
                rf"$U_{pair}$ (haematology)"
            )

            ax.set_ylabel(
                rf"$V_{pair}$ (glycaemia)"
            )

            rho = cca.loc[
                cca.pair.eq(pair),
                "discovery_canonical_corr",
            ].iloc[0]

            title = (
                rf"Pair {pair}, "
                rf"$\rho={rho:.2f}$"
            )

        else:

            ax.set_xlabel(
                rf"$S_{pair}$ (shared)"
            )

            ax.set_ylabel(
                rf"$D_{pair}$ (discordant)"
            )

            title = "Reparameterised"

        panel(ax, "abcd"[i], title)

    cax = fig.add_subplot(left[2, :])

    cb = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
    )

    cb.set_label(
        "Survey weight per bin (%)"
    )

    # Exact standardized Jacobian from Script74.
    ax = fig.add_subplot(outer[0, 1])

    jac = pd.read_csv(
        R / "74_dZ_dX_standardized_matrix.csv",
        index_col=0,
    )

    coord_order = [
        "Shared1",
        "Shared2",
        "Discord1",
        "Discord2",
        "A_private3",
        "A_private4",
        "G_private3",
    ]

    marker_order = [
        "Hb",
        "RBC",
        "MCV",
        "RDW",
        "HbA1c",
        "Glucose",
        "logInsulin",
    ]

    values = jac.loc[
        coord_order,
        marker_order,
    ].to_numpy(float).T

    bound = np.max(np.abs(values))

    image = matrix_plot(
        ax,
        values,
        ["S1", "S2", "D1", "D2",
         "H1", "H2", "G1"],
        ["Hb", "RBC", "MCV", "RDW",
         "HbA1c", "Glucose", "log insulin"],
        DIVERGING,
        TwoSlopeNorm(
            vmin=-bound,
            vcenter=0,
            vmax=bound,
        ),
        lambda v: f"{v:.2f}",
    )

    for boundary in (1.5, 3.5, 5.5):
        ax.axvline(
            boundary,
            color=WHITE,
            lw=2,
        )

    ax.axhline(
        3.5,
        color=WHITE,
        lw=2,
    )

    for tick, color in zip(
        ax.get_xticklabels(),
        [
            SHARED, SHARED,
            DISCORD, DISCORD,
            HEM, HEM, GLYC,
        ],
    ):
        tick.set_color(color)

    panel(
        ax,
        "e",
        "Frozen measurement sensitivity",
    )

    cb = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.025,
    )

    cb.set_label(
        "Coefficient per marker SD"
    )

    cb.locator = MaxNLocator(4)
    cb.update_ticks()

    # Selective recovery of known simulated factors.
    ax = fig.add_subplot(outer[1, 1])

    sim = read(
        R116 / "116_summary.csv"
    )

    sim = sim.loc[
        sim.scenario.eq("discordant")
        & sim.method.eq("HG7")
        & sim.actual_dim.eq(7)
    ]

    matrix = np.array([
        [
            row(
                sim,
                scope=group,
                target=target,
            ).mean_R2
            for target in GROUPS
        ]
        for group in GROUPS
    ])

    vmax = max(
        1.0,
        np.max(matrix),
    )

    vmin = min(
        -0.05,
        np.min(matrix),
    )

    recovery_map = (
        LinearSegmentedColormap.from_list(
            "signed_recovery",
            [DISCORD, WHITE, SHARED],
        )
    )

    im = matrix_plot(
        ax,
        matrix,
        ["Shared", "Discord.", "H-private", "G-private"],
        GROUP_LABELS,
        recovery_map,
        TwoSlopeNorm(
            vmin=vmin,
            vcenter=0,
            vmax=vmax,
        ),
        lambda v: f"{v:.2f}",
    )

    ax.tick_params(
        axis="x",
        labelrotation=25,
    )

    ax.set_xlabel(
        "Known simulated factor"
    )

    ax.set_ylabel(
        "Recovered coordinate group"
    )

    for tick, group in zip(
        ax.get_yticklabels(),
        GROUPS,
    ):
        tick.set_color(
            GROUP_COLORS[group]
        )

    for i in range(4):

        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5),
                1, 1,
                fill=False,
                edgecolor=INK,
                linewidth=0.9,
            )
        )

    panel(
        ax,
        "f",
        "Selective structural recovery",
    )

    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.025,
    )

    cb.set_label(
        "Held-out mean $R^2$"
    )

    save(
        fig,
        "02_representation_and_recovery",
    )


# ============================================================
# FIGURE 3
# Exact information retention versus prediction
# ============================================================

def figure_information():

    rec = read(
        R112 / "112_reconstruction.csv"
    )

    pred = read(
        R112 /
        "112_paired_PSU_comparisons.csv"
    )

    periods = [
        "test_2009-2018",
        "validate_2021-2023",
    ]

    fig = figure(150)

    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[
            1.1, 1, 1.05, 0.18
        ],
    )

    # Reconstruction at the five-dimensional budget.
    ax = fig.add_subplot(gs[0, :])

    methods = [
        "FrozenZ5",
        "PCA5",
        "JIVE5",
        "AJIVE5_rankchecked",
    ]

    labels = [
        "HG5",
        "PCA5",
        "JIVE5",
        "AJIVE5",
    ]

    for i, method in enumerate(methods):

        vals = [
            row(
                rec,
                method=method,
                period=period,
            ).reconstruction_R2
            for period in periods
        ]

        color = (
            SHARED
            if method == "FrozenZ5"
            else INK
        )

        ax.plot(
            [i - 0.12, i + 0.12],
            vals,
            color=color,
            linewidth=1,
            alpha=0.65,
        )

        for j, value in enumerate(vals):

            ax.scatter(
                i + [-0.12, 0.12][j],
                value,
                marker=["o", "s"][j],
                facecolor=WHITE if j == 0 else color,
                edgecolor=color,
                s=28,
                zorder=3,
            )

    full = rec.loc[
        rec.dimensions.eq(7)
        & rec.period.isin(periods)
    ]

    np.testing.assert_allclose(
        full.reconstruction_R2,
        1.0,
        atol=1e-7,
    )

    ax.axhline(
        1,
        color=GREY,
        linestyle="--",
        linewidth=0.9,
    )

    ax.annotate(
        "All five full-rank methods: $R^2=1$",
        (2.0, 1),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=7,
    )

    ax.set_ylim(
        rec.reconstruction_R2.min() - 0.025,
        1.035,
    )

    ax.set_xticks(
        range(4),
        labels,
    )

    ax.set_ylabel(
        "Reconstruction $R^2$"
    )

    panel(
        ax,
        "a",
        "Full information is preserved at seven dimensions",
    )

    outcomes = [
        ("PHQ_total", "Total PHQ-9"),
        ("Somatic", "Somatic"),
        ("CogAff", "Cognitive–affective"),
    ]

    configurations = [
        (
            "rank5",
            "PCA5",
            [
                "FrozenZ5",
                "JIVE5",
                "AJIVE5_rankchecked",
            ],
            ["HG5", "JIVE5", "AJIVE5"],
        ),
        (
            "rank7",
            "RawX7",
            [
                "FrozenZ7",
                "JIVE7",
                "AJIVE7_rankchecked",
                "PCA7",
            ],
            ["HG7", "JIVE7", "AJIVE7", "PCA7"],
        ),
    ]

    for i, (
        family,
        reference,
        methods,
        labels,
    ) in enumerate(configurations):

        for j, (
            outcome,
            title,
        ) in enumerate(outcomes):

            ax = fig.add_subplot(
                gs[i + 1, j]
            )

            sub = pred.loc[
                pred.analysis.eq("prediction")
                & pred.family.eq(family)
                & pred.reference.eq(reference)
                & pred.outcome.eq(outcome)
            ]

            zero(ax)

            for m, method in enumerate(methods):

                color = (
                    SHARED
                    if method.startswith("FrozenZ")
                    else INK
                )

                for t, period in enumerate(periods):

                    rr = row(
                        sub,
                        method=method,
                        period=period,
                    )

                    point_interval(
                        ax,
                        m + [-0.15, 0.15][t],
                        rr.paired_gain_positive_better,
                        rr.approx_PSU_low,
                        rr.approx_PSU_high,
                        color=color,
                        filled=t == 1,
                        marker=["o", "s"][t],
                        size=15,
                    )

            finish_forest(
                ax,
                labels if j == 0
                else [""] * len(labels),
            )

            letter = "bcdefg"[
                i * 3 + j
            ]

            panel(
                ax,
                letter,
                title,
            )

            ax.set_xlabel(
                "ΔMSE vs PCA5"
                if i == 0
                else "ΔMSE vs raw7"
            )

    legend_axis(
        fig.add_subplot(gs[3, :]),
        period_handles(
            "2009–2018",
            "2021–2023",
        ),
        ncol=2,
    )

    save(
        fig,
        "03_information_and_prediction",
    )


# ============================================================
# FIGURE 4
# Structural stability does not ensure clinical transfer
# ============================================================

def figure_transport():

    structural = read(
        R106 /
        "106_design_rediscovery.csv"
    )

    phq = read(
        R106 /
        "106_phq_psu_uncertainty.csv"
    )

    fig = figure(84)

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 0.17],
        width_ratios=[1, 1],
    )

    ax = fig.add_subplot(gs[0, 0])

    groups = [
        "Shared2",
        "Discordant2",
        "H_Isolated2",
        "G_Isolated1",
    ]

    for i, group in enumerate(groups):

        color = GROUP_COLORS[GROUPS[i]]

        for j, cy in enumerate(PERIODS):

            r = row(
                structural,
                reference="0506",
                target=cy,
                group=group,
            )

            point_interval(
                ax,
                i + [-0.15, 0.15][j],
                r.observed_similarity,
                r.psu_rescaled_low_2p5,
                r.psu_rescaled_high_97p5,
                color,
                filled=j == 1,
                marker=["o", "s"][j],
            )

    finish_forest(ax, GROUP_LABELS)

    ax.set_xlim(0, 1.025)

    ax.set_xlabel(
        "Reference subspace similarity"
    )

    panel(
        ax,
        "a",
        "Physiological rediscovery",
    )

    ax = fig.add_subplot(gs[0, 1])

    methods = [
        "Covariates",
        "RawX7",
        "H_G_CCA7",
    ]

    colors = [GREY, INK, SHARED]

    zero(ax)

    for i, method in enumerate(methods):

        for j, cy in enumerate(PERIODS):

            r = row(
                phq,
                cohort=cy,
                method=method,
            )

            point_interval(
                ax,
                i + [-0.15, 0.15][j],
                r.point_frozen_phq_r2,
                r.psu_rescaled_low_2p5,
                r.psu_rescaled_high_97p5,
                colors[i],
                filled=j == 1,
                marker=["o", "s"][j],
            )

    finish_forest(
        ax,
        [
            "Covariates",
            "Raw markers",
            "HG7",
        ],
    )

    ax.set_xlabel(
        "Held-out PHQ-9 $R^2$"
    )

    panel(
        ax,
        "b",
        "Frozen phenotype mapping",
    )

    legend_axis(
        fig.add_subplot(gs[1, :]),
        period_handles(),
        ncol=2,
    )

    save(
        fig,
        "04_structure_and_transport",
    )


# ============================================================
# FIGURE 5
# Biomedical anchoring and comparative utility
# ============================================================

def figure_biomedical():

    perf = read(
        R118 /
        "118_ferritin_performance.csv"
    )

    pairs = read(
        R118 /
        "118_paired_comparisons.csv"
    )

    pairs = pairs.loc[
        pairs.candidate.eq("HG_Shared2")
    ]

    anchor = read(
        R106 /
        "106_ferritin_psu_uncertainty.csv"
    )

    styles = {
        "HG_Shared2": (SHARED, "D"),
        "RawX7": (INK, "s"),
        "VAE2_mean_loss": (HEM, "^"),
        "JIVE_Joint2": (GREY, "o"),
        "AJIVE_Joint2": (GREY, "P"),
        "Threshold2": (GREY, "X"),
    }

    fig = figure(169)

    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[
            1,
            0.16,
            1.45,
            0.17,
        ],
    )

    subset = perf.loc[
        perf.method.isin(styles)
        & perf.cohort.isin(PERIODS)
    ]

    xmin = subset.weighted_mse.min() - 0.02
    xmax = subset.weighted_mse.max() + 0.02

    ymin = (
        subset.low_ferritin_weighted_AUC.min()
        - 0.025
    )

    ymax = (
        subset.low_ferritin_weighted_AUC.max()
        + 0.025
    )

    for i, cy in enumerate(PERIODS):

        ax = fig.add_subplot(gs[0, i])

        hg = row(
            perf,
            cohort=cy,
            method="HG_Shared2",
        )

        ax.axvline(
            hg.weighted_mse,
            color=SHARED,
            linestyle=":",
            linewidth=0.8,
            alpha=0.65,
        )

        ax.axhline(
            hg.low_ferritin_weighted_AUC,
            color=SHARED,
            linestyle=":",
            linewidth=0.8,
            alpha=0.65,
        )

        for method, (
            color,
            marker,
        ) in styles.items():

            r = row(
                perf,
                cohort=cy,
                method=method,
            )

            ax.scatter(
                r.weighted_mse,
                r.low_ferritin_weighted_AUC,
                marker=marker,
                s=55 if method == "HG_Shared2" else 30,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.6,
                zorder=5,
            )

        ax.annotate(
            "HG shared",
            (
                hg.weighted_mse,
                hg.low_ferritin_weighted_AUC,
            ),
            xytext=(5, -11),
            textcoords="offset points",
            color=SHARED,
            fontsize=7,
            fontweight="bold",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(
            "Log-ferritin MSE"
        )

        if i == 0:
            ax.set_ylabel(
                "Low-ferritin AUC"
            )

        ax.xaxis.set_major_locator(
            MaxNLocator(4)
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(4)
        )

        ax.grid(
            color=LIGHT,
            linewidth=0.5,
        )

        panel(
            ax,
            "ab"[i],
            ["2007–2008", "2021–2023"][i],
        )

    handles = [
        Line2D(
            [], [],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor=WHITE,
            markersize=6,
            label=METHODS[method],
        )
        for method, (
            color,
            marker,
        ) in styles.items()
    ]

    legend_axis(
        fig.add_subplot(gs[1, :]),
        handles,
        ncol=6,
    )

    # Script106: original survey-weighted anchoring.
    ax = fig.add_subplot(gs[2, 0])

    methods = [
        "Shared2",
        "Discordant2",
        "H_Isolated2",
        "G_Isolated1",
        "H4",
        "G3",
        "PCA2",
        "RawX7",
    ]

    labels = [
        "Shared",
        "Discordant",
        "H-private",
        "G-private",
        "H markers",
        "G markers",
        "PCA",
        "Raw markers",
    ]

    colors = [
        SHARED,
        DISCORD,
        HEM,
        GLYC,
        HEM,
        GLYC,
        GREY,
        INK,
    ]

    zero(ax)

    for i, method in enumerate(methods):

        for j, cy in enumerate(PERIODS):

            r = row(
                anchor,
                cohort=cy,
                method=method,
            )

            point_interval(
                ax,
                i + [-0.14, 0.14][j],
                r.delta_mse_vs_covariates,
                r.psu_rescaled_low_2p5,
                r.psu_rescaled_high_97p5,
                colors[i],
                filled=j == 1,
                marker=["o", "s"][j],
                size=17,
            )

    ax.axhline(
        3.5,
        color=LIGHT,
        linewidth=0.8,
    )

    finish_forest(ax, labels)

    ax.set_xlabel(
        "ΔMSE vs covariates"
    )

    panel(
        ax,
        "c",
        "Survey-weighted assay anchoring",
    )

    # Script118: direct matched-dimensional contrasts.
    ax = fig.add_subplot(gs[2, 1])

    methods = [
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

    zero(ax)

    for i, method in enumerate(methods):

        for j, cy in enumerate(PERIODS):

            r = row(
                pairs,
                cohort=cy,
                reference=method,
            )

            lo = r.psu_low95
            hi = r.psu_high95

            color = (
                SHARED if lo > 0
                else DISCORD if hi < 0
                else GREY
            )

            point_interval(
                ax,
                i + [-0.14, 0.14][j],
                r.delta_mse_positive_candidate,
                lo,
                hi,
                color,
                filled=j == 1,
                marker=["o", "s"][j],
                size=17,
            )

    ax.axhline(
        8.5,
        color=LIGHT,
        linewidth=0.8,
    )

    finish_forest(
        ax,
        [
            METHODS[m]
            if m != "RawX7"
            else "Raw markers (7D)"
            for m in methods
        ],
    )

    ax.set_xlabel(
        "Comparator MSE − HG shared MSE"
    )

    panel(
        ax,
        "d",
        "Direct comparative utility",
    )

    legend_axis(
        fig.add_subplot(gs[3, :]),
        [
            *period_handles(),
            Patch(
                facecolor=SHARED,
                label="Favours HG",
            ),
            Patch(
                facecolor=DISCORD,
                label="Favours comparator",
            ),
        ],
        ncol=4,
    )

    save(
        fig,
        "05_biomedical_utility",
    )


# ============================================================
# FIGURE 6
# Explaining the temporal performance reversal
# ============================================================

def waterfall(
    ax,
    start,
    changes,
    end,
):

    changes = np.asarray(changes)

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

    width = 0.64

    ax.bar(
        0,
        abs(start),
        bottom=min(0, start),
        width=width,
        color=INK,
    )

    colors = [
        DISCORD,
        GOLD,
        SHARED,
    ]

    for i, change in enumerate(
        changes,
        start=1,
    ):

        previous = vals[i - 1]
        current = vals[i]

        ax.bar(
            i,
            abs(change),
            bottom=min(
                previous,
                current,
            ),
            width=width,
            color=colors[i - 1],
        )

    ax.bar(
        len(vals) - 1,
        abs(end),
        bottom=min(0, end),
        width=width,
        color=INK,
    )

    for i in range(len(vals) - 1):

        ax.plot(
            [
                i + width / 2,
                i + 1 - width / 2,
            ],
            [vals[i], vals[i]],
            linestyle=":",
            color=GREY,
            linewidth=0.9,
        )

    annotations = np.r_[
        start,
        changes,
        end,
    ]

    for i, value in enumerate(
        annotations
    ):

        if i in (0, len(vals) - 1):
            top = max(0, vals[i])
        else:
            top = max(
                vals[i - 1],
                vals[i],
            )

        ax.annotate(
            signed(value, 3),
            (i, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
        )

    ax.axhline(
        0,
        color=INK,
        linewidth=0.6,
    )

    ax.set_xticks(
        range(5),
        [
            "2007–08",
            "Mean\nbias",
            "Slope\nerror",
            "Residual",
            "2021–23",
        ],
    )

    ax.tick_params(
        axis="x",
        length=0,
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(5)
    )

    return vals


def figure_drift():

    gaps = read(
        R119 / "119_error_gap.csv"
    )

    changes = read(
        R119 / "119_error_gap_change.csv"
    )

    attribution = read(
        R119 /
        "119_coordinate_error_change.csv"
    )

    fig = figure(145)

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[
            1.05,
            1.15,
            0.16,
        ],
    )

    metrics = [
        "mean_bias_error",
        "slope_error",
        "residual_error",
    ]

    all_vals = []
    axes = []

    for i, (
        method,
        title,
    ) in enumerate([
        ("RawX7", "Raw markers"),
        ("VAE2_mean_loss", "Multi-decoder VAE"),
    ]):

        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

        early = row(
            gaps,
            cohort="0708",
            reference=method,
            metric="mse",
        ).estimate

        late = row(
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

        vals = waterfall(
            ax,
            early,
            increments,
            late,
        )

        all_vals.extend(vals)

        total = row(
            changes,
            reference=method,
            metric="mse",
        ).estimate

        calibration = row(
            changes,
            reference=method,
            metric="calibration_error",
        ).estimate

        fraction = (
            100 * calibration / total
        )

        panel(
            ax,
            "ab"[i],
            title,
        )

        ax.text(
            0.04,
            0.94,
            f"{fraction:.1f}% calibration",
            transform=ax.transAxes,
            va="top",
            color=DISCORD,
            fontsize=7.5,
            fontweight="bold",
        )

    lo = min(all_vals)
    hi = max(all_vals)
    span = hi - lo

    for ax in axes:

        ax.set_ylim(
            lo - 0.12 * span,
            hi + 0.29 * span,
        )

        ax.set_ylabel(
            "Comparator − HG MSE"
        )

    # Cross-method calibration deterioration.
    ax = fig.add_subplot(gs[1, 0])

    methods = [
        "RawX7",
        "H_PCA2",
        "PCA2",
        "VAE2_mean_loss",
        "JIVE_Joint2",
        "AJIVE_Joint2",
        "Threshold2",
    ]

    zero(ax)

    for i, method in enumerate(methods):

        for j, (
            metric,
            color,
        ) in enumerate([
            ("calibration_error", DISCORD),
            ("residual_error", SHARED),
        ]):

            r = row(
                changes,
                reference=method,
                metric=metric,
            )

            point_interval(
                ax,
                i + [-0.14, 0.14][j],
                r.estimate,
                r.psu_low95,
                r.psu_high95,
                color=color,
                filled=True,
                marker=["o", "s"][j],
                size=17,
            )

    finish_forest(
        ax,
        [METHODS[m] for m in methods],
    )

    ax.set_xlabel(
        "Change in relative error"
    )

    panel(
        ax,
        "c",
        "Calibration versus residual change",
    )

    # Exact algebraic attribution of the raw-model gap.
    ax = fig.add_subplot(gs[1, 1])

    groups = [
        "intercept",
        "covariates",
        "shared",
        "discordant",
        "H_private",
        "G_private",
        "total",
    ]

    labels = [
        "Intercept",
        "Covariates",
        "Shared",
        "Discordant",
        "H-private",
        "G-private",
        "Total",
    ]

    colors = [
        GREY,
        GREY,
        SHARED,
        DISCORD,
        HEM,
        GLYC,
        INK,
    ]

    zero(ax)

    point_sum = 0.0

    for i, group in enumerate(groups):

        r = row(
            attribution,
            group=group,
        )

        point_interval(
            ax,
            i,
            r.estimate,
            r.psu_low95,
            r.psu_high95,
            colors[i],
            filled=True,
            marker="D" if group == "total" else "o",
            size=27 if group == "total" else 20,
        )

        if group != "total":
            point_sum += r.estimate

        else:
            np.testing.assert_allclose(
                point_sum,
                r.estimate,
                atol=1e-7,
            )

    ax.axhline(
        5.5,
        color=LIGHT,
        linewidth=0.8,
    )

    finish_forest(ax, labels)

    ax.set_xlabel(
        "Contribution to error-gap change"
    )

    panel(
        ax,
        "d",
        "Coordinate-level attribution",
    )

    legend_axis(
        fig.add_subplot(gs[2, :]),
        [
            Patch(
                facecolor=DISCORD,
                label="Mean bias / calibration",
            ),
            Patch(
                facecolor=GOLD,
                label="Slope calibration",
            ),
            Patch(
                facecolor=SHARED,
                label="Residual error",
            ),
        ],
        ncol=3,
    )

    save(
        fig,
        "06_temporal_error_attribution",
    )


# ============================================================
# SUPPLEMENTARY FIGURE 1
# Physiological shift across survey periods
# ============================================================

def figure_shift():

    df = read(
        R119 /
        "119_distribution_shift.csv"
    )

    variables = [
        "LBXHGB",
        "LBXRBCSI",
        "LBXMCVSI",
        "LBXRDW",
        "LBXGH",
        "LBXGLU",
        "LOG_IN",
        "S1",
        "S2",
        "D1",
        "D2",
        "H1",
        "H2",
        "G1",
        "log_ferritin",
    ]

    labels = [
        "Haemoglobin",
        "RBC count",
        "MCV",
        "RDW",
        "HbA1c",
        "Fasting glucose",
        "Log insulin",
        "S1",
        "S2",
        "D1",
        "D2",
        "H1",
        "H2",
        "G1",
        "Log ferritin",
    ]

    colors = [
        HEM, HEM, HEM, HEM,
        GLYC, GLYC, GLYC,
        SHARED, SHARED,
        DISCORD, DISCORD,
        HEM, HEM,
        GLYC,
        INK,
    ]

    mean = np.array([
        [
            row(
                df,
                cohort=cy,
                variable=variable,
            ).mean_shift_in_reference_SD
            for cy in PERIODS
        ]
        for variable in variables
    ])

    ratio = np.array([
        [
            row(
                df,
                cohort=cy,
                variable=variable,
            ).sd_ratio_to_reference
            for cy in PERIODS
        ]
        for variable in variables
    ])

    if np.any(ratio <= 0):
        raise ValueError(
            "Nonpositive dispersion ratio"
        )

    log_ratio = np.log2(ratio)

    bound_mean = max(
        abs(mean).max(),
        0.1,
    )

    bound_ratio = max(
        abs(log_ratio).max(),
        0.1,
    )

    fig = figure(128)

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[
            1, 0.055
        ],
    )

    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])

    im1 = matrix_plot(
        a,
        mean,
        ["2007–08", "2021–23"],
        labels,
        DIVERGING,
        TwoSlopeNorm(
            vmin=-bound_mean,
            vcenter=0,
            vmax=bound_mean,
        ),
        lambda v: signed(v, 2),
        x_top=True,
    )

    ratio_map = (
        LinearSegmentedColormap.from_list(
            "relative_dispersion",
            [HEM, WHITE, SHARED],
        )
    )

    im2 = matrix_plot(
        b,
        log_ratio,
        ["2007–08", "2021–23"],
        [""] * len(labels),
        ratio_map,
        TwoSlopeNorm(
            vmin=-bound_ratio,
            vcenter=0,
            vmax=bound_ratio,
        ),
        lambda v: f"{2 ** v:.2f}×",
        x_top=True,
    )

    for tick, color in zip(
        a.get_yticklabels(),
        colors,
    ):

        tick.set_color(color)

    for ax in (a, b):

        for boundary in (
            3.5,
            6.5,
            8.5,
            10.5,
            12.5,
            13.5,
        ):

            ax.axhline(
                boundary,
                color=WHITE,
                linewidth=2,
            )

    panel(
        a,
        "a",
        "Mean displacement",
    )

    panel(
        b,
        "b",
        "Relative dispersion",
    )

    c1 = fig.colorbar(
        im1,
        cax=fig.add_subplot(gs[1, 0]),
        orientation="horizontal",
    )

    c1.set_ticks([
        -bound_mean,
        0,
        bound_mean,
    ])

    c1.set_ticklabels([
        signed(-bound_mean, 2),
        "0",
        signed(bound_mean, 2),
    ])

    c1.set_label(
        "Discovery standard deviations"
    )

    c2 = fig.colorbar(
        im2,
        cax=fig.add_subplot(gs[1, 1]),
        orientation="horizontal",
    )

    c2.set_ticks([
        -bound_ratio,
        0,
        bound_ratio,
    ])

    c2.set_ticklabels([
        f"{2 ** (-bound_ratio):.2f}×",
        "1×",
        f"{2 ** bound_ratio:.2f}×",
    ])

    c2.set_label(
        "Standard deviation ratio"
    )

    save(
        fig,
        "S01_physiological_shift",
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    figure_representation()
    figure_information()
    figure_transport()
    figure_biomedical()
    figure_drift()
    figure_shift()

    print()
    print("=" * 64)
    print("MANUSCRIPT FIGURE SUITE COMPLETE")
    print(f"Output: {OUT}")
    print("=" * 64)
