# -*- coding: utf-8 -*-

"""
90_literature_landscape_v2.py

Purpose
-------
Create a concise, presentation-ready literature landscape from:
    PhD_Literature_Library.xlsx

Figures
-------
1. 90A_evidence_streams
   Where is the reviewed literature concentrated?

2. 90B_novelty_timeline
   How quickly is the field moving, and where is novelty pressure coming from?

3. 90C_global_clinical_footprint
   Where is the geographically anchored clinical evidence?

4. 90D_competitor_landscape
   Which research ecosystems are closest to our target space?

The visual language intentionally matches the presentation:
    sns.color_palette("Blues", as_cmap=True)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\Hematological and Glycemic Decomposition of Depression"
)

INPUT_XLSX = (
    PROJECT_ROOT / "PhD_Literature_Library.xlsx"
)

OUTDIR = (
    PROJECT_ROOT / "EDA" / "Results" / "90_literature_landscape"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PRESENTATION STYLE
# =============================================================================

sns.set_theme(
    style="whitegrid",
    context="talk"
)

CMAP = sns.color_palette("Blues", as_cmap=True)

# Discrete tones from the same presentation family.
BLUES = sns.color_palette("Blues", 9)

VERY_LIGHT = BLUES[1]
LIGHT = BLUES[2]
MID_LIGHT = BLUES[3]
MID = BLUES[5]
DARK = BLUES[7]
VERY_DARK = BLUES[8]

TEXT = "#1f2937"
MUTED = "#64748b"
GRID = "#dbe3ec"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.edgecolor": TEXT,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 140,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def save_both(fig, stem):
    fig.savefig(
        OUTDIR / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        OUTDIR / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white"
    )
    print(f"Saved {stem}")


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_excel(
    INPUT_XLSX,
    sheet_name="00_Master_Triage",
    header=2
)

df = df.dropna(how="all").copy()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()

df = df.replace({
    "nan": np.nan,
    "NaN": np.nan,
    "None": np.nan,
    "": np.nan
})

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Novelty Threat"] = (
    df["Novelty Threat"]
    .fillna("Low")
)

df["STAR?"] = (
    df["STAR?"]
    .fillna("No")
)


# =============================================================================
# FIGURE A
# WHERE IS THE LITERATURE ACTUALLY CONCENTRATED?
# =============================================================================

sheet_labels = {
    "06_Math_Frameworks":
        "Mathematical / computational\nframeworks",

    "05_General_Relevant":
        "Supporting physiology /\npopulation context",

    "01_A_G_Depression":
        "A–G–depression\nintersection",

    "02_A_Cognition":
        "Haematology ×\ncognition",

    "03_G_Cognition":
        "Glycaemia ×\ncognition",

    "04_A_G_Interaction":
        "A–G physiological\ncoupling",
}

stream = (
    df["Primary Sheet"]
    .value_counts()
    .rename_axis("sheet")
    .reset_index(name="papers")
)

stream["label"] = (
    stream["sheet"]
    .map(sheet_labels)
    .fillna(stream["sheet"])
)

stream = stream.sort_values(
    "papers",
    ascending=True
)

stream["pct"] = (
    stream["papers"] / len(df) * 100
)

# Shade by count, all from Blues.
norm = mpl.colors.Normalize(
    vmin=stream["papers"].min(),
    vmax=stream["papers"].max()
)

colors = [
    CMAP(norm(v))
    for v in stream["papers"]
]

fig, ax = plt.subplots(
    figsize=(11, 6.7)
)

y = np.arange(len(stream))

# faint stems
for yi, val in zip(y, stream["papers"]):
    ax.hlines(
        yi,
        0,
        val,
        color=GRID,
        linewidth=4,
        zorder=1
    )

# large terminal dots
ax.scatter(
    stream["papers"],
    y,
    s=330,
    c=colors,
    edgecolor="white",
    linewidth=1.8,
    zorder=3
)

for yi, val, pct in zip(
    y,
    stream["papers"],
    stream["pct"]
):
    ax.text(
        val + 0.6,
        yi,
        f"{val}  ({pct:.0f}%)",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT
    )

ax.set_yticks(y)
ax.set_yticklabels(
    stream["label"]
)

ax.set_xlim(
    0,
    stream["papers"].max() + 7
)

ax.set_xlabel("Papers in curated library")

ax.set_title(
    "Where the reviewed literature is concentrated",
    loc="left",
    pad=18
)

ax.text(
    0,
    1.01,
    "62-paper literature audit · primary classification",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=11
)

ax.grid(
    axis="x",
    alpha=0.18
)
ax.grid(
    axis="y",
    visible=False
)

sns.despine(
    ax=ax,
    left=True
)

save_both(
    fig,
    "90A_evidence_streams"
)

plt.close(fig)


# =============================================================================
# FIGURE B
# FIELD ACCELERATION + NOVELTY PRESSURE
# =============================================================================

recent = df[
    df["Year"].notna()
].copy()

recent["Year bucket"] = recent["Year"].astype(int)

# Older literature is background/foundation.
recent.loc[
    recent["Year bucket"] <= 2018,
    "Year bucket"
] = 2018

timeline = pd.crosstab(
    recent["Year bucket"],
    recent["Novelty Threat"]
)

for c in ["Low", "Medium", "High"]:
    if c not in timeline.columns:
        timeline[c] = 0

timeline = timeline[
    ["Low", "Medium", "High"]
]

timeline.index = [
    "≤2018" if y == 2018 else str(y)
    for y in timeline.index
]

fig, ax = plt.subplots(
    figsize=(11.5, 6.5)
)

bottom = np.zeros(len(timeline))

threat_colors = {
    "Low": LIGHT,
    "Medium": MID,
    "High": VERY_DARK
}

for threat in [
    "Low",
    "Medium",
    "High"
]:
    vals = timeline[threat].values

    ax.bar(
        timeline.index,
        vals,
        bottom=bottom,
        color=threat_colors[threat],
        label=threat,
        width=0.72,
        edgecolor="white",
        linewidth=0.7
    )

    bottom += vals

# totals
for x, total in enumerate(
    timeline.sum(axis=1)
):
    if total > 0:
        ax.text(
            x,
            total + 0.4,
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=TEXT
        )

# Core message.
high_2026 = int(
    (
        (df["Year"] == 2026)
        &
        (df["Novelty Threat"] == "High")
    ).sum()
)

high_total = int(
    (df["Novelty Threat"] == "High").sum()
)

recent_25_26 = int(
    df["Year"]
    .isin([2025, 2026])
    .sum()
)

ax.annotate(
    f"{high_2026}/{high_total} high-threat papers\nare from 2026",
    xy=(
        len(timeline) - 1,
        timeline.iloc[-1].sum()
    ),
    xytext=(
        len(timeline) - 3.2,
        timeline.iloc[-1].sum() * 0.67
    ),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=1.4,
        color=VERY_DARK
    ),
    fontsize=11,
    fontweight="bold",
    color=VERY_DARK,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor=GRID
    )
)

ax.set_title(
    "The methodological landscape is moving quickly",
    loc="left",
    pad=18
)

ax.text(
    0,
    1.01,
    f"{recent_25_26}/62 reviewed papers were published in 2025–26",
    transform=ax.transAxes,
    fontsize=11,
    color=MUTED
)

ax.set_ylabel("Number of papers")
ax.set_xlabel("Publication year")

ax.legend(
    title="Novelty threat",
    frameon=False,
    ncol=3,
    loc="upper left"
)

ax.grid(
    axis="y",
    alpha=0.2
)
ax.grid(
    axis="x",
    visible=False
)

sns.despine(ax=ax)

save_both(
    fig,
    "90B_novelty_timeline"
)

plt.close(fig)


# =============================================================================
# FIGURE C
# GEOGRAPHIC CLINICAL FOOTPRINT
# =============================================================================

# Deliberately exclude:
# - global systematic reviews
# - purely methodological papers
# - papers with country metadata not stored
#
# This avoids pretending these categories represent countries.

country_counts = {
    "India": 0,
    "China": 0,
    "United Kingdom": 0,
    "United States": 0,
    "Brazil": 0,
}

for loc in df["Country / Region"].dropna():

    if loc == "India":
        country_counts["India"] += 1

    elif loc == "China":
        country_counts["China"] += 1

    elif loc == "United Kingdom":
        country_counts["United Kingdom"] += 1

    elif loc == "Brazil":
        country_counts["Brazil"] += 1

    elif loc == "India + United States":
        country_counts["India"] += 1
        country_counts["United States"] += 1

    elif loc == "United States (multi-hospital external cohort)":
        country_counts["United States"] += 1


geo = pd.DataFrame({
    "country": [
        "India",
        "China",
        "United Kingdom",
        "United States",
        "Brazil",
    ],
    "papers": [
        country_counts["India"],
        country_counts["China"],
        country_counts["United Kingdom"],
        country_counts["United States"],
        country_counts["Brazil"],
    ],
    "lat": [
        22.8,
        35.8,
        55.0,
        39.5,
        -10.0,
    ],
    "lon": [
        79.0,
        104.0,
        -3.0,
        -98.0,
        -52.0,
    ],
})

# plotly custom Blues scale
blues_scale = [
    [0.00, sns.color_palette("Blues", 9).as_hex()[1]],
    [0.35, sns.color_palette("Blues", 9).as_hex()[3]],
    [0.65, sns.color_palette("Blues", 9).as_hex()[5]],
    [1.00, sns.color_palette("Blues", 9).as_hex()[8]],
]

fig_map = go.Figure()

fig_map.add_trace(
    go.Scattergeo(
        lon=geo["lon"],
        lat=geo["lat"],

        text=[
            f"{c}<br>{n} papers"
            for c, n
            in zip(
                geo["country"],
                geo["papers"]
            )
        ],

        mode="markers+text",

        textposition=[
            "bottom center",
            "bottom center",
            "top center",
            "top center",
            "bottom center"
        ],

        marker=dict(
            size=14 + geo["papers"] * 4.2,
            color=geo["papers"],
            colorscale=blues_scale,
            cmin=1,
            cmax=geo["papers"].max(),
            line=dict(
                width=1.3,
                color="white"
            ),
            colorbar=dict(
                title="Papers",
                thickness=13,
                len=0.55
            )
        ),

        hovertemplate=(
            "<b>%{text}</b>"
            "<extra></extra>"
        )
    )
)

fig_map.update_geos(
    projection_type="natural earth",
    showland=True,
    landcolor="#f3f6f9",
    showcountries=True,
    countrycolor="#9aa7b2",
    countrywidth=0.7,
    showcoastlines=True,
    coastlinecolor="#64748b",
    coastlinewidth=0.6,
    showocean=True,
    oceancolor="white",
    showframe=False,
)

fig_map.update_layout(
    title=dict(
        text=(
            "<b>Where the geographically anchored "
            "clinical evidence comes from</b>"
            "<br>"
            "<sup>"
            "Country-specific papers only; global reviews and "
            "methodological papers excluded"
            "</sup>"
        ),
        x=0.03,
        xanchor="left",
        font=dict(
            size=24,
            color=TEXT
        )
    ),

    margin=dict(
        l=10,
        r=20,
        t=90,
        b=10
    ),

    paper_bgcolor="white",
    font=dict(
        family="Arial",
        color=TEXT
    ),
)

fig_map.write_html(
    OUTDIR / "90C_global_clinical_footprint.html"
)

try:
    fig_map.write_image(
        str(
            OUTDIR /
            "90C_global_clinical_footprint.png"
        ),
        width=1600,
        height=850,
        scale=2
    )

    fig_map.write_image(
        str(
            OUTDIR /
            "90C_global_clinical_footprint.svg"
        )
    )

except Exception as exc:
    print(
        "Static map export unavailable. "
        "HTML was still created."
    )
    print(exc)


# =============================================================================
# FIGURE D — COMPETITOR LANDSCAPE (ID-BASED, CLEANER)
# =============================================================================

teams = pd.DataFrame({
    "id": [
        "JN", "WW", "UB", "HN", "PM", "LG", "LD", "TM"
    ],
    "team": [
        "Jiang / Nanjing",
        "West China / WHALE",
        "Uhler / Broad–MIT",
        "HCL / NUS",
        "PULSE / MUST",
        "Lei / Fudan–Geneva",
        "LASI-DAD",
        "ToMMo",
    ],
    "clinical_overlap": [
        0.94, 0.77, 0.25, 0.27, 0.40, 0.27, 0.64, 0.57
    ],
    "validation_depth": [
        0.34, 0.70, 0.56, 0.62, 0.93, 0.97, 0.82, 0.72
    ],
})

fig, ax = plt.subplots(figsize=(11.6, 7.0))

# -----------------------------------------------------------------------------
# LIMITS
# -----------------------------------------------------------------------------
ax.set_xlim(0.10, 1.02)
ax.set_ylim(0.22, 1.02)

# -----------------------------------------------------------------------------
# TITLE / SUBTITLE
# -----------------------------------------------------------------------------
ax.set_title(
    "Closest research ecosystems occupy different parts of the problem",
    loc="center",
    pad=24,
    fontsize=18,
    fontweight="bold"
)

ax.text(
    0.5,
    1.005,
    "Qualitative positioning from the reviewed papers and group focus; not a performance ranking",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10.2,
    color=MUTED
)

# -----------------------------------------------------------------------------
# TARGET BOX
# -----------------------------------------------------------------------------
target = Rectangle(
    (0.73, 0.80),
    0.23,
    0.17,
    facecolor=MID_LIGHT,
    alpha=0.14,
    edgecolor=MID_LIGHT,
    linestyle="--",
    linewidth=1.5,
    zorder=1
)
ax.add_patch(target)

ax.text(
    0.845,
    0.94,
    "Target space",
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=DARK
)

ax.text(
    0.845,
    0.905,
    "Contributor-specific physiology\n+ temporal / external validation",
    ha="center",
    va="center",
    fontsize=9.0,
    color=MUTED,
    linespacing=1.2
)

# -----------------------------------------------------------------------------
# POINTS
# -----------------------------------------------------------------------------
ax.scatter(
    teams["clinical_overlap"],
    teams["validation_depth"],
    s=300,
    color=DARK,
    edgecolor="white",
    linewidth=1.8,
    zorder=4
)

# Put the short IDs INSIDE the points
for _, row in teams.iterrows():
    ax.text(
        row["clinical_overlap"],
        row["validation_depth"],
        row["id"],
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="white",
        zorder=5
    )

# -----------------------------------------------------------------------------
# SIDE KEY / LEGEND PANEL
# -----------------------------------------------------------------------------
legend_x = 0.115
legend_y = 0.405

ax.text(
    legend_x,
    legend_y + 0.135,
    "Ecosystem key",
    fontsize=10.4,
    fontweight="bold",
    color=TEXT,
    ha="left",
    va="top"
)

legend_lines = [
    "LG  Lei / Fudan–Geneva",
    "PM  PULSE / MUST",
    "UB  Uhler / Broad–MIT",
    "HN  HCL / NUS",
    "TM  ToMMo",
    "LD  LASI-DAD",
    "WW  West China / WHALE",
    "JN  Jiang / Nanjing",
]

for i, line in enumerate(legend_lines):
    ax.text(
        legend_x,
        legend_y + 0.115 - i * 0.025,
        line,
        fontsize=9.2,
        color=TEXT,
        ha="left",
        va="top"
    )

# -----------------------------------------------------------------------------
# TAKEAWAY TEXT
# -----------------------------------------------------------------------------
ax.text(
    0.115,
    0.30,
    "Upper-right = closer to our goal:\n"
    "more clinical overlap and stronger temporal / external depth.",
    fontsize=9.3,
    color=MUTED,
    ha="left",
    va="top"
)

# -----------------------------------------------------------------------------
# AXES
# -----------------------------------------------------------------------------
ax.set_xlabel(
    "Clinical overlap with the A–G–phenotype problem  →",
    fontsize=12.5
)

ax.set_ylabel(
    "Longitudinal / transport / rediscovery depth  →",
    fontsize=12.5
)

# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
ax.grid(
    color=GRID,
    linewidth=0.8,
    alpha=0.38
)

sns.despine(ax=ax, top=True, right=True)

ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

plt.tight_layout()
save_both(fig, "90D_competitor_landscape_clean")
plt.close(fig)

# =============================================================================
# FIGURE E — PHQ-9 SEVERITY COMPOSITION ACROSS PERIODS
# Source: EDA/Results/86_phq_severity_distribution.csv
# =============================================================================

SEVERITY_FILE = (
    PROJECT_ROOT / "EDA" / "Results" / "86_phq_severity_distribution.csv"
)

sev = pd.read_csv(SEVERITY_FILE)

periods = [
    "2005-2008",
    "2009-2018",
    "2021-2023",
]

severity_order = [
    "0-4 minimal",
    "5-9 mild",
    "10-14 moderate",
    "15-19 moderately severe",
    "20-27 severe",
]

severity_labels = {
    "0-4 minimal": "Minimal",
    "5-9 mild": "Mild",
    "10-14 moderate": "Moderate",
    "15-19 moderately severe": "Moderately severe",
    "20-27 severe": "Severe",
}

# Light → dark as symptom burden increases
severity_colors = [
    CMAP(0.18),
    CMAP(0.35),
    CMAP(0.55),
    CMAP(0.72),
    CMAP(0.90),
]

fig, ax = plt.subplots(figsize=(9.8, 6.2))

bottom = np.zeros(len(periods))

for severity, color in zip(severity_order, severity_colors):

    vals = []

    for period in periods:

        row = sev[
            (sev["period"] == period)
            &
            (sev["category"] == severity)
        ]

        vals.append(
            100 * float(row["weighted_prop"].iloc[0])
        )

    vals = np.array(vals)

    bars = ax.bar(
        periods,
        vals,
        bottom=bottom,
        width=0.62,
        color=color,
        edgecolor="white",
        linewidth=1.0,
        label=severity_labels[severity],
    )

    # Segment labels only where useful
    for i, v in enumerate(vals):

        if v >= 4:
            ax.text(
                i,
                bottom[i] + v / 2,
                f"{v:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="medium",
                color=(
                    "white"
                    if color[0] < 0.45
                    else TEXT
                ),
            )

    bottom += vals


ax.set_ylim(0, 100)

ax.set_ylabel(
    "Survey-weighted participants (%)",
    fontsize=12
)

ax.set_xlabel("")

ax.set_title(
    "Depressive-symptom burden increased in later NHANES periods",
    loc="left",
    fontsize=18,
    fontweight="bold",
    pad=18,
)

ax.text(
    0,
    1.01,
    "PHQ-9 severity composition across the three analysis periods",
    transform=ax.transAxes,
    fontsize=10.5,
    color=MUTED,
)

ax.legend(
    title="PHQ-9 severity",
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9.5,
    title_fontsize=10,
)

ax.grid(False)

sns.despine(
    ax=ax,
    top=True,
    right=True,
    left=True,
)

ax.tick_params(
    axis="y",
    length=0,
)

plt.tight_layout()

save_both(
    fig,
    "90E_phq_severity_composition"
)

plt.close(fig)

# =============================================================================
# FIGURE F — PHQ-9 ≥10 WITHIN A / G PHYSIOLOGICAL STRATA
# Source: EDA/Results/59_phq_within_physiology_strata.csv
# =============================================================================

STRATA_FILE = (
    PROJECT_ROOT / "EDA" / "Results" / "59_phq_within_physiology_strata.csv"
)

strata = pd.read_csv(STRATA_FILE)

periods = [
    "2005-2008",
    "2009-2018",
    "2021-2023",
]

strata_order = [
    "Neither anaemia nor diabetes-range",
    "Anaemia only",
    "Diabetes-range only",
    "Anaemia + diabetes-range",
]

strata_labels = [
    "Neither",
    "A only",
    "G only",
    "A + G",
]

# Period colours: same Blues family
period_colors = [
    CMAP(0.35),
    CMAP(0.58),
    CMAP(0.85),
]

fig, ax = plt.subplots(figsize=(10.6, 6.3))

x = np.arange(len(strata_order))
width = 0.23

for j, (period, color) in enumerate(
    zip(periods, period_colors)
):

    vals = []
    ns = []

    for s in strata_order:

        row = strata[
            (strata["period"] == period)
            &
            (strata["physiological_stratum"] == s)
        ]

        vals.append(
            float(
                row[
                    "PHQ10_weighted_percent"
                ].iloc[0]
            )
        )

        ns.append(
            int(row["n"].iloc[0])
        )

    pos = x + (j - 1) * width

    bars = ax.bar(
        pos,
        vals,
        width=width,
        color=color,
        edgecolor="white",
        linewidth=1.0,
        label=period,
    )

    for xx, yy in zip(pos, vals):

        ax.text(
            xx,
            yy + 0.35,
            f"{yy:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )


ax.set_xticks(x)

ax.set_xticklabels(
    strata_labels,
    fontsize=11,
)

ax.set_ylabel(
    "Weighted prevalence of PHQ-9 ≥10 (%)",
    fontsize=12,
)

ax.set_xlabel(
    "Physiological stratum",
    fontsize=12,
)

ax.set_title(
    "Depressive burden appears across both A- and G-defined physiology",
    loc="left",
    fontsize=18,
    fontweight="bold",
    pad=18,
)

ax.text(
    0,
    1.01,
    "Threshold groups are descriptive; the combined A + G stratum is comparatively small",
    transform=ax.transAxes,
    fontsize=10.5,
    color=MUTED,
)

ax.legend(
    title="NHANES period",
    frameon=False,
    ncol=3,
    loc="upper left",
    fontsize=9.5,
    title_fontsize=10,
)

ax.grid(
    axis="y",
    alpha=0.18,
)

ax.grid(
    axis="x",
    visible=False,
)

sns.despine(
    ax=ax,
    top=True,
    right=True,
)

plt.tight_layout()

save_both(
    fig,
    "90F_phq10_by_physiology_strata"
)

plt.close(fig)



# =============================================================================
# FIGURE G — SAME PHQ-9 SCORE, DIFFERENT PHYSIOLOGICAL STATES
# Script 95 | Real NHANES participants | Seven-biomarker fingerprint
# =============================================================================

import runpy
from sklearn.cluster import KMeans

s95 = runpy.run_path(
    str(PROJECT_ROOT / "EDA" / "95_physiological_configuration_atlas.py")
)

cohorts, _ = s95["load_real_data"]()
discovery = cohorts["train_2005-2006"].copy()

ZCOLS = [
    "SP_SHARED1", "SP_SHARED2", "SP_DISCORD1",
    "SP_DISCORD2", "SP_A_PRIVATE3",
    "SP_A_PRIVATE4", "SP_G_PRIVATE3",
]

RAWCOLS = [
    "LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW",
    "LBXGH", "LBXGLU", "LOG_IN",
]

RAWLABELS = [
    "Hb", "RBC", "MCV", "RDW",
    "HbA1c", "Glucose", "ln insulin",
]

weights = discovery["SURVEY_WT"].to_numpy(float)
z = discovery[ZCOLS].to_numpy(float)

# Reproduce the original Script 95 cluster assignments.
z_mean, z_sd = s95["weighted_mean_sd"](z, weights)
z_scaled = (z - z_mean) / z_sd

km = KMeans(n_clusters=4, n_init=20, random_state=95)
km.fit(z_scaled, sample_weight=weights / weights.mean())

order = np.argsort(km.cluster_centers_[:, 0])
mapping = {int(old): int(new) for new, old in enumerate(order)}
clusters = np.array([mapping[int(c)] for c in km.labels_])

# Exactly the original PHQ-9 = 10 participants.
mask = discovery["PHQ9_ITEMS_TOTAL"].eq(10).to_numpy()
focus = discovery.loc[mask].copy()
focus["group"] = clusters[mask]

if len(focus) != 22:
    raise RuntimeError(
        f"Expected 22 real observations; found {len(focus)}. "
        "Check Script 95 source coverage."
    )

if not np.isfinite(focus[RAWCOLS].to_numpy(float)).all():
    raise ValueError("Incomplete raw biomarkers in selected cohort")

# Standardize original seven biomarkers relative to discovery.
raw = discovery[RAWCOLS].to_numpy(float)
mu = np.average(raw, axis=0, weights=weights)
sd = np.sqrt(
    np.average((raw - mu) ** 2, axis=0, weights=weights)
)

if not np.isfinite(sd).all() or np.any(sd <= 0):
    raise ValueError("Invalid discovery biomarker scaling")

# Keep participants anonymous and sort for visual interpretation.
focus = focus.sort_values(
    ["group", "LBXGH"]
).reset_index(drop=True)

heat = (focus[RAWCOLS].to_numpy(float) - mu) / sd
groups = focus["group"].to_numpy(int)

colors = [
    CMAP(0.28), CMAP(0.47),
    CMAP(0.68), CMAP(0.90),
]

fig, (ax, hx) = plt.subplots(
    1, 2,
    figsize=(15.8, 8.2),
    gridspec_kw={"width_ratios": [1.1, 1.32]},
)

fig.patch.set_facecolor("white")

# ------------------------------------------------------------------
# PANEL A: PHYSIOLOGICAL COORDINATES
# ------------------------------------------------------------------

for group in range(4):
    sub = focus.loc[focus["group"] == group]

    ax.scatter(
        sub["SP_SHARED1"],
        sub["SP_A_PRIVATE3"],
        s=110,
        color=colors[group],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.95,
        label=f"Group {group} (n={len(sub)})",
        zorder=3,
    )

ax.set_title(
    "a   Different positions in the frozen Z space",
    loc="left", pad=15,
)

ax.set_xlabel("Shared physiological component 1")
ax.set_ylabel("Hematology-private component 3")

ax.grid(color=GRID, alpha=0.55, linewidth=0.75)
ax.legend(loc="upper right", frameon=False, fontsize=9.5)
sns.despine(ax=ax, top=True, right=True)

# ------------------------------------------------------------------
# PANEL B: ALL SEVEN REAL BIOMARKERS
# ------------------------------------------------------------------

im = hx.imshow(
    np.clip(heat, -2.5, 2.5),
    cmap="RdBu_r",
    vmin=-2.5,
    vmax=2.5,
    aspect="auto",
    interpolation="nearest",
)

hx.set_title(
    "b   Seven observed biomarker fingerprints",
    loc="left", pad=15,
)

hx.set_xticks(
    np.arange(len(RAWLABELS)),
    labels=RAWLABELS,
    rotation=35,
    ha="right",
)

hx.set_yticks(
    np.arange(len(focus)),
    labels=[f"{i:02d}" for i in range(1, len(focus) + 1)],
)

hx.set_ylabel("Anonymous participant (same PHQ-9 = 10)")
hx.tick_params(axis="both", length=0)

# Separate hematology from glycemia.
hx.axvline(3.5, color="white", linewidth=2.8)

# Visual boundaries between exploratory groups.
for i in range(1, len(focus)):
    if groups[i] != groups[i - 1]:
        hx.axhline(
            i - 0.5,
            color="white",
            linewidth=2.8,
        )

# Coloured markers link the rows with panel A.
hx.scatter(
    np.full(len(focus), -0.76),
    np.arange(len(focus)),
    c=[colors[g] for g in groups],
    s=28,
    marker="s",
    edgecolors="none",
    clip_on=False,
)

cb = fig.colorbar(im, ax=hx, fraction=0.035, pad=0.03)
cb.set_label("Difference from discovery mean (SD)", fontsize=10)
cb.ax.tick_params(labelsize=9)

# ------------------------------------------------------------------
# TITLE, ACTUAL OBSERVED RANGES, AND FOOTNOTES
# ------------------------------------------------------------------

hba1c = focus["LBXGH"].to_numpy(float)
glucose = focus["LBXGLU"].to_numpy(float)

fig.suptitle(
    "One depressive-symptom score, multiple physiological states",
    x=0.06, y=0.975,
    ha="left",
    fontsize=19,
    fontweight="bold",
    color=TEXT,
)

fig.text(
    0.06, 0.925,
    f"NHANES 2005–06  |  PHQ-9 = 10  |  n = {len(focus)}  |  "
    f"HbA1c {hba1c.min():.1f}–{hba1c.max():.1f}%  |  "
    f"Glucose {glucose.min():.0f}–{glucose.max():.0f} mg/dL",
    fontsize=11.1,
    color=MUTED,
)

fig.text(
    0.06, 0.065,
    "Each point and heatmap row represents one real participant. "
    "Biomarkers are standardized against the discovery cohort; "
    "colours saturate at ±2.5 SD.",
    fontsize=9.5,
    color=MUTED,
)

fig.text(
    0.06, 0.040,
    "Exploratory K-means groups (K=4; silhouette=0.109) are "
    "visual guides, not validated clinical subtypes.",
    fontsize=9.5,
    color=MUTED,
)

fig.subplots_adjust(
    left=0.075, right=0.95,
    bottom=0.17, top=0.855,
    wspace=0.28,
)

save_both(
    fig,
    "95_PHQ10_heterogeneity_physiological_fingerprints"
)

plt.close(fig)


# =============================================================================
# FIGURE H — OBSERVED VS PREDICTED PHENOTYPE SCORES
# Script 98 | Frozen fitted equations | NHANES temporal transfer
# =============================================================================

import importlib.util
import json

results98 = PROJECT_ROOT / "EDA" / "Results"

spec98 = importlib.util.spec_from_file_location(
    "script98_plot_source",
    PROJECT_ROOT / "EDA" / "98_regularized_phenotype_equation.py"
)
s98 = importlib.util.module_from_spec(spec98)
spec98.loader.exec_module(s98)

# Initialise the source script's required state.
s98.yield_msg = []
s98.normalizer = {}
s98.categories_reference = {}
s98.categories_feature_names = []

# Reproduce Script 98's exact analytical cohort and feature definitions.
data98 = s98.load_real()
features98 = s98.build_features(data98)

terms98 = pd.read_csv(results98 / "98_equation_terms.csv")
metrics98 = pd.read_csv(
    results98 / "98_frozen_holdout_performance.csv"
)

with open(results98 / "98_manifest.json", encoding="utf-8") as f:
    manifest98 = json.load(f)

winners98 = {
    r["outcome"]: r
    for r in manifest98["outcome_winners"]
}

periods98 = [
    ("train_2005-2008", "Discovery | 2005–08"),
    ("test_2009-2018", "Frozen test | 2009–18"),
    ("validate_2021-2023", "Frozen validation | 2021–23"),
]

outcomes98 = {
    "PHQ_total": ("PHQ-9 total", 27),
    "Somatic": ("Somatic symptoms", 9),
    "CogAff": ("Cognitive-affective symptoms", 18),
}

for outcome, (label, upper) in outcomes98.items():

    selected = winners98[outcome]
    scope = selected["selected_scope"]
    design = selected["selected_design"]

    # Load the REAL stored coefficients of the selected equation.
    t = terms98.loc[
        terms98["outcome"].eq(outcome)
        & terms98["scope"].eq(scope)
        & terms98["design"].eq(design)
        & terms98["method"].eq(selected["method"])
    ].copy()

    if t.empty or t["feature"].duplicated().any():
        raise RuntimeError(f"Missing/duplicate equation terms: {outcome}")

    coef_names = t["feature"].tolist()
    if not set(coef_names).issubset(features98.columns):
        raise RuntimeError(f"Missing engineered features: {outcome}")

    beta = t["beta_raw"].to_numpy(float)
    intercept = float(t["intercept_raw"].iloc[0])

    # Evaluate the SAVED equation, with no fitting or recalibration.
    predicted = (
        intercept
        + features98[coef_names].to_numpy(float) @ beta
    )
    observed = data98[s98.Y_TARGETS[outcome]].to_numpy(float)
    weights = data98["SURVEY_WT"].to_numpy(float)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 6),
        sharey=True
    )

    for ax, (period, subtitle) in zip(axes, periods98):

        mask = data98["SPLIT"].eq(period).to_numpy()

        y = observed[mask]
        yh = predicted[mask]
        w = weights[mask]

        r2, rmse, mae, bias = s98.wmetrics(y, yh, w)

        # Verify against the original Script 98 results.
        reference = metrics98.loc[
            metrics98["outcome"].eq(outcome)
            & metrics98["scope"].eq(scope)
            & metrics98["design"].eq(design)
            & metrics98["period"].eq(period)
        ]

        if len(reference) != 1:
            raise RuntimeError("Missing reference performance row")

        ref = reference.iloc[0]

        if (
            len(y) != int(ref["n"])
            or not np.isclose(
                r2, ref["weighted_R2"],
                atol=1e-5, rtol=0
            )
            or not np.isclose(
                mae, ref["weighted_MAE"],
                atol=1e-5, rtol=0
            )
        ):
            raise RuntimeError(
                f"Frozen equation reproduction failed: "
                f"{outcome}, {period}"
            )

        # A: Actual participant-level outcome distribution.
        ax.hexbin(
            yh, y,
            gridsize=(30, 24),
            mincnt=1,
            bins="log",
            cmap=CMAP,
            linewidths=0,
            alpha=0.85,
            zorder=2
        )

        # B: Weighted calibration curve in prediction deciles.
        edges = np.unique(
            np.quantile(yh, np.linspace(0, 1, 11))
        )

        if len(edges) > 2:
            bins = np.searchsorted(
                edges[1:-1], yh, side="right"
            )

            calibration = []

            for k in range(len(edges) - 1):
                sel = bins == k

                if not sel.any():
                    continue

                calibration.append([
                    np.average(yh[sel], weights=w[sel]),
                    np.average(y[sel], weights=w[sel]),
                ])

            calibration = np.array(calibration)

            ax.plot(
                calibration[:, 0],
                calibration[:, 1],
                "-o",
                color=DARK,
                linewidth=2.1,
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1.5,
                zorder=5
            )

        # Ideal prediction: observed = predicted.
        xmin = min(-0.3, float(yh.min()) - 0.35)
        xmax = max(1.0, float(yh.max()) + 0.35)

        ax.plot(
            [xmin, xmax],
            [xmin, xmax],
            "--",
            color=MUTED,
            linewidth=1.3,
            zorder=3
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.5, upper + 0.5)

        ax.set_title(
            subtitle,
            fontsize=13,
            fontweight="bold",
            color=TEXT,
            pad=14
        )

        ax.set_xlabel("Predicted score", fontsize=11)

        ax.text(
            0.04, 0.97,
            f"n = {len(y):,}\n"
            f"Weighted R² = {r2:+.3f}\n"
            f"Weighted MAE = {mae:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            color=TEXT,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.92,
                pad=4
            )
        )

        ax.grid(axis="y", color=GRID, alpha=0.3)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax, top=True, right=True)

    axes[0].set_ylabel("Observed score", fontsize=11)

    fig.suptitle(
        f"{label} | How closely does our fitted equation predict Y?",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=TEXT
    )

    fig.text(
        0.055, 0.875,
        f"Script 98 | {selected['method']} | {scope}/{design} | "
        "Coefficients frozen from NHANES 2005–08",
        fontsize=10,
        color=MUTED
    )

    fig.text(
        0.055, 0.045,
        "Hexagons: unweighted participant density (log scale). "
        "Blue line: survey-weighted decile calibration. "
        "Dashed: perfect prediction. "
        "Horizontal axes zoomed to fitted prediction ranges.",
        fontsize=9,
        color=MUTED
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.19,
        top=0.79,
        wspace=0.17
    )

    save_both(
        fig,
        f"98_observed_vs_predicted_{outcome}"
    )

    plt.close(fig)

    print(
        f"[VERIFIED] {outcome}: numerical equation "
        "reproduces stored Script 98 metrics."
    )


# =============================================================================
# FIGURE H — OBJECTIVE COGNITION: DOMAIN SPECIFICITY AND A/G ATTRIBUTION
# Script 101 | NHANES 2011–12 training → 2013–14 evaluation
# =============================================================================

R101 = PROJECT_ROOT / "EDA" / "Results"

models = pd.read_csv(R101 / "101_cognitive_models.csv")
intervals = pd.read_csv(R101 / "101_cognitive_paired_intervals.csv")
alloc = pd.read_csv(R101 / "101_cognitive_predictive_AG_allocation.csv")

domains = [
    ("DSST_processing_speed", "Processing speed (DSST)"),
    ("Animal_verbal_fluency", "Verbal fluency"),
    ("CERAD_delayed_memory", "Delayed memory"),
    ("CERAD_learning_sum", "Learning (CERAD)"),
]

fig, (ax, bx) = plt.subplots(
    1, 2,
    figsize=(14.9, 6.8),
    gridspec_kw={"width_ratios": [1.35, 1.0]}
)

fig.patch.set_facecolor("white")

# ------------------------------------------------------------------
# PANEL A — HOW MUCH DOES JOINT A/G IMPROVE EACH COGNITIVE DOMAIN?
#
# Normalise ΔMSE by the covariates-only MSE.
# This allows different cognitive-score scales to be compared.
# ------------------------------------------------------------------

ax.axvline(
    0, color=MUTED, linestyle="--",
    linewidth=1.4, zorder=1
)

ax.axhspan(
    2.55, 3.45,
    color=CMAP(0.10), alpha=0.52, zorder=0
)

for index, (key, label) in enumerate(domains):

    y = 3 - index

    base = models.loc[
        models.outcome.eq(key)
        & models.specification.eq("covariates")
    ]

    joint = models.loc[
        models.outcome.eq(key)
        & models.specification.eq("raw_A_plus_G")
    ]

    bounds = intervals.loc[
        intervals.outcome.eq(key)
        & intervals.candidate.eq("raw_A_plus_G")
    ]

    if not (len(base) == len(joint) == len(bounds) == 1):
        raise ValueError(f"Missing results for {key}")

    mse0 = float(base.iloc[0].test_MSE)
    delta = float(bounds.iloc[0].delta_MSE)

    if mse0 <= 0:
        raise ValueError("Invalid baseline MSE")

    if not np.isclose(
        delta,
        float(joint.iloc[0].delta_MSE_vs_covariates),
        atol=1e-5
    ):
        raise ValueError(f"ΔMSE mismatch for {key}")

    effect, low, high = 100 * np.array([
        delta,
        float(bounds.iloc[0].descriptive_ci_low),
        float(bounds.iloc[0].descriptive_ci_high)
    ]) / mse0

    color = VERY_DARK if index == 0 else MID

    ax.errorbar(
        effect, y,
        xerr=[[effect - low], [high - effect]],
        fmt="o",
        ms=9.5,
        color=color,
        ecolor=color,
        capsize=5,
        elinewidth=2.2,
        capthick=1.6,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=5
    )

    ax.text(
        effect, y + 0.29,
        f"{effect:+.1f}%",
        ha="center",
        fontsize=11,
        fontweight="bold" if index == 0 else "normal",
        color=color
    )

ax.set_xlim(-9, 17)
ax.set_ylim(-0.65, 3.8)

ax.set_yticks(
    [3, 2, 1, 0],
    [label for _, label in domains]
)

ax.tick_params(axis="y", length=0, pad=10)

ax.set_xlabel(
    "Reduction in covariates-only MSE (%)",
    fontsize=11
)

ax.set_title(
    "a   Do biomarkers improve each cognitive test?",
    loc="left",
    pad=19,
    fontsize=14.5,
    fontweight="bold",
    color=TEXT
)

ax.grid(axis="x", color=GRID, alpha=0.7)
ax.grid(axis="y", visible=False)

sns.despine(ax=ax, left=True, top=True, right=True)

# ------------------------------------------------------------------
# PANEL B — DSST PREDICTIVE CONTRIBUTION OF A AND G
# ------------------------------------------------------------------

key = "DSST_processing_speed"

parts = alloc.loc[
    alloc.outcome.eq(key)
].set_index("block")

A_gain = float(parts.loc["A", "shapley_delta_MSE"])
G_gain = float(parts.loc["G", "shapley_delta_MSE"])
total = float(parts.loc["A", "total_rawX_gain"])

if (
    A_gain <= 0
    or G_gain <= 0
    or not np.isclose(A_gain + G_gain, total, atol=1e-5)
):
    raise ValueError("Unexpected DSST allocation")

base = models.loc[
    models.outcome.eq(key)
    & models.specification.eq("covariates")
].iloc[0]

raw = models.loc[
    models.outcome.eq(key)
    & models.specification.eq("raw_A_plus_G")
].iloc[0]

z_model = models.loc[
    models.outcome.eq(key)
    & models.specification.eq("Z_shared_private")
].iloc[0]

bx.set_title(
    "b   DSST: who accounts for the improvement?",
    loc="left",
    pad=19,
    fontsize=14.5,
    fontweight="bold",
    color=TEXT
)

bx.set_xlim(0, 1)
bx.set_ylim(0, 1)
bx.axis("off")

bx.text(
    0.02, 0.88,
    f"R²  {base.test_R2:.3f}  →  {raw.test_R2:.3f}",
    transform=bx.transAxes,
    fontsize=23,
    fontweight="bold",
    color=VERY_DARK
)

bx.text(
    0.02, 0.80,
    f"Frozen Z yields R² = {z_model.test_R2:.3f}",
    transform=bx.transAxes,
    fontsize=11,
    color=MUTED
)

bx.text(
    0.02, 0.66,
    f"Allocated error reduction (ΔMSE = +{total:.2f})",
    transform=bx.transAxes,
    fontsize=11,
    color=TEXT
)

start, span = 0.02, 0.94

bx.barh(
    0.49, span * A_gain / total,
    height=0.18, left=start,
    color=VERY_DARK,
    edgecolor="white"
)

bx.barh(
    0.49, span * G_gain / total,
    height=0.18,
    left=start + span * A_gain / total,
    color=MID_LIGHT,
    edgecolor="white"
)

bx.text(
    start + span * A_gain / total / 2,
    0.49,
    f"A  {100 * A_gain / total:.1f}%",
    ha="center", va="center",
    fontsize=12,
    fontweight="bold",
    color="white"
)

bx.text(
    start + span * (A_gain + G_gain / 2) / total,
    0.49,
    f"G {100 * G_gain / total:.1f}%",
    ha="center", va="center",
    fontsize=9,
    fontweight="bold",
    color=TEXT
)

bx.text(
    0.02, 0.30,
    f"Hematology +{A_gain:.2f}    |    Glycemia +{G_gain:.2f}",
    transform=bx.transAxes,
    fontsize=11,
    color=TEXT
)

bx.text(
    0.02, 0.19,
    "Shapley shares partition predictive gain,",
    transform=bx.transAxes,
    fontsize=10,
    color=MUTED
)

bx.text(
    0.02, 0.14,
    "not causal shares of cognitive impairment.",
    transform=bx.transAxes,
    fontsize=10,
    color=MUTED
)

# ------------------------------------------------------------------
# PUBLICATION-STYLE FINISH
# ------------------------------------------------------------------

fig.suptitle(
    "Physiology improves processing speed, but not all cognitive domains",
    x=0.07, y=0.97,
    ha="left",
    fontsize=19,
    fontweight="bold",
    color=TEXT
)

fig.text(
    0.07, 0.91,
    "NHANES 2011–12 trained → 2013–14 held-out | "
    "Adults ≥60 | Fixed 7-biomarker panel",
    fontsize=10.5,
    color=MUTED
)

fig.text(
    0.07, 0.06,
    "a: Paired participant-bootstrap 95% intervals; "
    "normalised by each outcome's covariates-only MSE. "
    "Positive = lower prediction error. "
    "b: A/G Shapley allocation (no confidence intervals estimated).",
    fontsize=9.2,
    color=MUTED
)

fig.text(
    0.07, 0.029,
    "Exploratory observational comparisons; "
    "complex-survey inference and multiplicity adjustment "
    "remain outstanding.",
    fontsize=9.2,
    color=MUTED
)

fig.subplots_adjust(
    left=0.22, right=0.97,
    top=0.78, bottom=0.20,
    wspace=0.22
)

save_both(
    fig,
    "101_cognition_domain_specificity_and_AG_allocation"
)

plt.close(fig)


# =============================================================================
# QUICK CONSOLE SUMMARY
# =============================================================================

print("\n--- LITERATURE LANDSCAPE ---")
print(f"Total papers: {len(df)}")
print(
    f"STAR papers: "
    f"{(df['STAR?'] == 'Yes').sum()}"
)
print(
    f"High novelty threat: "
    f"{(df['Novelty Threat'] == 'High').sum()}"
)
print(
    f"Published 2025–26: "
    f"{df['Year'].isin([2025, 2026]).sum()}"
)
print(
    f"2026 high-threat papers: "
    f"{((df['Year'] == 2026) & (df['Novelty Threat'] == 'High')).sum()}"
)
print("\nGeographic clinical papers:")
print(geo[["country", "papers"]].to_string(index=False))

print(f"\nOutputs: {OUTDIR}")