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