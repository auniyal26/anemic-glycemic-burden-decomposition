#!/usr/bin/env python
# =============================================================================
# 88_publication_figure_pack.py
#
# Publication/thesis figure pack for the Computational Physiological
# Decomposition project.
#
# Design principles:
#   - clean white background
#   - Arial/Helvetica-style typography where available
#   - accessible Wong-inspired palette
#   - no background grids / no decorative effects
#   - vector export (PDF + SVG) plus 600-dpi PNG
#   - ~180 mm two-column figure width
#   - descriptive quantities kept separate from inferential quantities
#
# MAIN PAPER PANELS GENERATED AS SEPARATE FILES:
#   F1A  PHQ severity composition by period
#   F1B  Higher symptom-burden prevalence by period
#   F1C  Exact PHQ score distribution
#   F2A  Haematological marker profile across periods
#   F2B  Glycaemic marker profile across periods
#   F2C  PHQ>=10 across H1 x G1 quartiles
#   F3A  Somatic component-by-time coefficient heatmap
#   F3B  Cognitive-affective component-by-time coefficient heatmap
#   F3C  Total PHQ component-by-time coefficient heatmap
#   F4A  Deterministic reverse-bridge reconstruction
#   F4B  Conformal reverse-bridge calibration
#
# OPTIONAL / SUPPLEMENTARY:
#   F5A  Skip-informed simulation: R2 vs physiological continuity
#   F5B  Skip-informed simulation: uncertainty volume vs continuity
#   F5C  Delta-state simulation: incremental ΔY information
#
# Terminology:
#   H = haematological state
#   G = glycaemic state
#   H1/G1 = dominant outcome-independent within-domain PCA axes used only for
#           descriptive EDA visualization. They are NOT replacements for the
#           full shared/discordant/private representation Z.
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
OUT = RESULTS / "88_publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Publication style
# -------------------------------------------------------------------------

MM = 1 / 25.4
DOUBLE_COL_W = 180 * MM
SINGLE_COL_W = 88 * MM

# Wong / Nature-friendly accessible palette.
COL = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#777777",
    "lightgrey": "#D9D9D9",
}

PERIOD_COLORS = {
    "2005-2008": COL["blue"],
    "2009-2018": COL["orange"],
    "2021-2023": COL["green"],
}

SEVERITY_COLORS = {
    "0-4 minimal": "#D9D9D9",
    "5-9 mild": COL["sky"],
    "10-14 moderate": COL["yellow"],
    "15-19 moderately severe": COL["orange"],
    "20-27 severe": COL["vermillion"],
}

font_candidates = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": font_candidates,
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def save_all(fig, stem):
    for ext in ["pdf", "svg", "png"]:
        kwargs = dict(bbox_inches="tight", facecolor="white")
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(OUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(w[ok] * x[ok]) / np.sum(w[ok]))


def weighted_sd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    mu = weighted_mean(x[ok], w[ok])
    return float(np.sqrt(np.sum(w[ok] * (x[ok] - mu) ** 2) / np.sum(w[ok])))


def weighted_prop(mask, w):
    mask = np.asarray(mask, bool)
    w = np.asarray(w, float)
    ok = np.isfinite(w) & (w > 0)
    return float(np.sum(w[ok] * mask[ok]) / np.sum(w[ok]))


def weighted_quantile(x, w, probs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return np.interp(probs, cdf, x)


def weighted_pca_fit(X, w):
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    sw = w.sum()
    mu = np.sum(w[:, None] * X, axis=0) / sw
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw)
    sd = np.where(sd > 1e-12, sd, 1.0)

    Z = (X - mu) / sd
    Zw = Z * np.sqrt(w[:, None] / sw)
    _, s, vt = np.linalg.svd(Zw, full_matrices=False)
    loadings = vt.T
    var = s**2
    frac = var / var.sum()
    return mu, sd, loadings, frac


def score_pca(X, mu, sd, loadings):
    Z = (np.asarray(X, float) - mu) / sd
    return Z @ loadings


def resolve_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    lowered = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lowered:
            return lowered[n.lower()]
    return None


# -------------------------------------------------------------------------
# Input discovery
# -------------------------------------------------------------------------

periods = ["2005-2008", "2009-2018", "2021-2023"]

MODEL_INPUT_CANDS = [
    RESULTS / "72_shared_private_model_input.csv",
    RESULTS / "73_frozen_phenotype_map_input.csv",
    ROOT / "72_shared_private_model_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
]
MODEL_INPUT = next((p for p in MODEL_INPUT_CANDS if p.exists()), None)

SEVERITY_FILE = RESULTS / "86_phq_severity_distribution.csv"
PHQ_EXACT_FILE = RESULTS / "86_phq_exact_score_distribution.csv"
PHQ_SUMMARY_FILE = RESULTS / "86_phq_summary.csv"
TIME_COEF_FILE = RESULTS / "86_component_period_coefficients.csv"
TIME_TEST_FILE = RESULTS / "86_component_time_interaction_audit.csv"

BRIDGE81_CANDS = [
    RESULTS / "81_multidimensional_Y_bridge_summary.csv",
    RESULTS / "81_multidimensional_Y_bridge_v2_summary.csv",
]
BRIDGE81 = next((p for p in BRIDGE81_CANDS if p.exists()), None)

CONF83 = RESULTS / "83_conformal_reverse_bridge_summary.csv"
SIM84 = RESULTS / "84_skip_informed_reverse_bridge_simulation.csv"
SIM85 = RESULTS / "85_delta_state_informed_bridge_simulation.csv"

made = []
stats_rows = []

# =============================================================================
# FIGURE 1A — PHQ severity composition
# =============================================================================

if SEVERITY_FILE.exists():
    s = pd.read_csv(SEVERITY_FILE)
    sev_order = [
        "0-4 minimal",
        "5-9 mild",
        "10-14 moderate",
        "15-19 moderately severe",
        "20-27 severe",
    ]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.78, 3.25))
    bottom = np.zeros(len(periods))

    for sev in sev_order:
        vals = []
        counts = []
        for p in periods:
            r = s[(s["period"] == p) & (s["category"] == sev)].iloc[0]
            vals.append(100 * float(r["weighted_prop"]))
            counts.append(int(r["n"]))

        vals = np.asarray(vals)
        ax.bar(
            periods, vals, bottom=bottom,
            width=0.62,
            color=SEVERITY_COLORS[sev],
            edgecolor="white", linewidth=0.45,
            label=sev.replace("0-4 ", "").replace("5-9 ", "")
        )

        # Label segments when visible enough.
        for i, (v, n) in enumerate(zip(vals, counts)):
            if v >= 4.0:
                ax.text(
                    i, bottom[i] + v / 2,
                    f"{v:.1f}% (n={n:,})",
                    ha="center", va="center",
                    fontsize=6.3
                )
        bottom += vals

    ax.set_ylabel("Weighted participants (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Depressive-symptom severity increased in later NHANES periods", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        title="PHQ-9 category",
        frameon=False,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left"
    )
    save_all(fig, "F1A_PHQ_severity_composition")
    made.append("F1A")

# =============================================================================
# FIGURE 1B — higher symptom burden thresholds
# =============================================================================

if PHQ_SUMMARY_FILE.exists():
    p = pd.read_csv(PHQ_SUMMARY_FILE)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.72, 3.0))
    xs = np.arange(len(periods))

    specs = [
        ("weighted_prop_PHQ_ge10", "PHQ-9 ≥10", COL["blue"]),
        ("weighted_prop_PHQ_ge15", "PHQ-9 ≥15", COL["orange"]),
        ("weighted_prop_PHQ_ge20", "PHQ-9 ≥20", COL["vermillion"]),
    ]

    for col, label, color in specs:
        vals = [100 * float(p.loc[p["period"].eq(per), col].iloc[0]) for per in periods]
        ax.plot(xs, vals, marker="o", ms=5, lw=1.6, color=color, label=label)
        for x, y in zip(xs, vals):
            ax.text(x, y + 0.45, f"{y:.1f}%", ha="center", va="bottom", fontsize=4.5, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(periods)
    ax.set_ylabel("Weighted participants (%)")
    ax.set_title("Higher depressive-symptom burden became more prevalent over time", loc="left", fontweight="bold")
    ax.axhline(0, color=COL["black"], lw=0.6)
    clean_axis(ax)
    ax.legend(frameon=False,
    ncol=3,
    loc="upper left",
    fontsize=6)
    save_all(fig, "F1B_PHQ_high_burden_prevalence")
    made.append("F1B")

# =============================================================================
# FIGURE 1C — exact score distribution
# =============================================================================

if PHQ_EXACT_FILE.exists():
    e = pd.read_csv(PHQ_EXACT_FILE)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 3.15))
    for per in periods:
        q = e[e["period"].eq(per)].sort_values("PHQ9_score")
        ax.plot(
            q["PHQ9_score"],
            100 * q["weighted_prop"],
            lw=1.6,
            color=PERIOD_COLORS[per],
            label=per,
        )

    ax.set_xlabel("PHQ-9 total score")
    ax.set_ylabel("Weighted participants (%)")
    ax.set_xlim(0, 27)
    ax.set_title("The full PHQ-9 distribution shifts toward greater symptom burden", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False)
    save_all(fig, "F1C_PHQ_exact_distribution")
    made.append("F1C")

# =============================================================================
# FIGURE 2 — physiological raw-marker profiles + outcome-independent H1/G1 EDA
# =============================================================================

if MODEL_INPUT is not None:
    d = pd.read_csv(MODEL_INPUT)

    aliases = {
        "Hb": ["LBXHGB", "Hb", "HGB"],
        "RBC": ["LBXRBCSI", "RBC"],
        "MCV": ["LBXMCVSI", "MCV"],
        "RDW": ["LBXRDW", "RDW"],
        "HbA1c": ["LBXGH", "HbA1c", "GHB", "LBDGH"],
        "Glucose": ["LBXGLU", "Glucose", "GLU"],
        "logInsulin": ["LOG_IN", "logInsulin", "LogInsulin", "LOG_INSULIN", "log_insulin"],
    }
    cols = {k: resolve_col(d, v) for k, v in aliases.items()}

    # Keep only if raw physiology is resolvable.
    if all(cols[k] is not None for k in cols) and {"PERIOD", "SURVEY_WT", "PHQ9_TOTAL"}.issubset(d.columns):
        dd = d[d["PERIOD"].isin(periods)].copy()

        for k, c in cols.items():
            dd[k] = pd.to_numeric(dd[c], errors="coerce")
        dd["SURVEY_WT"] = pd.to_numeric(dd["SURVEY_WT"], errors="coerce")
        dd["PHQ9_TOTAL"] = pd.to_numeric(dd["PHQ9_TOTAL"], errors="coerce")

        raw_names = list(cols.keys())
        dd = dd[
            dd[raw_names + ["SURVEY_WT", "PHQ9_TOTAL"]].notna().all(axis=1)
            & (dd["SURVEY_WT"] > 0)
        ].copy()

        discovery = dd[dd["PERIOD"].eq("2005-2008")]

        # Discovery-standardized raw-marker profiles.
        raw_mu = {}
        raw_sd = {}
        for v in raw_names:
            raw_mu[v] = weighted_mean(discovery[v], discovery["SURVEY_WT"])
            raw_sd[v] = weighted_sd(discovery[v], discovery["SURVEY_WT"])
            dd[f"z_{v}"] = (dd[v] - raw_mu[v]) / (raw_sd[v] if raw_sd[v] > 1e-12 else 1.0)

        # F2A — H marker profile
        hvars = ["Hb", "RBC", "MCV", "RDW"]
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.82, 3.2))
        x = np.arange(len(hvars))
        offsets = [-0.18, 0.0, 0.18]

        for off, per in zip(offsets, periods):
            vals = [
                weighted_mean(
                    dd.loc[dd["PERIOD"].eq(per), f"z_{v}"],
                    dd.loc[dd["PERIOD"].eq(per), "SURVEY_WT"]
                )
                for v in hvars
            ]
            ax.plot(
                x + off, vals,
                marker="o", ms=5, lw=1.2,
                color=PERIOD_COLORS[per], label=per
            )

        ax.axhline(0, color=COL["black"], lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(["Haemoglobin", "RBC count", "MCV", "RDW"])
        ax.set_ylabel("Weighted mean (discovery SD)")
        ax.set_title("Haematological measurements show distinct temporal shifts", loc="left", fontweight="bold")
        clean_axis(ax)
        ax.legend(frameon=False, ncol=3, loc="upper left")
        save_all(fig, "F2A_haematological_marker_profile")
        made.append("F2A")

        # F2B — G marker profile
        gvars = ["HbA1c", "Glucose", "logInsulin"]
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.72, 3.2))
        x = np.arange(len(gvars))
        for off, per in zip(offsets, periods):
            vals = [
                weighted_mean(
                    dd.loc[dd["PERIOD"].eq(per), f"z_{v}"],
                    dd.loc[dd["PERIOD"].eq(per), "SURVEY_WT"]
                )
                for v in gvars
            ]
            ax.plot(
                x + off, vals,
                marker="o", ms=5, lw=1.2,
                color=PERIOD_COLORS[per], label=per
            )
        ax.axhline(0, color=COL["black"], lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(["HbA1c", "Fasting glucose", "log insulin"])
        ax.set_ylabel("Weighted mean (discovery SD)")
        ax.set_title("Glycaemic measurements show structured temporal shifts", loc="left", fontweight="bold")
        clean_axis(ax)
        ax.legend(frameon=False, ncol=3, loc="upper left")
        save_all(fig, "F2B_glycaemic_marker_profile")
        made.append("F2B")

        # Outcome-independent H1/G1 axes for descriptive burden landscape.
        H0 = discovery[hvars].to_numpy(float)
        G0 = discovery[gvars].to_numpy(float)
        w0 = discovery["SURVEY_WT"].to_numpy(float)

        hmu, hsd, hload, hfrac = weighted_pca_fit(H0, w0)
        gmu, gsd, gload, gfrac = weighted_pca_fit(G0, w0)

        Hscores = score_pca(dd[hvars].to_numpy(float), hmu, hsd, hload)
        Gscores = score_pca(dd[gvars].to_numpy(float), gmu, gsd, gload)

        # Orient H1 so larger = lower Hb; orient G1 so larger = higher HbA1c.
        H1 = Hscores[:, 0]
        G1 = Gscores[:, 0]
        if np.corrcoef(H1, dd["Hb"].to_numpy(float))[0, 1] > 0:
            H1 *= -1
            hload[:, 0] *= -1
        if np.corrcoef(G1, dd["HbA1c"].to_numpy(float))[0, 1] < 0:
            G1 *= -1
            gload[:, 0] *= -1

        dd["H1"] = H1
        dd["G1"] = G1

        # Weighted quartiles using discovery only, then frozen.
        hq = weighted_quantile(
            dd.loc[dd["PERIOD"].eq("2005-2008"), "H1"],
            dd.loc[dd["PERIOD"].eq("2005-2008"), "SURVEY_WT"],
            [0.25, 0.50, 0.75]
        )
        gq = weighted_quantile(
            dd.loc[dd["PERIOD"].eq("2005-2008"), "G1"],
            dd.loc[dd["PERIOD"].eq("2005-2008"), "SURVEY_WT"],
            [0.25, 0.50, 0.75]
        )

        dd["H1_q"] = pd.cut(
            dd["H1"], [-np.inf, *hq, np.inf],
            labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True
        )
        dd["G1_q"] = pd.cut(
            dd["G1"], [-np.inf, *gq, np.inf],
            labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True
        )

        heat = np.full((4, 4), np.nan)
        nheat = np.zeros((4, 4), dtype=int)
        qlabs = ["Q1", "Q2", "Q3", "Q4"]
        for i, h in enumerate(qlabs):
            for j, g in enumerate(qlabs):
                z = dd[(dd["H1_q"] == h) & (dd["G1_q"] == g)]
                nheat[i, j] = len(z)
                if len(z) > 0:
                    heat[i, j] = 100 * weighted_prop(
                        z["PHQ9_TOTAL"].to_numpy(float) >= 10,
                        z["SURVEY_WT"].to_numpy(float)
                    )

        # Custom white -> blue sequential colormap.
        cmap_seq = LinearSegmentedColormap.from_list(
            "seq", ["#FFFFFF", "#BFD7EA", COL["blue"]]
        )

        fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.35, 3.35))
        im = ax.imshow(heat[::-1, :], cmap=cmap_seq, aspect="equal")
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4\nhigh"])
        ax.set_yticklabels(["Q4\nhigh", "Q3", "Q2", "Q1\nlow"])
        ax.set_xlabel("G1: dominant glycaemic axis")
        ax.set_ylabel("H1: dominant haematological axis")
        ax.set_title("Moderate-or-greater depressive symptoms across H1 × G1", loc="left", fontweight="bold")

        for i in range(4):
            for j in range(4):
                val = heat[::-1, :][i, j]
                n = nheat[::-1, :][i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:.1f}%\n{n:,}",
                        ha="center", va="center",
                        fontsize=6.1,
                        color="black"
                    )

        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cb.set_label("Weighted PHQ-9 ≥10 (%)")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        save_all(fig, "F2C_H1_G1_PHQ10_heatmap")
        made.append("F2C")

        # Save PCA loading transparency table.
        load_rows = []
        for i, v in enumerate(hvars):
            load_rows.append({
                "domain": "H1",
                "variable": v,
                "loading": float(hload[i, 0]),
                "variance_fraction_PC1": float(hfrac[0]),
            })
        for i, v in enumerate(gvars):
            load_rows.append({
                "domain": "G1",
                "variable": v,
                "loading": float(gload[i, 0]),
                "variance_fraction_PC1": float(gfrac[0]),
            })
        pd.DataFrame(load_rows).to_csv(OUT / "F2C_H1_G1_loadings.csv", index=False)

# =============================================================================
# FIGURE 3 — temporal component coefficients
# =============================================================================

if TIME_COEF_FILE.exists():
    c = pd.read_csv(TIME_COEF_FILE)

    component_labels = {
        "SP_SHARED1": "Shared 1",
        "SP_SHARED2": "Shared 2",
        "SP_DISCORD1": "Discordant 1",
        "SP_DISCORD2": "Discordant 2",
        "SP_A_PRIVATE3": "H-private 3",
        "SP_A_PRIVATE4": "H-private 4",
        "SP_G_PRIVATE3": "G-private 3",
    }
    comp_order = list(component_labels)

    vmax = float(np.nanmax(np.abs(c["coefficient"])))
    vmax = max(vmax, 0.1)

    cmap_div = LinearSegmentedColormap.from_list(
        "div", [COL["blue"], "#FFFFFF", COL["vermillion"]]
    )

    outcome_files = [
        ("SOMATIC_SCORE", "F3A", "Somatic symptoms"),
        ("COGAFF_SUM", "F3B", "Cognitive-affective symptoms"),
        ("PHQ9_TOTAL", "F3C", "Total PHQ-9"),
    ]

    sig = None
    if TIME_TEST_FILE.exists():
        sig = pd.read_csv(TIME_TEST_FILE)

    for outcome, code, label in outcome_files:
        z = c[c["outcome"].eq(outcome)].copy()
        mat = np.full((len(comp_order), len(periods)), np.nan)

        for i, comp in enumerate(comp_order):
            for j, per in enumerate(periods):
                r = z[(z["component"] == comp) & (z["period"] == per)]
                if len(r):
                    mat[i, j] = float(r["coefficient"].iloc[0])

        fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.72, 3.5))
        im = ax.imshow(mat, cmap=cmap_div, vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods)
        ax.set_yticks(range(len(comp_order)))
        ax.set_yticklabels([component_labels[x] for x in comp_order])

        ax.set_title(f"{label}: component associations across time", loc="left", fontweight="bold")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center", fontsize=6.4)

        if sig is not None:
            ss = sig[sig["outcome"].eq(outcome)]
            for i, comp in enumerate(comp_order):
                r = ss[ss["component"].eq(comp)]
                if len(r) and float(r["block_time_interaction_q_BH"].iloc[0]) < 0.05:
                    ax.text(
                        len(periods) - 0.05, i,
                        " *", ha="left", va="center",
                        fontsize=9, fontweight="bold",
                        color=COL["black"]
                    )

        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cb.set_label("Coefficient")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        save_all(fig, f"{code}_{outcome}_time_coefficients")
        made.append(code)

# =============================================================================
# FIGURE 4A — deterministic reverse bridge
# =============================================================================

if BRIDGE81 is not None:
    b = pd.read_csv(BRIDGE81)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.76, 3.1))

    reps = [
        ("Y1_total", "PHQ total", COL["grey"]),
        ("Y2_somatic_cogaff", "Somatic + cognitive-affective", COL["blue"]),
        ("Y9_items", "9 PHQ items", COL["vermillion"]),
    ]
    xs = np.arange(len(periods))
    for rep, label, color in reps:
        z = b[b["representation"].eq(rep)]
        vals = [
            float(z.loc[z["period"].eq(per), "Y_to_Z_R2"].iloc[0])
            for per in periods
        ]
        ax.plot(xs, vals, marker="o", ms=5, lw=1.5, color=color, label=label)

    ax.axhline(0, color=COL["black"], lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(periods)
    ax.set_ylabel(r"$R^2$: phenotype $\rightarrow$ physiology")
    ax.set_title("Phenotype does not deterministically reconstruct physiological state", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower left")
    save_all(fig, "F4A_deterministic_reverse_bridge")
    made.append("F4A")

# =============================================================================
# FIGURE 4B — conformal calibration
# =============================================================================

if CONF83.exists():
    conf = pd.read_csv(CONF83)

    label_map = {
        "2007-2008_CAL": "2007–08 calibration",
        "2009-2018": "2009–18",
        "2021-2023": "2021–23",
    }
    colors = {
        "2007-2008_CAL": COL["grey"],
        "2009-2018": COL["orange"],
        "2021-2023": COL["green"],
    }

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W * 1.35, 3.2))
    nominal = np.array([0.50, 0.80, 0.95])

    ax.plot(
        nominal * 100, nominal * 100,
        ls="--", lw=1.0, color=COL["black"],
        label="Perfect calibration"
    )

    for period in ["2007-2008_CAL", "2009-2018", "2021-2023"]:
        r = conf[conf["period"].eq(period)].iloc[0]
        obs = np.array([
            float(r["coverage_50_unweighted"]),
            float(r["coverage_80_unweighted"]),
            float(r["coverage_95_unweighted"]),
        ])
        ax.plot(
            nominal * 100, obs * 100,
            marker="o", ms=5, lw=1.5,
            color=colors[period],
            label=label_map[period]
        )

    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Observed coverage (%)")
    ax.set_xlim(45, 100)
    ax.set_ylim(45, 100)
    ax.set_title("Conformal reverse regions remain calibrated under temporal transfer", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    save_all(fig, "F4B_conformal_reverse_calibration")
    made.append("F4B")

# =============================================================================
# FIGURE 5A/B — skip-informed simulation
# =============================================================================

if SIM84.exists():
    s84 = pd.read_csv(SIM84)

    model_specs = [
        ("Y_only", "Phenotype only", COL["grey"]),
        ("Zprev_only", "Previous physiology only", COL["orange"]),
        ("Y_plus_Zprev", "Phenotype + previous physiology", COL["blue"]),
    ]

    # F5A R2
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.70, 3.1))
    for model, label, color in model_specs:
        z = s84[s84["model"].eq(model)].sort_values("rho")
        ax.plot(z["rho"], z["test_R2"], marker="o", ms=4.5, lw=1.5, color=color, label=label)
    ax.set_xlabel(r"Physiological continuity ($\rho$)")
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title("Prior physiological state constrains current-state reconstruction in simulation", loc="left", fontweight="bold")
    clean_axis(ax)
    ax.legend(frameon=False)
    save_all(fig, "F5A_skip_simulation_R2")
    made.append("F5A")

    # F5B relative uncertainty volume
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.70, 3.1))
    z = s84[s84["model"].eq("Y_plus_Zprev")].sort_values("rho")
    ax.plot(
        z["rho"], z["relative_volume_vs_Y_only"],
        marker="o", ms=4.5, lw=1.6, color=COL["blue"]
    )
    ax.axhline(1, color=COL["black"], lw=0.8, ls="--")
    ax.set_xlabel(r"Physiological continuity ($\rho$)")
    ax.set_ylabel("Uncertainty volume / phenotype-only")
    ax.set_title("An informed reference channel sharply narrows plausible physiology", loc="left", fontweight="bold")
    clean_axis(ax)
    save_all(fig, "F5B_skip_simulation_uncertainty")
    made.append("F5B")

# =============================================================================
# FIGURE 5C — delta-state incremental information
# =============================================================================

if SIM85.exists():
    s85 = pd.read_csv(SIM85)
    z = s85[s85["model"].eq("Zprev_plus_deltaY")].copy()

    rhos = sorted(z["rho"].unique())
    kappas = sorted(z["kappa"].unique())
    mat = np.full((len(kappas), len(rhos)), np.nan)

    for i, k in enumerate(kappas):
        for j, r in enumerate(rhos):
            rr = z[(z["kappa"] == k) & (z["rho"] == r)]
            if len(rr):
                # Some versions store this directly, otherwise reconstruct using baseline.
                if "deltaZ_R2_gain_vs_Zprev" in rr.columns and pd.notna(rr["deltaZ_R2_gain_vs_Zprev"].iloc[0]):
                    gain = float(rr["deltaZ_R2_gain_vs_Zprev"].iloc[0])
                else:
                    base = s85[
                        (s85["kappa"] == k)
                        & (s85["rho"] == r)
                        & (s85["model"] == "Zprev_only")
                    ]
                    gain = float(rr["deltaZ_R2"].iloc[0] - base["deltaZ_R2"].iloc[0])
                mat[i, j] = gain

    cmap_gain = LinearSegmentedColormap.from_list(
        "gain", ["#FFFFFF", "#BFD7EA", COL["blue"]]
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_W * 0.65, 3.0))
    im = ax.imshow(mat, cmap=cmap_gain, aspect="auto")
    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([f"{r:.2f}" for r in rhos])
    ax.set_yticks(range(len(kappas)))
    ax.set_yticklabels([f"{k:.2f}" for k in kappas])
    ax.set_xlabel(r"Physiological continuity ($\rho$)")
    ax.set_ylabel(r"Phenotype-noise persistence ($\kappa$)")
    ax.set_title(r"Phenotype change adds modest information about $\Delta Z$ in simulation", loc="left", fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center", fontsize=6.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label(r"Incremental $\Delta R^2$")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    save_all(fig, "F5C_delta_state_incremental_information")
    made.append("F5C")

# =============================================================================
# README / captions
# =============================================================================

readme = f"""
PUBLICATION FIGURE PACK — SCRIPT 88
===================================

Terminology
-----------
H = haematological state
G = glycaemic state

Use "H state" and "G state" after defining them once.
Avoid "anaemic effect" as the general name of the H block because the block
contains Hb, RBC, MCV and RDW and therefore represents broader measured
haematological state.

H1/G1 note
----------
F2C uses outcome-independent within-domain PCA axes solely as a descriptive
EDA visualization:
  H1 = dominant haematological axis, oriented so larger H1 corresponds to
       lower haemoglobin.
  G1 = dominant glycaemic axis, oriented so larger G1 corresponds to
       higher HbA1c.

These DO NOT replace the full shared/discordant/private Z representation.

Recommended MAIN paper figures
------------------------------
F1A  PHQ severity composition
F2A  Haematological marker profile
F2B  Glycaemic marker profile
F2C  H1 x G1 depressive-burden landscape
F3B  Cognitive-affective temporal coefficient map
F3C  Total-PHQ temporal coefficient map
F4A  Deterministic reverse bridge
F4B  Conformal reverse calibration

Recommended supplementary / poster-support figures
---------------------------------------------------
F1B, F1C, F3A, F5A, F5B, F5C

Interpretive boundaries
-----------------------
- Weighted descriptive figures are descriptive, not causal.
- H1/G1 are exploratory visual summaries, not final disease labels.
- Component-by-time coefficients are association/stability results.
- Reverse-bridge conformal regions quantify plausible physiology; they do not
  imply phenotype uniquely identifies physiological state.
- Script 84/85 figures are controlled simulations, not observed longitudinal
  evidence.

Generated panels
----------------
{", ".join(made)}
"""

(OUT / "README_figure_strategy.txt").write_text(readme, encoding="utf-8")

print("=" * 112)
print("SCRIPT 88 — PUBLICATION FIGURE PACK")
print("=" * 112)
print(f"Output folder: {OUT}")
print(f"Generated panels: {', '.join(made) if made else 'none'}")
print()
print("Each panel is exported as:")
print("  PDF  — vector, preferred for paper/poster")
print("  SVG  — vector, easy to edit")
print("  PNG  — 600 dpi, convenient for PowerPoint")
print()
print("Terminology locked:")
print("  H state = haematological state")
print("  G state = glycaemic state")
print()
print("See README_figure_strategy.txt for main-vs-supplementary recommendations.")
print("=" * 112)
