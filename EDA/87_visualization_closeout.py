#!/usr/bin/env python
# =============================================================================
# 87_visualization_closeout.py
#
# Quick presentation-ready descriptive visuals for Wednesday:
#   1) PHQ severity composition by period
#   2) High-burden PHQ thresholds by period (>=10, >=15, >=20)
#   3) Exact PHQ score distribution by period
#   4) Hematological burden distribution by period
#   5) Glycaemic burden distribution by period
#   6) PHQ severity across hematological x glycaemic burden quadrants
#
# Naming recommendation:
#   H = haematological burden/state
#   G = glycaemic burden/state
#
# The script computes simple descriptive H and G burden indices from raw markers:
#   H burden: low Hb, low RBC, low MCV, high RDW
#   G burden: high HbA1c, high Glucose, high logInsulin
#
# These are for descriptive visualization only, not the main inferential model.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
FIGDIR = RESULTS / "87_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Prefer the richer Script-72 input because it usually still has raw physiology.
CANDIDATES = [
    RESULTS / "72_shared_private_model_input.csv",
    RESULTS / "73_frozen_phenotype_map_input.csv",
    ROOT / "72_shared_private_model_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
]
INPUT = next((p for p in CANDIDATES if p.exists()), None)
if INPUT is None:
    raise FileNotFoundError("Could not find Script-72 or Script-73 model input CSV.")

PHQ_OUTCOME = "PHQ9_TOTAL"
PERIODS = ["2005-2008", "2009-2018", "2021-2023"]

RAW_H = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
RAW_G = ["LBXGH", "LBXGLU", "logInsulin"]
ALT_MAP = {
    "LBXHGB": ["Hb", "HGB", "hemoglobin", "Hemoglobin"],
    "LBXRBCSI": ["RBC", "rbc"],
    "LBXMCVSI": ["MCV", "mcv"],
    "LBXRDW": ["RDW", "rdw"],
    "LBXGH": ["HbA1c", "GHB", "ghb", "LBDGH", "LBXGH"],
    "LBXGLU": ["Glucose", "GLU", "glucose", "LBXGLU"],
    "logInsulin": [
        "logInsulin", "LogInsulin", "LOG_INSULIN", "log_insulin",
        "LOG_IN", "log_in", "LOGIN", "LBXIN", "LBDINSI"
    ],
}

SEVERITY_ORDER = [
    "Minimal (0-4)",
    "Mild (5-9)",
    "Moderate (10-14)",
    "Moderately severe (15-19)",
    "Severe (20-27)",
]

def resolve_col(df, canonical):
    if canonical in df.columns:
        return canonical

    for alt in ALT_MAP.get(canonical, []):
        if alt in df.columns:
            return alt

    # Robust fallback for insulin naming differences across saved model inputs.
    if canonical == "logInsulin":
        candidates = []
        for c in df.columns:
            lc = c.lower()
            if (
                "insul" in lc
                or lc in {"log_in", "login", "lbxin", "lbdinsi"}
                or ("log" in lc and "in" == lc.replace("log_", ""))
            ):
                candidates.append(c)

        if len(candidates) == 1:
            return candidates[0]

        if candidates:
            # Prefer an already log-transformed variable if several exist.
            for c in candidates:
                lc = c.lower()
                if "log" in lc or c.upper() == "LOG_IN":
                    return c
            return candidates[0]

    return None

def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    return float(np.sum(w * x) / np.sum(w))

def weighted_sd(x, w):
    mu = weighted_mean(x, w)
    return float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))

def weighted_prop(mask, w):
    mask = np.asarray(mask, float)
    w = np.asarray(w, float)
    return float(np.sum(w * mask) / np.sum(w))

def weighted_quantile(x, w, probs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return np.interp(probs, cdf, x)

def severity_label(score):
    if pd.isna(score):
        return np.nan
    if score <= 4:
        return "Minimal (0-4)"
    if score <= 9:
        return "Mild (5-9)"
    if score <= 14:
        return "Moderate (10-14)"
    if score <= 19:
        return "Moderately severe (15-19)"
    return "Severe (20-27)"

def burden_quartile_labels(values, weights):
    qs = weighted_quantile(values, weights, [0.25, 0.50, 0.75])
    bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, duplicates="drop")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
d = pd.read_csv(INPUT)

if PHQ_OUTCOME not in d.columns:
    raise ValueError(f"{PHQ_OUTCOME} not found in {INPUT.name}")

if "PERIOD" not in d.columns:
    raise ValueError("PERIOD column missing.")

wt_col = "SURVEY_WT" if "SURVEY_WT" in d.columns else None
if wt_col is None:
    raise ValueError("SURVEY_WT missing.")

resolved = {c: resolve_col(d, c) for c in RAW_H + RAW_G}
missing = [k for k, v in resolved.items() if v is None]
if missing:
    insulin_like = [
        c for c in d.columns
        if ("insul" in c.lower() or c.upper() == "LOG_IN")
    ]
    raise ValueError(
        f"Could not resolve raw physiology columns: {missing}. "
        f"Insulin-like columns found: {insulin_like}"
    )

print("Resolved raw physiology columns:")
for k, v in resolved.items():
    print(f"  {k:12s} -> {v}")
print()

# Keep usable cohort for descriptive visuals.
use_cols = ["PERIOD", PHQ_OUTCOME, wt_col] + list(resolved.values())
q = d[
    d["PERIOD"].isin(PERIODS)
    & d[use_cols].notna().all(axis=1)
    & (pd.to_numeric(d[wt_col], errors="coerce") > 0)
].copy()

q[wt_col] = pd.to_numeric(q[wt_col], errors="coerce")
q[PHQ_OUTCOME] = pd.to_numeric(q[PHQ_OUTCOME], errors="coerce")

# Canonical aliases
q["Hb"] = pd.to_numeric(q[resolved["LBXHGB"]], errors="coerce")
q["RBC"] = pd.to_numeric(q[resolved["LBXRBCSI"]], errors="coerce")
q["MCV"] = pd.to_numeric(q[resolved["LBXMCVSI"]], errors="coerce")
q["RDW"] = pd.to_numeric(q[resolved["LBXRDW"]], errors="coerce")
q["HbA1c"] = pd.to_numeric(q[resolved["LBXGH"]], errors="coerce")
q["Glucose"] = pd.to_numeric(q[resolved["LBXGLU"]], errors="coerce")
q["logInsulin"] = pd.to_numeric(q[resolved["logInsulin"]], errors="coerce")

# Descriptive burden indices standardized on 2005-08 only.
disc = q[q["PERIOD"].eq("2005-2008")].copy()
w0 = disc[wt_col].to_numpy(float)

for v in ["Hb", "RBC", "MCV", "RDW", "HbA1c", "Glucose", "logInsulin"]:
    mu = weighted_mean(disc[v].to_numpy(float), w0)
    sd = weighted_sd(disc[v].to_numpy(float), w0)
    sd = sd if sd > 1e-12 else 1.0
    q[f"z_{v}"] = (q[v] - mu) / sd

# H burden: more burden = lower Hb/RBC/MCV and higher RDW
q["H_burden_index"] = (
    -q["z_Hb"] - q["z_RBC"] - q["z_MCV"] + q["z_RDW"]
) / 4.0

# G burden: more burden = higher HbA1c/Glucose/logInsulin
q["G_burden_index"] = (
    q["z_HbA1c"] + q["z_Glucose"] + q["z_logInsulin"]
) / 3.0

# Severity
q["PHQ_severity"] = q[PHQ_OUTCOME].apply(severity_label)

# Quartiles based on full weighted cohort
q["H_quartile"] = burden_quartile_labels(
    q["H_burden_index"].to_numpy(float),
    q[wt_col].to_numpy(float),
)
q["G_quartile"] = burden_quartile_labels(
    q["G_burden_index"].to_numpy(float),
    q[wt_col].to_numpy(float),
)

# Simple high/low quadrants
q["H_level"] = np.where(q["H_burden_index"] >= 0, "Higher H burden", "Lower H burden")
q["G_level"] = np.where(q["G_burden_index"] >= 0, "Higher G burden", "Lower G burden")
q["HG_quadrant"] = q["H_level"] + " + " + q["G_level"]

# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
summary_rows = []
for period in PERIODS:
    a = q[q["PERIOD"].eq(period)]
    w = a[wt_col].to_numpy(float)
    y = a[PHQ_OUTCOME].to_numpy(float)

    summary_rows.append({
        "period": period,
        "n": len(a),
        "weighted_PHQ_mean": weighted_mean(y, w),
        "weighted_H_burden_mean": weighted_mean(a["H_burden_index"], w),
        "weighted_G_burden_mean": weighted_mean(a["G_burden_index"], w),
        "weighted_prop_PHQ_ge10": weighted_prop(y >= 10, w),
        "weighted_prop_PHQ_ge15": weighted_prop(y >= 15, w),
        "weighted_prop_PHQ_ge20": weighted_prop(y >= 20, w),
    })

pd.DataFrame(summary_rows).to_csv(FIGDIR / "87_visual_summary.csv", index=False)

# Severity composition table
sev_rows = []
for period in PERIODS:
    a = q[q["PERIOD"].eq(period)]
    w = a[wt_col].to_numpy(float)
    for sev in SEVERITY_ORDER:
        mask = a["PHQ_severity"].eq(sev).to_numpy()
        sev_rows.append({
            "period": period,
            "severity": sev,
            "n": int(mask.sum()),
            "weighted_prop": weighted_prop(mask, w),
        })
sev_tbl = pd.DataFrame(sev_rows)
sev_tbl.to_csv(FIGDIR / "87_severity_table.csv", index=False)

# HG quadrant vs severity
quad_rows = []
for quad in sorted(q["HG_quadrant"].unique()):
    a = q[q["HG_quadrant"].eq(quad)]
    w = a[wt_col].to_numpy(float)
    for sev in SEVERITY_ORDER:
        mask = a["PHQ_severity"].eq(sev).to_numpy()
        quad_rows.append({
            "quadrant": quad,
            "severity": sev,
            "n": int(mask.sum()),
            "weighted_prop": weighted_prop(mask, w),
        })
quad_tbl = pd.DataFrame(quad_rows)
quad_tbl.to_csv(FIGDIR / "87_HG_quadrant_severity_table.csv", index=False)

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

# 1. Stacked severity by period
fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(PERIODS))
for sev in SEVERITY_ORDER:
    vals = []
    for period in PERIODS:
        row = sev_tbl[(sev_tbl["period"] == period) & (sev_tbl["severity"] == sev)].iloc[0]
        vals.append(row["weighted_prop"] * 100)
    vals = np.array(vals)
    ax.bar(PERIODS, vals, bottom=bottom, label=sev)
    bottom += vals
ax.set_ylabel("Weighted %")
ax.set_title("PHQ-9 severity composition by period")
ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(FIGDIR / "87_fig1_PHQ_severity_by_period.png", dpi=200)
plt.close(fig)

# 2. High-burden thresholds grouped bars
thresholds = [10, 15, 20]
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(PERIODS))
width = 0.22
for i, thr in enumerate(thresholds):
    vals = []
    for period in PERIODS:
        a = q[q["PERIOD"].eq(period)]
        vals.append(weighted_prop(a[PHQ_OUTCOME].to_numpy(float) >= thr, a[wt_col].to_numpy(float)) * 100)
    ax.bar(x + (i - 1) * width, vals, width, label=f"PHQ ≥ {thr}")
ax.set_xticks(x)
ax.set_xticklabels(PERIODS)
ax.set_ylabel("Weighted %")
ax.set_title("Higher depressive-symptom burden by period")
ax.legend()
fig.tight_layout()
fig.savefig(FIGDIR / "87_fig2_PHQ_thresholds_by_period.png", dpi=200)
plt.close(fig)

# 3. Exact PHQ distribution lines
fig, ax = plt.subplots(figsize=(10, 6))
scores = list(range(28))
for period in PERIODS:
    vals = []
    a = q[q["PERIOD"].eq(period)]
    w = a[wt_col].to_numpy(float)
    y = a[PHQ_OUTCOME].to_numpy(float)
    for s in scores:
        vals.append(weighted_prop(y == s, w) * 100)
    ax.plot(scores, vals, marker="o", label=period)
ax.set_xlabel("PHQ-9 total score")
ax.set_ylabel("Weighted %")
ax.set_title("Exact PHQ-9 score distribution by period")
ax.legend()
fig.tight_layout()
fig.savefig(FIGDIR / "87_fig3_PHQ_exact_distribution.png", dpi=200)
plt.close(fig)

# 4. H burden box-style summary by period (matplotlib boxplot, unweighted visual)
fig, ax = plt.subplots(figsize=(9, 6))
data = [q.loc[q["PERIOD"].eq(p), "H_burden_index"].to_numpy(float) for p in PERIODS]
ax.boxplot(data, tick_labels=PERIODS)
ax.set_ylabel("H burden index")
ax.set_title("Haematological burden distribution by period")
fig.tight_layout()
fig.savefig(FIGDIR / "87_fig4_H_burden_boxplot.png", dpi=200)
plt.close(fig)

# 5. G burden box-style summary by period
fig, ax = plt.subplots(figsize=(9, 6))
data = [q.loc[q["PERIOD"].eq(p), "G_burden_index"].to_numpy(float) for p in PERIODS]
ax.boxplot(data, tick_labels=PERIODS)
ax.set_ylabel("G burden index")
ax.set_title("Glycaemic burden distribution by period")
fig.tight_layout()
fig.savefig(FIGDIR / "87_fig5_G_burden_boxplot.png", dpi=200)
plt.close(fig)

# 6. Heatmap: moderate+ burden across H/G quartiles
heat = np.zeros((4, 4))
for i, hq in enumerate(["Q1 low", "Q2", "Q3", "Q4 high"]):
    for j, gq in enumerate(["Q1 low", "Q2", "Q3", "Q4 high"]):
        a = q[(q["H_quartile"] == hq) & (q["G_quartile"] == gq)]
        if len(a) == 0:
            heat[3 - i, j] = np.nan
        else:
            heat[3 - i, j] = weighted_prop(
                a[PHQ_OUTCOME].to_numpy(float) >= 10,
                a[wt_col].to_numpy(float)
            ) * 100

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(heat, aspect="auto")
ax.set_xticks(range(4))
ax.set_xticklabels(["Q1 low", "Q2", "Q3", "Q4 high"])
ax.set_yticks(range(4))
ax.set_yticklabels(["Q4 high", "Q3", "Q2", "Q1 low"])
ax.set_xlabel("G burden quartile")
ax.set_ylabel("H burden quartile")
ax.set_title("Weighted % with PHQ ≥ 10 across H × G burden quartiles")
for i in range(4):
    for j in range(4):
        if np.isfinite(heat[i, j]):
            ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center")
fig.colorbar(im, ax=ax, label="Weighted %")
fig.tight_layout()
fig.savefig(FIGDIR / "87_fig6_HG_heatmap_PHQ10.png", dpi=200)
plt.close(fig)

# 7. Severe-category bar by H/G quadrant
quad_order = sorted(q["HG_quadrant"].unique())
vals = []
for quad in quad_order:
    a = q[q["HG_quadrant"].eq(quad)]
    vals.append(weighted_prop(a["PHQ_severity"].eq("Severe (20-27)").to_numpy(), a[wt_col].to_numpy(float)) * 100)

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(quad_order, vals)
ax.set_ylabel("Weighted %")
ax.set_title("Severe depressive symptoms across H/G burden quadrants")
ax.tick_params(axis='x', rotation=20)
fig.tight_layout()
fig.savefig(FIGDIR / "87_fig7_severe_by_HG_quadrant.png", dpi=200)
plt.close(fig)

# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
print("=" * 110)
print("SCRIPT 87 — PRESENTATION VISUALIZATION CLOSEOUT")
print("=" * 110)
print(f"Input used: {INPUT}")
print(f"Usable analytic rows: {len(q)}")
print("\nRECOMMENDED TERMINOLOGY")
print("  H burden / H state = haematological burden / haematological state")
print("  G burden / G state = glycaemic burden / glycaemic state")
print("  Avoid saying only A/G without expansion on slides.")
print()
print("QUICK NUMBERS")
for row in summary_rows:
    print(
        f"  {row['period']}: "
        f"PHQ mean={row['weighted_PHQ_mean']:.2f}, "
        f"PHQ>=10={100*row['weighted_prop_PHQ_ge10']:.1f}%, "
        f"PHQ>=15={100*row['weighted_prop_PHQ_ge15']:.1f}%, "
        f"PHQ>=20={100*row['weighted_prop_PHQ_ge20']:.1f}%"
    )
print("\nFILES WRITTEN TO")
print(f"  {FIGDIR}")
print("MAIN FIGURES")
for name in [
    "87_fig1_PHQ_severity_by_period.png",
    "87_fig2_PHQ_thresholds_by_period.png",
    "87_fig3_PHQ_exact_distribution.png",
    "87_fig4_H_burden_boxplot.png",
    "87_fig5_G_burden_boxplot.png",
    "87_fig6_HG_heatmap_PHQ10.png",
    "87_fig7_severe_by_HG_quadrant.png",
]:
    print(f"  {name}")
print("=" * 110)
