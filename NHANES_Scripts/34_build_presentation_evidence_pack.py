from pathlib import Path
import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 34 — PRESENTATION EVIDENCE PACK
# ============================================================
# Purpose:
#   Build a presentation-ready evidence pack from ALREADY-COMPUTED
#   project artifacts. This script does NOT refit the scientific models,
#   change thresholds, tune definitions, or overwrite preserved data.
#
# Outputs:
#   Results/Presentation_Evidence/
#       Figures/*.png
#       Tables/*.csv
#       34_PRESENTATION_EVIDENCE_GUIDE.md
#       34_generation_manifest.csv
#
# Important:
#   "y" below means the operational working representation used so far,
#   NOT a claimed final physiological representation. Final y -> y_hat
#   reconstruction for the final A_i/G_j model remains future work.
# ============================================================

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
SCRIPTS = ROOT / "Scripts"

OUT = RESULTS / "Presentation_Evidence"
FIG = OUT / "Figures"
TAB = OUT / "Tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC_ITEMS = ["DPQ030", "DPQ040", "DPQ050"]

CORE_FILES = {
    "0506": {
        "cbc": "CBC_D.XPT",
        "demo": "DEMO_D.xpt",
        "ghb": "GHB_D.XPT",
        "dpq": "DPQ_D.XPT",
    },
    "0708": {
        "cbc": "CBC_E.XPT",
        "demo": "DEMO_E.XPT",
        "ghb": "GHB_E.XPT",
        "dpq": "DPQ_E.XPT",
    },
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def savefig(name):
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def annotate_bars(ax, fmt="{:.0f}", rotation=0):
    for patch in ax.patches:
        h = patch.get_height()
        if np.isfinite(h):
            ax.annotate(
                fmt.format(h),
                (patch.get_x() + patch.get_width() / 2, h),
                ha="center", va="bottom",
                xytext=(0, 4), textcoords="offset points",
                fontsize=9, rotation=rotation
            )

def normalize_phq(d):
    d = d.copy()
    for c in ITEMS:
        if c not in d.columns:
            continue
        x = pd.to_numeric(d[c], errors="coerce")
        x.loc[x.abs() < 1e-10] = 0
        x.loc[~x.isin([0, 1, 2, 3])] = np.nan
        d[c] = x
    return d

def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if ok.sum() == 0 or w[ok].sum() == 0:
        return np.nan
    return np.sum(w[ok] * x[ok]) / np.sum(w[ok])

def weighted_sd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if ok.sum() == 0 or w[ok].sum() == 0:
        return np.nan
    m = weighted_mean(x[ok], w[ok])
    return np.sqrt(np.sum(w[ok] * (x[ok] - m) ** 2) / np.sum(w[ok]))

def safe_read_csv(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None

def fmt_num(x, digits=4):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def find_row(df, **conds):
    if df is None:
        return None
    q = df.copy()
    for k, v in conds.items():
        if k not in q.columns:
            return None
        q = q[q[k].astype(str) == str(v)]
    return q.iloc[0] if len(q) else None

print()
print("34 — PRESENTATION EVIDENCE PACK")
print("===============================")
print("READ-ONLY evidence build: no model refitting and no protocol changes.")
print()

# ------------------------------------------------------------
# 0) HASH THE PRESERVED CORE BEFORE ANYTHING ELSE
# ------------------------------------------------------------
core_working_path = PROCESSED / "nhanes_core_working.parquet"
core_hash_before = sha256(core_working_path) if core_working_path.exists() else None

# ------------------------------------------------------------
# 1) REBUILD DISCOVERY CORE FROM RAW XPT FOR FLOW + DESCRIPTIVES
# ------------------------------------------------------------
frames = []

for cycle, f in CORE_FILES.items():
    p = DATA / cycle
    missing = [name for name in f.values() if not (p / name).exists()]
    if missing:
        raise FileNotFoundError(f"{cycle}: missing raw core files: {missing}")

    cbc = pd.read_sas(p / f["cbc"], format="xport")
    demo = pd.read_sas(p / f["demo"], format="xport")
    ghb = pd.read_sas(p / f["ghb"], format="xport")
    dpq = normalize_phq(pd.read_sas(p / f["dpq"], format="xport"))

    keep_cbc = ["SEQN"] + [c for c in [
        "LBXHGB", "LBXRBCSI", "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXMC", "LBXRDW"
    ] if c in cbc.columns]

    keep_demo = ["SEQN"] + [c for c in [
        "RIDAGEYR", "RIAGENDR", "RIDRETH1"
    ] if c in demo.columns]

    keep_ghb = ["SEQN"] + [c for c in ["LBXGH"] if c in ghb.columns]
    keep_dpq = ["SEQN"] + ITEMS

    d = (
        demo[keep_demo]
        .merge(cbc[keep_cbc], on="SEQN", how="inner", validate="one_to_one")
        .merge(ghb[keep_ghb], on="SEQN", how="inner", validate="one_to_one")
        .merge(dpq[keep_dpq], on="SEQN", how="inner", validate="one_to_one")
    )
    d["CYCLE"] = cycle
    frames.append(d)

raw_merged_allages = pd.concat(frames, ignore_index=True)
core18 = raw_merged_allages[raw_merged_allages["RIDAGEYR"] >= 18].copy()

core18["PHQ9_TOTAL"] = core18[ITEMS].sum(axis=1, min_count=9)
core18["SOMATIC_SCORE"] = core18[SOMATIC_ITEMS].sum(axis=1, min_count=3)
core18["HB_THRESHOLD"] = np.where(core18["RIAGENDR"] == 1, 13.0, 12.0)
core18["A"] = (core18["HB_THRESHOLD"] - core18["LBXHGB"]).clip(lower=0)
core18["G_HBA1C"] = (core18["LBXGH"] - 5.7).clip(lower=0)
core18["AG_HBA1C"] = core18["A"] * core18["G_HBA1C"]

complete_core = core18.dropna(subset=["LBXHGB", "LBXGH", "PHQ9_TOTAL"]).copy()

flow = [
    ("Same-person raw core match, age 18+", len(core18)),
    ("Complete hemoglobin", int(core18["LBXHGB"].notna().sum())),
    ("Complete HbA1c", int(core18["LBXGH"].notna().sum())),
    ("Complete PHQ-9", int(core18["PHQ9_TOTAL"].notna().sum())),
    ("Complete A / G / PHQ", len(complete_core)),
]

parity_input = RESULTS / "25_survey_parity_input.csv"
survey = pd.read_csv(parity_input) if parity_input.exists() else None
if survey is not None:
    flow.append(("Final survey X3 analysis cohort", len(survey)))

flow_df = pd.DataFrame(flow, columns=["stage", "n"])
flow_df["retained_vs_previous_pct"] = (
    flow_df["n"] / flow_df["n"].shift(1) * 100
)
flow_df.loc[0, "retained_vs_previous_pct"] = 100.0
flow_df["retained_vs_first_pct"] = flow_df["n"] / flow_df.loc[0, "n"] * 100
flow_df.to_csv(TAB / "34_sample_flow.csv", index=False)

plt.figure(figsize=(10, 5.5))
ax = plt.gca()
ax.bar(np.arange(len(flow_df)), flow_df["n"])
ax.set_xticks(np.arange(len(flow_df)))
ax.set_xticklabels(flow_df["stage"], rotation=28, ha="right")
ax.set_ylabel("Participants")
ax.set_title("Discovery cohort flow: NHANES 2005–2008")
annotate_bars(ax, "{:,.0f}")
savefig("34_01_discovery_cohort_flow.png")

# ------------------------------------------------------------
# 2) TRANSFER COHORT COUNTS
# ------------------------------------------------------------
transfer_summary = safe_read_csv("30_transfer_harmonization_summary.csv")
if transfer_summary is not None:
    transfer_summary.to_csv(TAB / "34_transfer_cohort_counts.csv", index=False)

    labels = transfer_summary["cycle"].astype(str).tolist()
    eligible = transfer_summary["eligible_20plus_nonpreg_n"].to_numpy()
    complete = transfer_summary["primary_complete_case_n"].to_numpy()
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(9.5, 5.2))
    ax = plt.gca()
    ax.bar(x - width/2, eligible, width=width, label="Eligible 20+ non-pregnant")
    ax.bar(x + width/2, complete, width=width, label="Primary complete case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Participants")
    ax.set_title("Temporal-validation cohort sizes")
    ax.legend()
    savefig("34_02_transfer_cohort_sizes.png")

# ------------------------------------------------------------
# 3) DATA DISTRIBUTIONS
# ------------------------------------------------------------
# Use complete A/G/P core for transparent raw-sample distributions.
dist_vars = {
    "Age (years)": "RIDAGEYR",
    "Hemoglobin (g/dL)": "LBXHGB",
    "HbA1c (%)": "LBXGH",
    "PHQ-9 total": "PHQ9_TOTAL",
    "Somatic score": "SOMATIC_SCORE",
    "A: Hb deficit (g/dL)": "A",
    "G: HbA1c excess (%-point)": "G_HBA1C",
}

desc_rows = []
for label, col in dist_vars.items():
    x = pd.to_numeric(complete_core[col], errors="coerce").dropna()
    desc_rows.append({
        "variable": label,
        "n": len(x),
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "median": x.median(),
        "p25": x.quantile(.25),
        "p75": x.quantile(.75),
        "min": x.min(),
        "max": x.max(),
        "zero_pct": 100 * (x == 0).mean(),
    })
pd.DataFrame(desc_rows).to_csv(TAB / "34_core_distribution_summary.csv", index=False)

for num, (label, col) in enumerate([
    ("Age (years)", "RIDAGEYR"),
    ("Hemoglobin (g/dL)", "LBXHGB"),
    ("HbA1c (%)", "LBXGH"),
    ("PHQ-9 total", "PHQ9_TOTAL"),
    ("Somatic score (sleep + fatigue + appetite)", "SOMATIC_SCORE"),
], start=3):
    x = pd.to_numeric(complete_core[col], errors="coerce").dropna()
    plt.figure(figsize=(8.5, 4.8))
    plt.hist(x, bins=35)
    plt.xlabel(label)
    plt.ylabel("Participants")
    plt.title(f"Discovery distribution: {label}")
    savefig(f"34_{num:02d}_{col.lower()}_distribution.png")

# ------------------------------------------------------------
# 4) RAW x -> OPERATIONAL y TRANSFORMATION
# ------------------------------------------------------------
# A transformation plot
plot_a = complete_core[["LBXHGB", "RIAGENDR", "A"]].dropna().copy()
if len(plot_a) > 5000:
    plot_a = plot_a.sample(5000, random_state=42)

plt.figure(figsize=(8.5, 5))
for sex, marker, label in [(1, "o", "Male: threshold 13"), (2, "x", "Female: threshold 12")]:
    d = plot_a[plot_a["RIAGENDR"] == sex]
    plt.scatter(d["LBXHGB"], d["A"], s=12, alpha=0.35, marker=marker, label=label)
plt.xlabel("Raw hemoglobin, Hb (g/dL)")
plt.ylabel("A = max(sex-specific Hb threshold − Hb, 0)")
plt.title("Transformation from raw hemoglobin to hematological burden A")
plt.legend()
savefig("34_08_hb_to_A_transform.png")

# G transformation plot
plot_g = complete_core[["LBXGH", "G_HBA1C"]].dropna().copy()
if len(plot_g) > 5000:
    plot_g = plot_g.sample(5000, random_state=42)

plt.figure(figsize=(8.5, 5))
plt.scatter(plot_g["LBXGH"], plot_g["G_HBA1C"], s=12, alpha=0.35)
plt.xlabel("Raw HbA1c (%)")
plt.ylabel("G = max(HbA1c − 5.7, 0)")
plt.title("Transformation from raw HbA1c to glycemic burden G")
savefig("34_09_hba1c_to_G_transform.png")

# Operational y table.
# This is a summary of the working representation actually used so far.
y_cols = ["A", "G_HBA1C", "AG_HBA1C", "PHQ9_TOTAL", "SOMATIC_SCORE"]
y_summary = []
for col in y_cols:
    x = pd.to_numeric(complete_core[col], errors="coerce").dropna()
    y_summary.append({
        "y_dimension": col,
        "n": len(x),
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "median": x.median(),
        "p25": x.quantile(.25),
        "p75": x.quantile(.75),
        "zero_pct": 100 * (x == 0).mean(),
        "note": {
            "A": "Sex-thresholded hemoglobin deficit",
            "G_HBA1C": "HbA1c excess above 5.7%",
            "AG_HBA1C": "Scalar interaction A x G",
            "PHQ9_TOTAL": "Sum of 9 preserved ordinal PHQ items",
            "SOMATIC_SCORE": "Sleep + fatigue + appetite item sum",
        }[col]
    })
y_summary = pd.DataFrame(y_summary)
y_summary.to_csv(TAB / "34_operational_y_summary.csv", index=False)

corr = complete_core[y_cols].corr()
corr.to_csv(TAB / "34_operational_y_correlations.csv")

plt.figure(figsize=(7.5, 6))
ax = plt.gca()
im = ax.imshow(corr.to_numpy(), aspect="auto")
ax.set_xticks(np.arange(len(y_cols)))
ax.set_yticks(np.arange(len(y_cols)))
ax.set_xticklabels(y_cols, rotation=35, ha="right")
ax.set_yticklabels(y_cols)
for i in range(len(y_cols)):
    for j in range(len(y_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_title("Operational working representation y: observed correlation structure")
savefig("34_10_operational_y_correlation.png")

# Conceptual x -> y diagram.
plt.figure(figsize=(12, 5.5))
ax = plt.gca()
ax.axis("off")

boxes = [
    (0.06, 0.57, "Preserved x\nRaw same-person state\nHb · CBC · HbA1c · PHQ items · X"),
    (0.37, 0.57, "T(x)\nDomain-informed transform\nA = max(threshold−Hb,0)\nG = max(HbA1c−5.7,0)"),
    (0.68, 0.57, "Operational y\nA · G · A×G\nPHQ items / total / somatic\ncovariates + participant mapping"),
]
for x0, y0, txt in boxes:
    ax.text(
        x0, y0, txt, transform=ax.transAxes,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.6"),
        fontsize=11
    )
ax.annotate("", xy=(0.36, 0.57), xytext=(0.27, 0.57),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.5))
ax.annotate("", xy=(0.67, 0.57), xytext=(0.58, 0.57),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.5))
ax.text(
    0.50, 0.16,
    "Important: x is never replaced. y is a working representation linked back to x.\n"
    "The final physiological y and its final y→ŷ reconstruction-loss test are NOT yet complete.",
    transform=ax.transAxes, ha="center", va="center", fontsize=11
)
ax.set_title("What x → T(x) → y means in the experiments completed so far")
savefig("34_11_x_to_y_operational_pipeline.png")

# ------------------------------------------------------------
# 5) SIMPLE CLINICAL GROUPS
# ------------------------------------------------------------
complete_core["GROUP"] = np.select(
    [
        (complete_core["A"] <= 0) & (complete_core["G_HBA1C"] <= 0),
        (complete_core["A"] <= 0) & (complete_core["G_HBA1C"] > 0),
        (complete_core["A"] > 0) & (complete_core["G_HBA1C"] <= 0),
        (complete_core["A"] > 0) & (complete_core["G_HBA1C"] > 0),
    ],
    ["Neither", "G only", "A only", "Both"],
    default="Unclassified"
)
group_order = ["Neither", "G only", "A only", "Both"]
group_rows = []
for g in group_order:
    d = complete_core[complete_core["GROUP"] == g]
    group_rows.append({
        "group": g,
        "n": len(d),
        "mean_PHQ9": d["PHQ9_TOTAL"].mean(),
        "median_PHQ9": d["PHQ9_TOTAL"].median(),
        "PHQ9_ge_10_n": int((d["PHQ9_TOTAL"] >= 10).sum()),
        "PHQ9_ge_10_pct": 100 * (d["PHQ9_TOTAL"] >= 10).mean(),
        "mean_somatic": d["SOMATIC_SCORE"].mean(),
    })
group_df = pd.DataFrame(group_rows)
group_df.to_csv(TAB / "34_clinical_group_descriptives.csv", index=False)

plt.figure(figsize=(8.5, 4.8))
ax = plt.gca()
ax.bar(group_df["group"], group_df["n"])
ax.set_ylabel("Participants")
ax.set_title("A/G working groups in the 2005–2008 complete core")
annotate_bars(ax, "{:,.0f}")
savefig("34_12_ag_group_counts.png")

plt.figure(figsize=(8.5, 4.8))
ax = plt.gca()
ax.bar(group_df["group"], group_df["mean_PHQ9"])
ax.set_ylabel("Mean PHQ-9 total")
ax.set_title("Observed PHQ-9 burden across A/G groups")
annotate_bars(ax, "{:.2f}")
savefig("34_13_ag_group_mean_phq9.png")

# ------------------------------------------------------------
# 6) WEIGHTED ANALYSIS-COHORT DESCRIPTIVES
# ------------------------------------------------------------
if survey is not None:
    working = pd.read_parquet(core_working_path) if core_working_path.exists() else None
    s = survey.copy()
    if working is not None:
        wcols = [c for c in ["SEQN", "CYCLE", "LBXHGB", "LBXGH"] if c in working.columns]
        w = working[wcols].copy()
        s["CYCLE"] = s["CYCLE"].astype(str).str.replace(".0", "", regex=False)
        w["CYCLE"] = w["CYCLE"].astype(str).str.replace(".0", "", regex=False)
        s = s.merge(w, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

    weighted_rows = []
    for col in ["RIDAGEYR", "LBXHGB", "LBXGH", "A", "G_HBA1C", "PHQ9_TOTAL", "SOMATIC_SCORE"]:
        if col not in s.columns:
            continue
        x = pd.to_numeric(s[col], errors="coerce")
        ww = pd.to_numeric(s["WTMEC4YR"], errors="coerce")
        weighted_rows.append({
            "variable": col,
            "n_nonmissing": int(x.notna().sum()),
            "unweighted_mean": x.mean(),
            "unweighted_sd": x.std(ddof=1),
            "survey_weighted_mean": weighted_mean(x, ww),
            "survey_weighted_sd": weighted_sd(x, ww),
        })
    pd.DataFrame(weighted_rows).to_csv(
        TAB / "34_survey_cohort_weighted_descriptives.csv", index=False
    )

# ------------------------------------------------------------
# 7) FORMAL DISCOVERY EFFECTS + INCREMENTAL INFORMATION
# ------------------------------------------------------------
coef26 = safe_read_csv("26_design_based_coefficients.csv")
inc27 = safe_read_csv("27_incremental_information_tests.csv")

key_result_rows = []

if coef26 is not None:
    d = coef26[
        (coef26["population"].astype(str) == "Pooled") &
        (coef26["outcome"].astype(str) == "SOMATIC_SCORE") &
        (coef26["model"].astype(str) == "Additive_A+G+X")
    ].copy()

    if len(d):
        labels = {"A": "A: Hb deficit", "G_HBA1C": "G: HbA1c excess"}
        order_terms = ["A", "G_HBA1C"]
        d = d[d["term"].isin(order_terms)].set_index("term").reindex(order_terms).reset_index()

        y = np.arange(len(d))
        plt.figure(figsize=(8.3, 4.6))
        ax = plt.gca()
        for i, row in d.iterrows():
            ax.errorbar(
                row["beta"], i,
                xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]],
                fmt="o", capsize=4
            )
            ax.text(row["ci_high"] + 0.015, i,
                    f"β={row['beta']:.3f}, p={row['p']:.4f}", va="center", fontsize=9)
        ax.axvline(0, linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[t] for t in d["term"]])
        ax.set_xlabel("Survey-weighted somatic-score coefficient (95% CI)")
        ax.set_title("Discovery: strict additive A + G + X3 model, pooled 2005–2008")
        savefig("34_14_discovery_primary_effects.png")

        for _, r in d.iterrows():
            key_result_rows.append({
                "section": "Discovery strict additive survey model",
                "population": "2005-2008 pooled",
                "outcome": "Somatic score",
                "term": r["term"],
                "estimate": r["beta"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "p": r["p"],
                "interpretation": (
                    "Adjusted association conditional on the other physiology term and X3; "
                    "not a causal effect."
                )
            })

if inc27 is not None:
    d = inc27[
        (inc27["population"].astype(str) == "Pooled") &
        (inc27["outcome"].astype(str) == "SOMATIC_SCORE")
    ].copy()

    if len(d):
        name_map = {
            "Add_A_to_G_plus_X": "Add A beyond G + X",
            "Add_G_to_A_plus_X": "Add G beyond A + X",
            "Add_AxG_to_A_plus_G_plus_X": "Add A×G beyond additive",
        }
        d["display"] = d["test"].map(name_map).fillna(d["test"])
        plt.figure(figsize=(9, 4.8))
        ax = plt.gca()
        vals = 100 * d["weighted_delta_R2"].to_numpy()
        ax.bar(d["display"], vals)
        ax.set_ylabel("Incremental weighted R² (percentage points)")
        ax.set_title("How much unique weighted outcome variance each term adds")
        ax.tick_params(axis="x", rotation=22)
        annotate_bars(ax, "{:.3f}")
        savefig("34_15_incremental_weighted_R2.png")

        d.to_csv(TAB / "34_incremental_information_tests.csv", index=False)

# ------------------------------------------------------------
# 8) CFA + MIMIC: PHENOTYPE STRUCTURE
# ------------------------------------------------------------
cfa_fit = safe_read_csv("20_cfa_fit.csv")
cfa_load = safe_read_csv("20_cfa_two_factor_loadings.csv")
cfa_inv = safe_read_csv("20_cfa_invariance.csv")
mimic = safe_read_csv("21_mimic_paths.csv")
mimic_fit = safe_read_csv("21_mimic_fit.csv")

if cfa_fit is not None:
    # CFI comparison, no hard-coded fit values.
    models = ["one_factor", "two_factor"]
    labels_model = ["One factor", "Somatic + cognitive-affective"]
    x = np.arange(len(models))
    plt.figure(figsize=(8.5, 4.8))
    ax = plt.gca()
    for cycle in sorted(cfa_fit["cycle"].astype(str).unique()):
        dd = cfa_fit[cfa_fit["cycle"].astype(str) == cycle].set_index("model").reindex(models)
        ax.plot(x, dd["cfi"].to_numpy(), marker="o", label=cycle)
        for xi, v in zip(x, dd["cfi"]):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_model)
    ax.set_ylabel("CFI")
    ax.set_ylim(max(0, cfa_fit["cfi"].min() - .04), 1.005)
    ax.set_title("Ordinal CFA: does PHQ-9 behave as one factor or two?")
    ax.legend()
    savefig("34_16_cfa_model_comparison.png")

if cfa_load is not None:
    som = cfa_load[cfa_load["lhs"].astype(str) == "Somatic"].copy()
    if len(som):
        item_order = ["DPQ030", "DPQ040", "DPQ050"]
        item_labels = ["Sleep", "Fatigue", "Appetite"]
        x = np.arange(3)
        plt.figure(figsize=(8.5, 4.8))
        ax = plt.gca()
        for cycle in sorted(som["cycle"].astype(str).unique()):
            dd = som[som["cycle"].astype(str) == cycle].set_index("rhs").reindex(item_order)
            ax.plot(x, dd["est.std"].to_numpy(), marker="o", label=cycle)
        ax.set_xticks(x)
        ax.set_xticklabels(item_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Standardized factor loading")
        ax.set_title("Validated somatic factor: sleep, fatigue and appetite")
        ax.legend()
        savefig("34_17_somatic_factor_loadings.png")

if mimic is not None:
    som = mimic[mimic["lhs"].astype(str) == "Somatic"].copy()
    if len(som):
        term_order = ["A_Z", "G_HBA1C_Z", "AG_Z"]
        term_labels = ["A", "G", "A×G"]
        xbase = np.arange(3)
        offsets = np.linspace(-0.10, 0.10, max(1, som["cycle"].nunique()))
        plt.figure(figsize=(8.5, 4.8))
        ax = plt.gca()
        for off, cycle in zip(offsets, sorted(som["cycle"].astype(str).unique())):
            dd = som[som["cycle"].astype(str) == cycle].set_index("rhs").reindex(term_order)
            xx = xbase + off
            ax.errorbar(
                xx, dd["est"].to_numpy(),
                yerr=[
                    dd["est"].to_numpy() - dd["ci.lower"].to_numpy(),
                    dd["ci.upper"].to_numpy() - dd["est"].to_numpy()
                ],
                fmt="o", capsize=4, label=cycle
            )
        ax.axhline(0, linewidth=0.9)
        ax.set_xticks(xbase)
        ax.set_xticklabels(term_labels)
        ax.set_ylabel("Latent somatic regression coefficient")
        ax.set_title("Weighted ordinal MIMIC: physiology → latent somatic factor")
        ax.legend()
        savefig("34_18_mimic_somatic_paths.png")

if mimic_fit is not None:
    mimic_fit.to_csv(TAB / "34_mimic_fit_indices.csv", index=False)

# ------------------------------------------------------------
# 9) FASTING GLUCOSE SENSITIVITY
# ------------------------------------------------------------
fast = safe_read_csv("18_fasting_glucose_sensitivity.csv")
if fast is not None:
    # Show only X3 and G term: apples-to-apples sensitivity of the glycemia measure.
    d = fast[
        (fast["model"].astype(str) == "X3_kidney_direct_effect") &
        (fast["term"].astype(str) == "G")
    ].copy()
    if len(d):
        disp = {
            "HbA1c_excess": "HbA1c-based G",
            "Fasting_glucose_excess": "Fasting-glucose G"
        }
        d["display"] = d["representation"].map(disp).fillna(d["representation"])
        y = np.arange(len(d))
        plt.figure(figsize=(8.5, 4.6))
        ax = plt.gca()
        for i, r in d.reset_index(drop=True).iterrows():
            ax.errorbar(
                r["beta"], i,
                xerr=[[r["beta"] - r["ci_low"]], [r["ci_high"] - r["beta"]]],
                fmt="o", capsize=4
            )
            ax.text(r["ci_high"] + 0.004, i,
                    f"β={r['beta']:.3f}, p={r['p']:.4f}", va="center", fontsize=9)
        ax.axvline(0, linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(d["display"])
        ax.set_xlabel("Survey-weighted coefficient (95% CI)")
        ax.set_title("G sensitivity: HbA1c versus fasting glucose, X3")
        savefig("34_19_fasting_glucose_sensitivity.png")

# ------------------------------------------------------------
# 10) INTERNAL PHYSIOLOGY: REDUCED A PCA + G PCA
# ------------------------------------------------------------
reduced_a = safe_read_csv("25_reduced_A_pca_sensitivity.csv")
if reduced_a is not None:
    reduced_a.to_csv(TAB / "34_reduced_A_pca.csv", index=False)
    x = np.arange(len(reduced_a))
    width = 0.36
    plt.figure(figsize=(8.8, 4.8))
    ax = plt.gca()
    ax.bar(x - width/2, 100 * reduced_a["variance_0506"], width=width, label="2005–06")
    ax.bar(x + width/2, 100 * reduced_a["variance_0708"], width=width, label="2007–08")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{int(v)}" for v in reduced_a["component"]])
    ax.set_ylabel("Explained standardized CBC variance (%)")
    ax.set_title("Reduced non-redundant hematology basis: stable multivariate structure")
    ax.legend()
    savefig("34_20_reduced_A_pca_variance.png")

pca_var = safe_read_csv("24_internal_burden_explained_variance.csv")
pca_align = safe_read_csv("24_internal_burden_cross_cycle_alignment.csv")
if pca_var is not None:
    gvar = pca_var[pca_var["block"].astype(str) == "G"].copy()
    if len(gvar):
        comps = sorted(gvar["component"].unique())
        x = np.arange(len(comps))
        width = 0.36
        plt.figure(figsize=(8.6, 4.8))
        ax = plt.gca()
        for shift, cycle in [(-width/2, "0506"), (width/2, "0708")]:
            dd = gvar[gvar["cycle"].astype(str) == cycle].set_index("component").reindex(comps)
            ax.bar(x + shift, 100 * dd["explained_variance_ratio"], width=width, label=cycle)
        ax.set_xticks(x)
        ax.set_xticklabels([f"PC{int(c)}" for c in comps])
        ax.set_ylabel("Explained standardized marker variance (%)")
        ax.set_title("Two-marker glycemic space: HbA1c + fasting glucose")
        ax.legend()
        savefig("34_21_G_pca_variance.png")

# ------------------------------------------------------------
# 11) TEMPORAL TRANSFER + MODERN HOLDOUT DIAGNOSTICS
# ------------------------------------------------------------
transfer_coef = safe_read_csv("32_transfer_design_based_coefficients.csv")
transfer_inc = safe_read_csv("32_transfer_incremental_tests.csv")
xladder = safe_read_csv("33_2123_x_ladder_coefficients.csv")
design33 = safe_read_csv("33_2123_design_diagnostics.csv")
dist33 = safe_read_csv("33_distribution_comparison.csv")

if transfer_coef is not None:
    d = transfer_coef[
        (transfer_coef["outcome"].astype(str) == "SOMATIC_SCORE") &
        (transfer_coef["model"].astype(str) == "Additive_A+G+X") &
        (transfer_coef["term"].isin(["A", "G_HBA1C"]))
    ].copy()
    if len(d):
        term_label = {"A": "A", "G_HBA1C": "G"}
        pops = ["2009-2018", "2021-2023"]
        terms = ["A", "G_HBA1C"]
        ypos = []
        labels = []
        row_num = 0

        plt.figure(figsize=(9.2, 5.3))
        ax = plt.gca()
        for pop in pops:
            for term in terms:
                rr = d[
                    (d["population"].astype(str) == pop) &
                    (d["term"].astype(str) == term)
                ]
                if not len(rr):
                    continue
                r = rr.iloc[0]
                y = row_num
                row_num += 1
                ypos.append(y)
                labels.append(f"{pop} · {term_label[term]}")
                if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
                    ax.errorbar(
                        r["beta"], y,
                        xerr=[[r["beta"] - r["ci_low"]], [r["ci_high"] - r["beta"]]],
                        fmt="o", capsize=4
                    )
                else:
                    ax.plot(r["beta"], y, marker="x", linestyle="none")
                    ax.text(r["beta"] + 0.01, y, "X3 CI/p not estimable", va="center", fontsize=9)
        ax.axvline(0, linewidth=0.9)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Frozen X3 somatic coefficient")
        ax.set_title("Frozen temporal transfer: what replicated and what did not")
        savefig("34_22_temporal_transfer_frozen_X3.png")

if xladder is not None:
    for term, file_stub, title in [
        ("A", "A", "2021–2023 A across the prespecified adjustment ladder"),
        ("G_HBA1C", "G", "2021–2023 G across the prespecified adjustment ladder"),
    ]:
        d = xladder[xladder["term"].astype(str) == term].copy()
        if len(d):
            order = ["X0", "X1", "X2", "X3"]
            d = d.set_index("model").reindex(order).reset_index()
            x = np.arange(len(order))
            plt.figure(figsize=(8.8, 4.8))
            ax = plt.gca()
            ax.plot(x, d["beta"], marker="o")
            ax.axhline(0, linewidth=0.9)

            # Plot valid design-based intervals where they exist.
            for i, r in d.iterrows():
                if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
                    ax.errorbar(
                        i, r["beta"],
                        yerr=[[r["beta"] - r["ci_low"]], [r["ci_high"] - r["beta"]]],
                        fmt="none", capsize=4
                    )
                ax.text(i, r["beta"] + 0.015, f"{r['beta']:.3f}", ha="center", fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(["X0\nage/sex/race", "X1\n+SES/edu/smoking", "X2\n+BMI", "X3\n+kidney"])
            ax.set_ylabel("Survey-weighted coefficient")
            ax.set_title(title)
            savefig(f"34_23_2123_{file_stub}_xladder.png" if file_stub == "A"
                    else f"34_24_2123_{file_stub}_xladder.png")

if design33 is not None:
    design33.to_csv(TAB / "34_2123_design_degrees_of_freedom.csv", index=False)
    d = design33.set_index("model").reindex(["X0", "X1", "X2", "X3"]).reset_index()
    plt.figure(figsize=(8.6, 4.8))
    ax = plt.gca()
    ax.bar(d["model"], d["residual_design_df"])
    ax.set_ylabel("Residual survey design df")
    ax.set_title("Why 2021–2023 X3 confidence intervals fail")
    annotate_bars(ax, "{:.0f}")
    savefig("34_25_2123_residual_design_df.png")

if dist33 is not None:
    dist33.to_csv(TAB / "34_temporal_distribution_comparison.csv", index=False)
    for var, label, outname in [
        ("A", "Mean A", "34_26_temporal_A_distribution_mean.png"),
        ("G_HBA1C", "Mean G", "34_27_temporal_G_distribution_mean.png"),
        ("SOMATIC_SCORE", "Mean somatic score", "34_28_temporal_somatic_mean.png"),
        ("PHQ9_TOTAL", "Mean PHQ-9 total", "34_29_temporal_phq9_mean.png"),
    ]:
        d = dist33[dist33["variable"].astype(str) == var].copy()
        if len(d):
            plt.figure(figsize=(7.6, 4.5))
            ax = plt.gca()
            ax.bar(d["period"], d["mean"])
            ax.set_ylabel(label)
            ax.set_title(f"{label}: 2009–2018 versus 2021–2023")
            annotate_bars(ax, "{:.3f}")
            savefig(outname)

# ------------------------------------------------------------
# 12) REFERENCE-PRESERVATION / FRESH-COPY SOURCE AUDIT
# ------------------------------------------------------------
audit_rows = []

if SCRIPTS.exists():
    for p in sorted(SCRIPTS.glob("*.py")):
        txt = p.read_text(encoding="utf-8", errors="ignore")

        reads_core = (
            "nhanes_core_working.parquet" in txt
            and "read_parquet" in txt
        )

        # Conservative write detector. It will flag obvious writes; it is not
        # a full static program proof, so the audit describes exactly that.
        explicit_core_write = bool(
            re.search(
                r"to_parquet\s*\([^)]*nhanes_core_working\.parquet",
                txt,
                flags=re.IGNORECASE | re.DOTALL
            )
        )
        working_path_alias = bool(
            re.search(
                r"\w+\s*=\s*[^#\n]*nhanes_core_working\.parquet",
                txt,
                flags=re.IGNORECASE
            )
        )
        write_via_alias = False
        if working_path_alias:
            aliases = re.findall(
                r"(\w+)\s*=\s*[^#\n]*nhanes_core_working\.parquet",
                txt,
                flags=re.IGNORECASE
            )
            for alias in aliases:
                if re.search(rf"to_parquet\s*\(\s*{re.escape(alias)}\b", txt):
                    write_via_alias = True

        audit_rows.append({
            "script": p.name,
            "reads_core_working": reads_core,
            "explicitly_writes_core_working": explicit_core_write or write_via_alias,
            "copy_calls": txt.count(".copy("),
            "read_parquet_calls": txt.count("read_parquet"),
            "to_parquet_calls": txt.count("to_parquet"),
            "merge_calls": txt.count(".merge("),
            "classification": (
                "original core producer / inspect manually"
                if p.name.startswith("02_")
                else
                "downstream reader/branch"
                if reads_core and not (explicit_core_write or write_via_alias)
                else
                "other/derived producer"
            )
        })

source_audit = pd.DataFrame(audit_rows)
source_audit.to_csv(TAB / "34_reference_preservation_source_audit.csv", index=False)

downstream_overwrites = source_audit[
    (~source_audit["script"].str.startswith("02_")) &
    (source_audit["explicitly_writes_core_working"] == True)
] if len(source_audit) else pd.DataFrame()

# Transformation ledger if already available.
ledger_path = AUDIT / "13_transformation_ledger.csv"
if ledger_path.exists():
    ledger = pd.read_csv(ledger_path)
    ledger.to_csv(TAB / "34_existing_transformation_ledger.csv", index=False)
else:
    ledger = None

# Reference branching diagram.
plt.figure(figsize=(12, 7))
ax = plt.gca()
ax.axis("off")

ax.text(0.08, 0.82, "RAW x\nNHANES XPT\nimmutable + SHA256",
        transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5"))
ax.text(0.36, 0.82, "Preserved working reference\nnhanes_core_working.parquet\nparticipant mapping retained",
        transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5"))
ax.annotate("", xy=(0.28, 0.82), xytext=(0.16, 0.82),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.5))

branches = [
    (0.67, 0.86, "Survey branch\nA/G + X0→X3"),
    (0.67, 0.68, "Phenotype branch\nICA → CFA → MIMIC/DIF"),
    (0.67, 0.50, "Sensitivity branch\nfasting glucose / clinical state"),
    (0.67, 0.32, "Physiology branch\nCBC / glycemic PCA"),
    (0.67, 0.14, "Transfer branch\nfrozen 2009–18 / 2021–23"),
]
for bx, by, txt in branches:
    ax.text(bx, by, txt, transform=ax.transAxes,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45"))
    ax.annotate("", xy=(bx - 0.12, by), xytext=(0.47, 0.82),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", linewidth=1.0))

ax.text(
    0.50, 0.02,
    "Branches create/reload fresh working DataFrames/copies. "
    "They are not intended as a serial chain where experiment B mutates experiment A's y.",
    transform=ax.transAxes, ha="center", va="bottom", fontsize=10
)
ax.set_title("Reference-preserving analysis architecture actually used")
savefig("34_30_reference_preserving_branch_architecture.png")

# ------------------------------------------------------------
# 13) STATISTICAL GLOSSARY / WHY EACH METHOD EXISTS
# ------------------------------------------------------------
glossary = pd.DataFrame([
    ["β (beta)", "Adjusted change in the modeled outcome per one-unit increase in the predictor, conditional on other terms in that model.",
     "A β=0.33 means roughly +0.33 somatic-score points per 1 g/dL larger Hb deficit, conditional on G and X, if using that exact additive model.",
     "Association, not causation."],
    ["95% CI", "Range produced by the survey-design inference procedure that represents uncertainty around β.",
     "If the interval excludes 0, the two-sided test usually gives p<0.05.",
     "A narrow CI is precision, not biological importance."],
    ["p-value", "Probability, under the fitted null model and survey design, of a test statistic at least this extreme if the tested coefficient were zero.",
     "Used here for design-based Wald inference.",
     "Not the probability that the hypothesis is true."],
    ["Weighted R²", "Descriptive fraction of weighted outcome variance accounted for by a fitted model.",
     "Used to compare model ladders.",
     "Not the primary NHANES inferential statistic."],
    ["ΔR²", "Increase in weighted R² after adding a term/block to a nested model.",
     "Shows unique incremental descriptive information beyond existing terms.",
     "0.0026 = 0.26 percentage points of weighted outcome variance, not 26%."],
    ["F / Wald test", "Design-based test of whether an added term or block contributes after the rest of the model.",
     "Used by survey::regTermTest in script 27.",
     "Depends on survey design degrees of freedom."],
    ["Survey design df", "Effective inferential degrees of freedom driven mainly by PSU/stratum structure, not raw participant N.",
     "2021–23 has many participants but X3 exhausts residual design df.",
     "When residual df=0, t-based CI/p are not estimable."],
    ["CFI", "Comparative fit index for the latent-factor model; closer to 1 means better relative fit.",
     "Used to compare one-factor versus two-factor PHQ structure.",
     "Good fit does not prove the factor model is uniquely true."],
    ["RMSEA", "Approximate model misfit per degree of freedom; smaller is better.",
     "Used with CFI/SRMR rather than alone.",
     "Not a causal metric."],
    ["SRMR", "Average standardized discrepancy between observed and model-implied relationships; smaller is better.",
     "Another latent-model fit diagnostic.",
     "Should be interpreted jointly with other fit measures."],
    ["Factor loading", "Standardized strength linking a PHQ item to a latent factor.",
     "Sleep/fatigue/appetite load on the somatic factor.",
     "A loading does not mean an item is caused by physiology."],
    ["PCA explained variance", "Share of standardized marker variation captured by a principal component.",
     "Used to ask whether one-number A/G proxies discard stable measurement structure.",
     "Principal components are mathematical axes, not automatically biological mechanisms."],
    ["Cross-cycle loading similarity", "Similarity of component loading vectors after sign/order alignment across independently fit cycles.",
     "Near 1 means the measurement pattern reproduced strongly.",
     "Stability does not by itself establish clinical meaning."],
    ["DIF + FDR", "Tests whether a specific PHQ item has a direct physiology association beyond the latent factor; FDR controls multiplicity.",
     "No FDR-stable single-item effect was found.",
     "Useful negative check against one item driving the result."],
], columns=["statistic_or_method", "plain_meaning", "why_we_used_it", "do_not_overclaim"])
glossary.to_csv(TAB / "34_statistical_glossary.csv", index=False)

why = pd.DataFrame([
    ["Sex-specific A threshold", "Raw low Hb has different clinical meaning by sex; naive -z(Hb) erased the A signal.", "Representation choice matters; threshold is domain-informed, not learned from outcome."],
    ["G threshold at HbA1c 5.7", "Creates a simple excess-above-reference glycemic burden.", "Working proxy only; not all glycemic physiology."],
    ["A×G term", "Tests departure from a purely additive A + G model.", "Secondary because it is small and not stable across all checks."],
    ["X0→X3 ladder", "Separates baseline confounding adjustment from BMI and kidney sensitivity/direct-effect assumptions.", "Avoids blindly adjusting for everything."],
    ["NHANES survey weights + strata + PSU", "NHANES is not a simple random sample.", "Required for population-relevant standard errors/inference."],
    ["Fasting glucose sensitivity", "HbA1c can be affected by RBC turnover/anemia.", "Checks whether G is only an HbA1c measurement artifact."],
    ["Ordinal CFA", "PHQ items are ordinal and total score can mix symptom dimensions.", "Tests whether somatic vs cognitive-affective structure is supported."],
    ["MIMIC/SEM", "Links A/G to validated latent symptom factors.", "Checks whether physiology aligns more with somatic than affective phenotype."],
    ["DIF", "Checks if one PHQ item is directly driving a latent association.", "Negative-result integrity check."],
    ["Internal PCA", "Tests whether scalar A/G proxies throw away stable marker structure.", "Motivates deeper A_i / G_j representation work without calling PCs biological mechanisms."],
    ["Frozen temporal transfer", "Tests the already-defined model on later NHANES without retuning.", "Prevents post-hoc rescue and measures transfer honestly."],
    ["2021–23 X ladder diagnostic", "Determines whether transfer failure is substantive or only due to collapsed survey df.", "Diagnostic only; did not alter frozen definitions."],
], columns=["choice", "reason", "interpretation_rule"])
why.to_csv(TAB / "34_why_each_method.csv", index=False)

# ------------------------------------------------------------
# 14) KEY RESULTS TABLE
# ------------------------------------------------------------
if transfer_coef is not None:
    d = transfer_coef[
        (transfer_coef["outcome"].astype(str) == "SOMATIC_SCORE") &
        (transfer_coef["model"].astype(str) == "Additive_A+G+X")
    ]
    for _, r in d.iterrows():
        key_result_rows.append({
            "section": "Frozen temporal transfer",
            "population": r["population"],
            "outcome": "Somatic score",
            "term": r["term"],
            "estimate": r["beta"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "p": r["p"],
            "interpretation": (
                "Frozen X3 temporal-validation coefficient. "
                "2021-2023 CI/p are not valid when residual design df is exhausted."
            )
        })

if fast is not None:
    d = fast[
        (fast["representation"].astype(str) == "Fasting_glucose_excess") &
        (fast["model"].astype(str) == "X3_kidney_direct_effect") &
        (fast["term"].astype(str) == "G")
    ]
    for _, r in d.iterrows():
        key_result_rows.append({
            "section": "Fasting-glucose sensitivity",
            "population": "2005-2008 fasting subset",
            "outcome": "Somatic score",
            "term": "G fasting",
            "estimate": r["beta"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "p": r["p"],
            "interpretation": "Alternative glycemia measurement; reduces concern that G is solely an HbA1c/RBC-turnover artifact."
        })

key_results = pd.DataFrame(key_result_rows)
key_results.to_csv(TAB / "34_key_results.csv", index=False)

# ------------------------------------------------------------
# 15) FINAL HASH / READ-ONLY CHECK
# ------------------------------------------------------------
core_hash_after = sha256(core_working_path) if core_working_path.exists() else None
hash_unchanged = (core_hash_before == core_hash_after) if core_hash_before else None

preservation_summary = pd.DataFrame([
    {
        "check": "Core working reference hash unchanged during script 34",
        "status": "PASS" if hash_unchanged else ("NOT AVAILABLE" if hash_unchanged is None else "FAIL"),
        "detail": core_hash_before
    },
    {
        "check": "Existing transformation ledger found",
        "status": "PASS" if ledger is not None else "MISSING",
        "detail": str(ledger_path)
    },
    {
        "check": "Downstream explicit writes to nhanes_core_working.parquet detected by static scan",
        "status": "PASS" if len(downstream_overwrites) == 0 else "REVIEW",
        "detail": "None detected" if len(downstream_overwrites) == 0 else ", ".join(downstream_overwrites["script"].tolist())
    },
    {
        "check": "Final y -> y_hat reconstruction for final physiological representation",
        "status": "NOT YET DONE",
        "detail": "Do not claim final reconstruction success in the presentation."
    },
])
preservation_summary.to_csv(TAB / "34_reference_preservation_summary.csv", index=False)

# ------------------------------------------------------------
# 16) HUMAN-READABLE GUIDE
# ------------------------------------------------------------
def extract_discovery(term):
    r = find_row(
        coef26,
        population="Pooled",
        outcome="SOMATIC_SCORE",
        model="Additive_A+G+X",
        term=term
    )
    return r

def extract_transfer(pop, term):
    r = find_row(
        transfer_coef,
        population=pop,
        outcome="SOMATIC_SCORE",
        model="Additive_A+G+X",
        term=term
    )
    return r

a_disc = extract_discovery("A")
g_disc = extract_discovery("G_HBA1C")
a_0918 = extract_transfer("2009-2018", "A")
g_0918 = extract_transfer("2009-2018", "G_HBA1C")
a_x0 = find_row(xladder, model="X0", term="A")
g_x0 = find_row(xladder, model="X0", term="G_HBA1C")

mfit = mimic_fit.iloc[0] if mimic_fit is not None and len(mimic_fit) else None

guide = f"""# Presentation Evidence Guide — Script 34

## 1. Scope lock

This evidence pack supports a **preliminary feasibility / proposal presentation**.

The defensible question is:

> Can hematological and glycemic physiological information be represented and statistically separated with respect to depressive symptom dimensions, and which parts remain stable under temporal transfer?

Do **not** present this as:
- causal proof that anemia or dysglycemia causes depression;
- a completed biological mechanism;
- a finished decomposition/reconstruction framework;
- proof that the model generalizes to all populations;
- proof that COVID caused the 2021–2023 change.

---

## 2. People and data flow

The script rebuilt the discovery core directly from the raw 2005–06 and 2007–08 XPT files.

{flow_df.to_markdown(index=False)}

The important distinction is:

- **9,743** (or the value above if your raw files reproduce differently) = complete first-pass A/G/PHQ discovery core.
- **X3 survey cohort** = stricter complete-case cohort used for the fully adjusted complex-survey models.
- Fasting, clinical-state, CFA/MIMIC and internal-PCA analyses have their own required-variable subsets.

Never present all analyses as if they used an identical N.

---

## 3. What x and y actually were

### Preserved x

For participant i, the preserved source state includes the measured inputs:

`x_i = [Hb, CBC markers, HbA1c, PHQ item responses, demographics/covariates, ...]`

The raw XPT files remain authoritative.

### First operational transformation

`A = max(Hb_threshold(sex) - Hb, 0)`

- male threshold = 13 g/dL
- female threshold = 12 g/dL

`G = max(HbA1c - 5.7, 0)`

`AG = A * G`

The first operational working representation can therefore be thought of as:

`y_i = [A, G, AG, preserved PHQ item state / derived phenotype, X]`

This is **not yet the final mathematical y** promised by the thesis.

The current y summary is in `Tables/34_operational_y_summary.csv`.

### Why the domain transformation mattered

The earlier naive `A = -z(Hb)` representation did not carry a useful A association.
The clinically informed deficit representation did. That is evidence that **representation choice can erase or expose statistical structure**, not proof that the thresholded scalar A is the final biological representation.

---

## 4. Did we repeatedly modify the same y?

### What the code architecture supports

The project audit shows the intended architecture is **reference preserving and branched**:

`raw XPT -> preserved working reference -> fresh analysis branches`

The main downstream scripts reload `nhanes_core_working.parquet` and/or a defined derivative cohort, then build new DataFrames with `.copy()`, merges, residual matrices, standardized matrices, latent-factor inputs, or sensitivity subsets.

Script 34 also performs a static source scan and writes:
- `Tables/34_reference_preservation_source_audit.csv`
- `Tables/34_reference_preservation_summary.csv`

Core hash unchanged during this evidence build: **{hash_unchanged}**

Downstream explicit core-overwrite scan: **{"none detected" if len(downstream_overwrites) == 0 else "REVIEW: " + ", ".join(downstream_overwrites["script"].tolist())}**

### Precise wording for the presentation

> We preserved the raw and core working references. Individual analyses branched from those references or explicitly defined derivative cohorts; they were not intended as a destructive serial chain in which each experiment overwrote the previous experiment's representation.

Do **not** say “every script rebuilt from raw.” Many correctly reload a preserved processed reference instead.

### Remaining reconstruction gap

We have **not** yet completed the final:

`final y -> decomposed components -> y_hat`

with a final prespecified reconstruction loss `L_y`.

Earlier exploratory algebraic/PCA/ICA reconstruction work exists, but the thesis-level final reconstruction test is still open.

That is a future-work box, not a result box.

---

## 5. Primary statistical result

### Discovery: 2005–2008, strict additive somatic model

A:
- beta = {fmt_num(a_disc["beta"] if a_disc is not None else np.nan)}
- 95% CI = [{fmt_num(a_disc["ci_low"] if a_disc is not None else np.nan)}, {fmt_num(a_disc["ci_high"] if a_disc is not None else np.nan)}]
- p = {fmt_num(a_disc["p"] if a_disc is not None else np.nan, 5)}

G:
- beta = {fmt_num(g_disc["beta"] if g_disc is not None else np.nan)}
- 95% CI = [{fmt_num(g_disc["ci_low"] if g_disc is not None else np.nan)}, {fmt_num(g_disc["ci_high"] if g_disc is not None else np.nan)}]
- p = {fmt_num(g_disc["p"] if g_disc is not None else np.nan, 5)}

Interpretation:

- A shows clear adjusted association in the strict additive pooled survey model.
- G is positive but is weaker/borderline in this exact strict additive discovery specification.
- G is supported by other prespecified/convergent analyses, especially fasting glucose and latent MIMIC.
- A×G is secondary because it is small and representation/replication sensitive.

---

## 6. What beta means in plain language

For the strict additive model:

`Somatic = beta0 + beta_A*A + beta_G*G + beta_X*X + error`

### A beta

A is measured in **g/dL below the sex-specific Hb threshold**.

So, if beta_A were 0.33:

> among otherwise model-comparable participants, a 1 g/dL larger Hb deficit is associated with about 0.33 higher points on the 0–9 somatic score, conditional on G and the modeled X variables.

### G beta

G is measured in **HbA1c percentage points above 5.7**.

So, if beta_G were 0.10:

> a 1 percentage-point larger HbA1c excess is associated with about 0.10 higher somatic-score points, conditional on A and X.

These are **conditional associations**, not individual-level causal predictions.

---

## 7. Why the survey machinery was necessary

NHANES is a complex probability sample.

Therefore the main regressions used:
- examination weights;
- sampling strata;
- primary sampling units (PSUs);
- cycle-aware pooling.

Raw N is not the same thing as inferential degrees of freedom.

This becomes crucial in 2021–2023: the modern holdout still has thousands of participants, but the fully adjusted X3 model consumes the residual survey-design degrees of freedom. That is why the X3 beta can be estimated while its usual t-based CI/p becomes non-estimable.

See:
- `Figures/34_25_2123_residual_design_df.png`
- `Tables/34_2123_design_degrees_of_freedom.csv`

---

## 8. Why X0 -> X3 instead of one giant adjustment set

The ladder was designed to make assumptions visible:

- **X0:** age, sex, race
- **X1:** + socioeconomic status, education, smoking
- **X2:** + BMI
- **X3:** + kidney function

BMI and kidney function can sit on or near plausible physiological pathways, so forcing them into the only model would hide an important modeling assumption.

The staged ladder shows whether A/G are robust or whether an association disappears only after adding a pathway-sensitive variable.

---

## 9. PHQ phenotype: why CFA / MIMIC

A PHQ-9 total score collapses multiple symptoms.

We therefore tested whether the item structure supports:
- a **somatic** factor: sleep, fatigue, appetite
- a **cognitive-affective** factor: remaining affective/cognitive items

Ordinal WLSMV CFA was used because PHQ items are ordered categories rather than continuous Gaussian measurements.

MIMIC then asks whether A/G relate to those validated latent factors.

Current weighted MIMIC fit:
- CFI = {fmt_num(mfit["cfi"] if mfit is not None and "cfi" in mfit else np.nan, 3)}
- RMSEA = {fmt_num(mfit["rmsea"] if mfit is not None and "rmsea" in mfit else np.nan, 3)}
- SRMR = {fmt_num(mfit["srmr"] if mfit is not None and "srmr" in mfit else np.nan, 3)}

Use fit indices as model diagnostics, not as proof that the factor model is “true.”

---

## 10. Why fasting glucose mattered

Anemia / altered RBC turnover can distort measured HbA1c.

Therefore the glycemia result had to survive an alternative measurement that does not rely on the same erythrocyte mechanism.

The fasting-glucose sensitivity is shown in:
`Figures/34_19_fasting_glucose_sensitivity.png`

If the fasting G coefficient remains positive with a valid interval excluding zero, it reduces the likelihood that the entire glycemia association is merely an HbA1c/RBC-turnover artifact.

It does not eliminate all measurement bias or confounding.

---

## 11. Why internal A/G decomposition mattered

The scalar A and G were always intended as first-pass representations.

Reduced hematology PCA asks:

> Is the CBC state really describable by one scalar deficit, or is stable multivariate information being discarded?

The reduced CBC basis uses:
- Hb
- RBC count
- MCV
- RDW

This avoids the most obvious algebraic redundancy in the larger CBC set.

The PCA axes are **stable measurement axes**, not yet named biological mechanisms.

For G, the current PCA uses only HbA1c and fasting glucose. If PC1 captures about 91% of their standardized variance, the correct claim is:

> the two-marker glycemic measurement space is dominated by one common axis.

Not:

> glycemic physiology is one-dimensional.

---

## 12. Temporal validation: what happened

2009–2018 frozen X3:

A:
- beta = {fmt_num(a_0918["beta"] if a_0918 is not None else np.nan)}
- CI = [{fmt_num(a_0918["ci_low"] if a_0918 is not None else np.nan)}, {fmt_num(a_0918["ci_high"] if a_0918 is not None else np.nan)}]
- p = {fmt_num(a_0918["p"] if a_0918 is not None else np.nan, 5)}

G:
- beta = {fmt_num(g_0918["beta"] if g_0918 is not None else np.nan)}
- CI = [{fmt_num(g_0918["ci_low"] if g_0918 is not None else np.nan)}, {fmt_num(g_0918["ci_high"] if g_0918 is not None else np.nan)}]
- p = {fmt_num(g_0918["p"] if g_0918 is not None else np.nan, 5)}

This is strong temporal replication of the simple A/G association pattern.

2021–2023 diagnostic X0:

A:
- beta = {fmt_num(a_x0["beta"] if a_x0 is not None else np.nan)}
- p = {fmt_num(a_x0["p"] if a_x0 is not None else np.nan, 5)}

G:
- beta = {fmt_num(g_x0["beta"] if g_x0 is not None else np.nan)}
- p = {fmt_num(g_x0["p"] if g_x0 is not None else np.nan, 5)}

A is already near zero at the simplest valid adjustment stage, while G remains positive.

Therefore the modern A failure cannot be dismissed as *only* an X3 degree-of-freedom problem.

Correct statement:

> The simple Hb-deficit A representation transferred through 2009–2018 but did not reproduce in the 2021–2023 holdout; G was more temporally stable. The reason for the A non-transfer is not yet known.

---

## 13. What the important numbers do and do not prove

### Strong enough to show

1. The discovery dataset is large enough for a serious feasibility analysis.
2. A has a clear adjusted somatic association in pooled discovery.
3. G contributes smaller/convergent information; the strict discovery additive p-value alone is borderline, but fasting and latent analyses support the direction.
4. The somatic PHQ structure is reproducible using ordinal CFA.
5. A/G are not highly redundant proxies in the discovery model.
6. 2009–2018 validates both simple A and G prospectively in time, under frozen definitions.
7. 2021–2023 exposes a real boundary: simple A does not transfer, while G remains more stable.
8. The physiological measurement space contains structure beyond the first scalar proxies.

### Not strong enough to claim

1. Causality.
2. Clinical diagnostic utility.
3. A final biological mechanism.
4. Stable A×G interaction.
5. A final A_i/G_j decomposition.
6. Final y reconstruction.
7. India transfer.
8. COVID as explanation for 2021–2023.

---

## 14. Recommended presentation figure sequence

Use the pack as a menu; do not put every figure in the main deck.

**Main deck**
1. `34_11_x_to_y_operational_pipeline.png`
2. `34_01_discovery_cohort_flow.png`
3. `34_08_hb_to_A_transform.png` + `34_09_hba1c_to_G_transform.png` (can be placed side-by-side later)
4. `34_13_ag_group_mean_phq9.png`
5. `34_14_discovery_primary_effects.png`
6. `34_16_cfa_model_comparison.png`
7. `34_17_somatic_factor_loadings.png`
8. `34_18_mimic_somatic_paths.png`
9. `34_19_fasting_glucose_sensitivity.png`
10. `34_20_reduced_A_pca_variance.png`
11. `34_22_temporal_transfer_frozen_X3.png`
12. `34_23_2123_A_xladder.png`
13. `34_30_reference_preserving_branch_architecture.png`
14. final scope / next-step diagram later in the slide-building stage.

**Appendix**
- detailed distributions;
- G PCA;
- 2021–23 design df;
- temporal raw distribution comparisons;
- reference-preservation source audit;
- full statistic glossary.

---

## 15. One-sentence presentation claim

> In NHANES, hematological and glycemic burden show separable but unequal associations with a reproducible somatic depressive phenotype; the pattern transfers through 2009–2018, while the simple hematological representation fails in the 2021–2023 holdout, motivating a more principled and information-preserving physiological decomposition.

That is a proposal-level result. It is neither “we solved depression” nor “we found nothing.”
"""

guide_path = OUT / "34_PRESENTATION_EVIDENCE_GUIDE.md"
guide_path.write_text(guide, encoding="utf-8")

# ------------------------------------------------------------
# 17) MANIFEST
# ------------------------------------------------------------
manifest = []
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        manifest.append({
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv(OUT / "34_generation_manifest.csv", index=False)

print("PASS  Discovery core rebuilt directly from raw XPT.")
print(f"PASS  Complete A/G/P core: n={len(complete_core):,}")
if survey is not None:
    print(f"PASS  Final discovery survey X3 cohort: n={len(survey):,}")
print(f"PASS  Core working reference hash unchanged during script: {hash_unchanged}")
if len(downstream_overwrites) == 0:
    print("PASS  Static scan found no explicit downstream overwrite of nhanes_core_working.parquet.")
else:
    print("REVIEW  Static scan found possible downstream core writes:")
    for s in downstream_overwrites["script"]:
        print("       ", s)
print("NOTE  Final thesis-level y -> y_hat reconstruction has NOT been completed; guide marks this explicitly.")
print()
print(f"SAVED  {OUT}")
print(f"FIGURES {len(list(FIG.glob('*.png')))}")
print(f"TABLES  {len(list(TAB.glob('*.csv')))}")
print("NEXT  Review 34_PRESENTATION_EVIDENCE_GUIDE.md and the generated figures before building slides.")
