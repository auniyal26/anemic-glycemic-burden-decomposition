#!/usr/bin/env python
# =============================================================================
# 58_cohort_phq_eda.py
#
# Purpose
#   Proper descriptive EDA of the locked NHANES A × G × depression cohorts.
#
# Design
#   - Uses Results/47_domain_corrected_AG_input.csv ONLY as the authoritative
#     cohort/domain scaffold.
#   - Re-reads raw NHANES XPT files to restore PHQ items and the raw physiology
#     variables intentionally omitted from the Script 47 inference input.
#   - Writes ONLY under EDA/Results and EDA/Figures.
#   - Does not refit PCA, does not rerun inferential models, and does not modify
#     locked Results/, Data/Processed/, or Audit/ outputs.
#
# Run from project root:
#   python EDA/58_cohort_phq_eda.py
# =============================================================================

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 0. Paths and project structure copied from the locked Script 47 pipeline
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"
LOCKED_INPUT = ROOT / "Results" / "47_domain_corrected_AG_input.csv"

OUT = ROOT / "EDA" / "Results"
FIG = ROOT / "EDA" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DISC = {"0506": "D", "0708": "E"}
TEMP = {
    "0910": "F",
    "1112": "G",
    "1314": "H",
    "1516": "I",
    "1718": "J",
    "2123": "L",
}
ALL_CYCLES = list(DISC) + list(TEMP)

PHQ = [f"DPQ0{i}0" for i in range(1, 10)]
PHQ_LABELS = {
    "DPQ010": "Little interest / pleasure",
    "DPQ020": "Feeling down / depressed",
    "DPQ030": "Sleep problems",
    "DPQ040": "Tired / little energy",
    "DPQ050": "Appetite problems",
    "DPQ060": "Feeling bad about self",
    "DPQ070": "Concentration problems",
    "DPQ080": "Psychomotor change",
    "DPQ090": "Death / self-harm thoughts",
}
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]

A_RAW = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G_RAW_REQUIRED = ["LBXGH", "LBXGLU"]
INSULIN_CANDIDATES = ["LBXIN", "LBXIN1", "LBXINSI", "LBDINSI"]

PRIMARY_DOMAIN = "DOMAIN_SOMATIC_SCORE_X3"
FULL_PHQ_DOMAIN = "DOMAIN_PHQ9_TOTAL_X3"


# -----------------------------------------------------------------------------
# 1. Utilities
# -----------------------------------------------------------------------------

def xpt_path(base: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")


def base_for(cycle: str) -> Path:
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def suffix_for(cycle: str) -> str:
    return (DISC if cycle in DISC else TEMP)[cycle]


def clean_phq_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PHQ:
        if c not in out.columns:
            out[c] = np.nan
        x = pd.to_numeric(out[c], errors="coerce")
        x.loc[x.abs() < 1e-10] = 0
        x.loc[~x.isin([0, 1, 2, 3])] = np.nan
        out[c] = x
    out["PHQ9_TOTAL_RAW"] = out[PHQ].sum(axis=1, min_count=9)
    out["SOMATIC_SCORE_RAW"] = out[SOMATIC].sum(axis=1, min_count=len(SOMATIC))
    out["COGAFF_SUM_RAW"] = out["PHQ9_TOTAL_RAW"] - out["SOMATIC_SCORE_RAW"]
    return out


def weighted_mean(x, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return np.sum(x[ok] * w[ok]) / np.sum(w[ok])


def weighted_sd(x, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.nan
    x, w = x[ok], w[ok]
    mu = np.sum(x * w) / np.sum(w)
    return np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w))


def weighted_quantile(x, w, probs=(0.25, 0.5, 0.75)):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return [np.nan] * len(probs)
    x, w = x[ok], w[ok]
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cdf = np.cumsum(w) / np.sum(w)
    return [float(np.interp(p, cdf, x)) for p in probs]


def weighted_corr(x, y, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    sw = w.sum()
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    cov = np.sum(w * (x - mx) * (y - my)) / sw
    vx = np.sum(w * (x - mx) ** 2) / sw
    vy = np.sum(w * (y - my) ** 2) / sw
    if vx <= 0 or vy <= 0:
        return np.nan
    return cov / np.sqrt(vx * vy)


def numeric_summary(df, variables, cohort_name):
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        x = pd.to_numeric(df[v], errors="coerce")
        valid = x.dropna()
        q1, med, q3 = (
            valid.quantile(0.25) if len(valid) else np.nan,
            valid.median() if len(valid) else np.nan,
            valid.quantile(0.75) if len(valid) else np.nan,
        )
        wq1, wmed, wq3 = weighted_quantile(x, df["SURVEY_WT"])
        rows.append({
            "cohort": cohort_name,
            "variable": v,
            "n": int(x.notna().sum()),
            "missing_n": int(x.isna().sum()),
            "missing_pct": float(100 * x.isna().mean()),
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "sd": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            "median": float(med) if pd.notna(med) else np.nan,
            "q1": float(q1) if pd.notna(q1) else np.nan,
            "q3": float(q3) if pd.notna(q3) else np.nan,
            "iqr": float(q3 - q1) if pd.notna(q1) and pd.notna(q3) else np.nan,
            "min": float(valid.min()) if len(valid) else np.nan,
            "max": float(valid.max()) if len(valid) else np.nan,
            "weighted_mean": weighted_mean(x, df["SURVEY_WT"]),
            "weighted_sd": weighted_sd(x, df["SURVEY_WT"]),
            "weighted_q1": wq1,
            "weighted_median": wmed,
            "weighted_q3": wq3,
        })
    return pd.DataFrame(rows)


def weighted_frequency(df, variable, cohort_name):
    if variable not in df.columns:
        return pd.DataFrame()
    d = df[[variable, "SURVEY_WT"]].copy()
    d = d[d[variable].notna() & d["SURVEY_WT"].notna() & (d["SURVEY_WT"] > 0)]
    if d.empty:
        return pd.DataFrame()
    out = (
        d.groupby(variable, dropna=False)
        .agg(n=(variable, "size"), weighted_n=("SURVEY_WT", "sum"))
        .reset_index()
    )
    out["pct"] = 100 * out["n"] / out["n"].sum()
    out["weighted_pct"] = 100 * out["weighted_n"] / out["weighted_n"].sum()
    out.insert(0, "cohort", cohort_name)
    return out


def period_primary(df, period):
    return df[(df["PERIOD"] == period) & (df[PRIMARY_DOMAIN] == 1)].copy()


def period_full_phq(df, period):
    return df[(df["PERIOD"] == period) & (df[FULL_PHQ_DOMAIN] == 1)].copy()


# -----------------------------------------------------------------------------
# 2. Locked Script 47 scaffold
# -----------------------------------------------------------------------------

if not LOCKED_INPUT.exists():
    raise FileNotFoundError(
        f"Missing locked scaffold: {LOCKED_INPUT}\nRun Script 47 first."
    )

locked = pd.read_csv(LOCKED_INPUT)

required_locked = [
    "SEQN", "CYCLE", "PERIOD", "SURVEY_WT",
    PRIMARY_DOMAIN, FULL_PHQ_DOMAIN,
    "PHQ9_TOTAL", "SOMATIC_SCORE", "COGAFF_SUM",
    "A", "G_HBA1C",
]
missing_locked = [c for c in required_locked if c not in locked.columns]
if missing_locked:
    raise ValueError(f"Script 47 input is missing required columns: {missing_locked}")

locked["SEQN"] = pd.to_numeric(locked["SEQN"], errors="coerce")
locked["CYCLE"] = locked["CYCLE"].astype(str).str.zfill(4)


# -----------------------------------------------------------------------------
# 3. Restore raw PHQ + chosen A/G variables from authoritative XPT files
# -----------------------------------------------------------------------------

raw_frames = []
file_audit = []

for cycle in ALL_CYCLES:
    s = suffix_for(cycle)
    base = base_for(cycle)

    dpq_path = xpt_path(base, f"DPQ_{s}")
    cbc_path = xpt_path(base, f"CBC_{s}")
    ghb_path = xpt_path(base, f"GHB_{s}")
    glu_path = xpt_path(base, f"GLU_{s}")

    dpq = pd.read_sas(dpq_path, format="xport")
    cbc = pd.read_sas(cbc_path, format="xport")
    ghb = pd.read_sas(ghb_path, format="xport")
    glu = pd.read_sas(glu_path, format="xport")

    # PHQ
    have_phq = [c for c in PHQ if c in dpq.columns]
    if len(have_phq) != 9:
        raise ValueError(f"{cycle}: expected all 9 PHQ items; found {have_phq}")
    q = clean_phq_items(dpq[["SEQN"] + PHQ].copy())

    # Haematology
    have_a = [c for c in A_RAW if c in cbc.columns]
    missing_a = [c for c in A_RAW if c not in cbc.columns]
    if missing_a:
        warnings.warn(f"{cycle}: CBC missing {missing_a}")
    a = cbc[["SEQN"] + have_a].copy()

    # Glycaemia
    g_cols = ["SEQN"]
    for c in G_RAW_REQUIRED:
        if c in ghb.columns and c not in g_cols:
            g_cols.append(c)
        if c in glu.columns and c not in g_cols:
            g_cols.append(c)

    # Insulin moved to a separate INS component in later NHANES cycles.
    insulin_source = "GLU"
    insulin_df = glu
    insulin_col = next((c for c in INSULIN_CANDIDATES if c in glu.columns), None)

    if insulin_col is None:
        try:
            ins_path = xpt_path(base, f"INS_{s}")
            insulin_df = pd.read_sas(ins_path, format="xport")
            insulin_col = next(
                (c for c in INSULIN_CANDIDATES if c in insulin_df.columns),
                None,
            )
            insulin_source = str(ins_path.relative_to(ROOT))
        except FileNotFoundError:
            insulin_source = ""

    # LBXGH is in GHB; fasting glucose is in GLU; insulin may be in GLU or INS.
    g1 = ghb[[c for c in ["SEQN", "LBXGH"] if c in ghb.columns]].copy()
    g2 = glu[[c for c in ["SEQN", "LBXGLU"] if c in glu.columns]].copy()
    g = g1.merge(g2, on="SEQN", how="outer", validate="one_to_one")

    if insulin_col is not None:
        gi = insulin_df[["SEQN", insulin_col]].copy()
        g = g.merge(gi, on="SEQN", how="outer", validate="one_to_one")

    d = q.merge(a, on="SEQN", how="outer", validate="one_to_one")
    d = d.merge(g, on="SEQN", how="outer", validate="one_to_one")
    d["CYCLE"] = cycle

    raw_frames.append(d)

    file_audit.append({
        "cycle": cycle,
        "DPQ_file": str(dpq_path.relative_to(ROOT)),
        "CBC_file": str(cbc_path.relative_to(ROOT)),
        "GHB_file": str(ghb_path.relative_to(ROOT)),
        "GLU_file": str(glu_path.relative_to(ROOT)),
        "PHQ_items_found": len(have_phq),
        "A_variables_found": ", ".join(have_a),
        "insulin_column": insulin_col or "",
        "insulin_source": insulin_source,
    })

raw = pd.concat(raw_frames, ignore_index=True)
raw["SEQN"] = pd.to_numeric(raw["SEQN"], errors="coerce")
raw["CYCLE"] = raw["CYCLE"].astype(str).str.zfill(4)

pd.DataFrame(file_audit).to_csv(OUT / "58_raw_file_variable_audit.csv", index=False)

eda = locked.merge(
    raw,
    on=["SEQN", "CYCLE"],
    how="left",
    validate="one_to_one",
)


# -----------------------------------------------------------------------------
# 4. Integrity checks against Script 47
# -----------------------------------------------------------------------------

checks = []

for raw_col, locked_col in [
    ("PHQ9_TOTAL_RAW", "PHQ9_TOTAL"),
    ("SOMATIC_SCORE_RAW", "SOMATIC_SCORE"),
    ("COGAFF_SUM_RAW", "COGAFF_SUM"),
]:
    ok = eda[raw_col].notna() & eda[locked_col].notna()
    diff = (eda.loc[ok, raw_col] - eda.loc[ok, locked_col]).abs()
    checks.append({
        "check": f"{raw_col}_vs_{locked_col}",
        "n_compared": int(ok.sum()),
        "max_abs_difference": float(diff.max()) if len(diff) else np.nan,
        "exact_match_n": int((diff < 1e-10).sum()) if len(diff) else 0,
        "exact_match_pct": float(100 * (diff < 1e-10).mean()) if len(diff) else np.nan,
    })

pd.DataFrame(checks).to_csv(OUT / "58_locked_input_integrity_check.csv", index=False)


# -----------------------------------------------------------------------------
# 5. Cohort definitions and flow
# -----------------------------------------------------------------------------

cohort_rows = []
for period in ["2005-2008", "2009-2018", "2021-2023"]:
    p = eda[eda["PERIOD"] == period]
    cohort_rows.append({
        "period": period,
        "design_rows": len(p),
        "eligible_adult_nonpreg_n": int(p["ELIGIBLE_ADULT_NONPREG"].fillna(0).eq(1).sum())
            if "ELIGIBLE_ADULT_NONPREG" in p.columns else np.nan,
        "primary_somatic_X3_n": int(p[PRIMARY_DOMAIN].fillna(0).eq(1).sum()),
        "full_PHQ9_X3_n": int(p[FULL_PHQ_DOMAIN].fillna(0).eq(1).sum()),
        "PHQ9_complete_n": int(p["PHQ9_TOTAL_RAW"].notna().sum()),
    })

pd.DataFrame(cohort_rows).to_csv(OUT / "58_cohort_flow.csv", index=False)

cohorts = {
    "discovery_primary_somatic_X3": period_primary(eda, "2005-2008"),
    "replication_primary_somatic_X3": period_primary(eda, "2009-2018"),
    "modern_primary_somatic_X3": period_primary(eda, "2021-2023"),
    "discovery_full_PHQ9_X3": period_full_phq(eda, "2005-2008"),
    "replication_full_PHQ9_X3": period_full_phq(eda, "2009-2018"),
    "modern_full_PHQ9_X3": period_full_phq(eda, "2021-2023"),
}


# -----------------------------------------------------------------------------
# 6. PHQ total distribution
# -----------------------------------------------------------------------------

phq_summary_rows = []
phq_score_rows = []
severity_rows = []

def severity_label(x):
    if pd.isna(x):
        return np.nan
    if x <= 4:
        return "0-4 Minimal"
    if x <= 9:
        return "5-9 Mild"
    if x <= 14:
        return "10-14 Moderate"
    if x <= 19:
        return "15-19 Moderately severe"
    return "20-27 Severe"

for name, d in cohorts.items():
    # Only use full-PHQ cohorts for total-score distribution.
    if "full_PHQ9" not in name:
        continue

    x = pd.to_numeric(d["PHQ9_TOTAL_RAW"], errors="coerce")
    valid = x.notna()
    dd = d.loc[valid].copy()
    x = dd["PHQ9_TOTAL_RAW"]

    if dd.empty:
        continue

    q1, med, q3 = x.quantile([0.25, 0.50, 0.75]).tolist()
    wq1, wmed, wq3 = weighted_quantile(x, dd["SURVEY_WT"])

    phq_summary_rows.append({
        "cohort": name,
        "n": len(dd),
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": x.min(),
        "max": x.max(),
        "zero_n": int((x == 0).sum()),
        "zero_pct": 100 * (x == 0).mean(),
        "phq10plus_n": int((x >= 10).sum()),
        "phq10plus_pct": 100 * (x >= 10).mean(),
        "weighted_mean": weighted_mean(x, dd["SURVEY_WT"]),
        "weighted_sd": weighted_sd(x, dd["SURVEY_WT"]),
        "weighted_q1": wq1,
        "weighted_median": wmed,
        "weighted_q3": wq3,
    })

    freq = weighted_frequency(dd, "PHQ9_TOTAL_RAW", name)
    if not freq.empty:
        freq = freq.rename(columns={"PHQ9_TOTAL_RAW": "PHQ9_score"})
        phq_score_rows.append(freq)

    dd["PHQ_severity"] = dd["PHQ9_TOTAL_RAW"].map(severity_label)
    sev = weighted_frequency(dd, "PHQ_severity", name)
    if not sev.empty:
        severity_rows.append(sev)

pd.DataFrame(phq_summary_rows).to_csv(OUT / "58_phq9_total_summary.csv", index=False)
if phq_score_rows:
    pd.concat(phq_score_rows, ignore_index=True).to_csv(
        OUT / "58_phq9_score_distribution.csv", index=False
    )
if severity_rows:
    pd.concat(severity_rows, ignore_index=True).to_csv(
        OUT / "58_phq9_severity_distribution.csv", index=False
    )


# -----------------------------------------------------------------------------
# 7. Item-level PHQ distribution + item 9
# -----------------------------------------------------------------------------

item_rows = []
item9_rows = []

for name, d in cohorts.items():
    if "full_PHQ9" not in name:
        continue

    for item in PHQ:
        tmp = d[[item, "SURVEY_WT"]].copy()
        tmp = tmp[tmp[item].notna()]
        if tmp.empty:
            continue

        f = weighted_frequency(tmp, item, name)
        f = f.rename(columns={item: "score"})
        f.insert(2, "item", item)
        f.insert(3, "item_label", PHQ_LABELS[item])
        item_rows.append(f)

    q9 = pd.to_numeric(d["DPQ090"], errors="coerce")
    valid = q9.notna()
    dd = d.loc[valid].copy()
    if not dd.empty:
        positive = (dd["DPQ090"] > 0).astype(int)
        item9_rows.append({
            "cohort": name,
            "valid_n": len(dd),
            "positive_n": int(positive.sum()),
            "positive_pct": 100 * positive.mean(),
            "weighted_positive_pct":
                100 * np.sum(dd["SURVEY_WT"] * positive) / np.sum(dd["SURVEY_WT"]),
            "score_2_or_3_n": int((dd["DPQ090"] >= 2).sum()),
            "score_2_or_3_pct": 100 * (dd["DPQ090"] >= 2).mean(),
        })

if item_rows:
    pd.concat(item_rows, ignore_index=True).to_csv(
        OUT / "58_phq9_item_distributions.csv", index=False
    )
pd.DataFrame(item9_rows).to_csv(
    OUT / "58_phq9_item9_summary.csv", index=False
)


# -----------------------------------------------------------------------------
# 8. Demographic EDA in the exact primary somatic cohorts
# -----------------------------------------------------------------------------

NUMERIC_DEMOGRAPHICS = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "EGFR_2021"]
CATEGORICAL_DEMOGRAPHICS = ["RIAGENDR", "RACE", "EDUC3", "SMOKING3"]

demo_num = []
demo_cat = []

for name, d in cohorts.items():
    if "primary_somatic_X3" not in name:
        continue

    z = numeric_summary(d, NUMERIC_DEMOGRAPHICS, name)
    if not z.empty:
        demo_num.append(z)

    for v in CATEGORICAL_DEMOGRAPHICS:
        f = weighted_frequency(d, v, name)
        if not f.empty:
            demo_cat.append(f.assign(variable=v))

if demo_num:
    pd.concat(demo_num, ignore_index=True).to_csv(
        OUT / "58_demographic_numeric_summary.csv", index=False
    )
if demo_cat:
    pd.concat(demo_cat, ignore_index=True).to_csv(
        OUT / "58_demographic_categorical_summary.csv", index=False
    )


# -----------------------------------------------------------------------------
# 9. Raw physiology EDA in the exact primary somatic cohorts
# -----------------------------------------------------------------------------

insulin_cols = [c for c in INSULIN_CANDIDATES if c in eda.columns]
insulin_col = insulin_cols[0] if insulin_cols else None
PHYS = [c for c in A_RAW + G_RAW_REQUIRED + ([insulin_col] if insulin_col else [])
        if c in eda.columns]

phys_rows = []
for name, d in cohorts.items():
    if "primary_somatic_X3" not in name:
        continue
    z = numeric_summary(d, PHYS + ["A", "G_HBA1C"], name)
    if not z.empty:
        phys_rows.append(z)

if phys_rows:
    pd.concat(phys_rows, ignore_index=True).to_csv(
        OUT / "58_physiology_summary.csv", index=False
    )


# -----------------------------------------------------------------------------
# 10. A ↔ G cross-block correlation audit
# -----------------------------------------------------------------------------

corr_rows = []

A_FOR_CORR = [c for c in A_RAW if c in eda.columns]
G_FOR_CORR = [c for c in G_RAW_REQUIRED + ([insulin_col] if insulin_col else [])
              if c in eda.columns]

for name, d in cohorts.items():
    if "primary_somatic_X3" not in name:
        continue

    for a in A_FOR_CORR:
        for g in G_FOR_CORR:
            pair = d[[a, g, "SURVEY_WT"]].copy()
            pair[a] = pd.to_numeric(pair[a], errors="coerce")
            pair[g] = pd.to_numeric(pair[g], errors="coerce")
            pair = pair.dropna()
            if len(pair) < 3:
                continue

            pearson = pair[a].corr(pair[g], method="pearson")
            spearman = pair[a].corr(pair[g], method="spearman")
            wcorr = weighted_corr(pair[a], pair[g], pair["SURVEY_WT"])

            corr_rows.append({
                "cohort": name,
                "A_variable": a,
                "G_variable": g,
                "n_complete": len(pair),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "weighted_pearson_r": wcorr,
            })

pd.DataFrame(corr_rows).to_csv(
    OUT / "58_AG_crossblock_correlations.csv", index=False
)


# -----------------------------------------------------------------------------
# 11. Missingness audit
# -----------------------------------------------------------------------------

missing_rows = []
AUDIT_VARS = (
    PHQ + ["PHQ9_TOTAL_RAW", "SOMATIC_SCORE_RAW", "COGAFF_SUM_RAW"]
    + PHYS + ["A", "G_HBA1C"]
    + NUMERIC_DEMOGRAPHICS + CATEGORICAL_DEMOGRAPHICS
)

for name, d in cohorts.items():
    if "primary_somatic_X3" not in name and "full_PHQ9_X3" not in name:
        continue
    for v in dict.fromkeys(AUDIT_VARS):
        if v not in d.columns:
            continue
        missing_rows.append({
            "cohort": name,
            "variable": v,
            "n": len(d),
            "missing_n": int(d[v].isna().sum()),
            "missing_pct": float(100 * d[v].isna().mean()),
        })

pd.DataFrame(missing_rows).to_csv(
    OUT / "58_missingness_summary.csv", index=False
)


# -----------------------------------------------------------------------------
# 12. Figures
# -----------------------------------------------------------------------------

# Main discovery PHQ histogram
d = cohorts["discovery_full_PHQ9_X3"]
x = pd.to_numeric(d["PHQ9_TOTAL_RAW"], errors="coerce").dropna()

if len(x):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(-0.5, 28.5, 1)
    ax.hist(x, bins=bins)
    ax.set_xlabel("PHQ-9 total score")
    ax.set_ylabel("Participants")
    ax.set_title("Discovery full-PHQ X3 cohort: PHQ-9 distribution")
    fig.tight_layout()
    fig.savefig(FIG / "58_discovery_phq9_histogram.png", dpi=300)
    plt.close(fig)

# Discovery PHQ item distributions
d = cohorts["discovery_full_PHQ9_X3"]
long = (
    d[PHQ]
    .melt(var_name="item", value_name="score")
    .dropna()
)
if not long.empty:
    counts = (
        long.groupby(["item", "score"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts["pct"] = counts.groupby("item")["n"].transform(lambda s: 100 * s / s.sum())

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey=True)
    for ax, item in zip(axes.flat, PHQ):
        q = counts[counts["item"] == item]
        ax.bar(q["score"].astype(str), q["pct"])
        ax.set_title(f"{item}: {PHQ_LABELS[item]}", fontsize=9)
        ax.set_xlabel("Response")
        ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(FIG / "58_discovery_phq9_items.png", dpi=300)
    plt.close(fig)

# Main physiology histograms
d = cohorts["discovery_primary_somatic_X3"]
if PHYS:
    ncols = 3
    nrows = math.ceil(len(PHYS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, v in zip(axes, PHYS):
        x = pd.to_numeric(d[v], errors="coerce").dropna()
        ax.hist(x, bins=30)
        ax.set_title(v)
        ax.set_ylabel("Participants")
    for ax in axes[len(PHYS):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "58_discovery_physiology_histograms.png", dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 13. Compact manifest
# -----------------------------------------------------------------------------

manifest = {
    "script": "58_cohort_phq_eda.py",
    "locked_scaffold": str(LOCKED_INPUT.relative_to(ROOT)),
    "primary_domain": PRIMARY_DOMAIN,
    "full_phq_domain": FULL_PHQ_DOMAIN,
    "discovery_primary_n": len(cohorts["discovery_primary_somatic_X3"]),
    "discovery_full_phq_n": len(cohorts["discovery_full_PHQ9_X3"]),
    "replication_primary_n": len(cohorts["replication_primary_somatic_X3"]),
    "replication_full_phq_n": len(cohorts["replication_full_PHQ9_X3"]),
    "A_raw_variables": A_FOR_CORR,
    "G_raw_variables": G_FOR_CORR,
    "insulin_column_detected": insulin_col,
    "locked_outputs_modified": False,
}

(OUT / "58_eda_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)


print("PASS  Script 47 cohort scaffold loaded.")
print("PASS  Raw PHQ items and chosen physiology restored from XPT files.")
print(f"PASS  Discovery primary somatic X3 n = {manifest['discovery_primary_n']}")
print(f"PASS  Discovery full-PHQ X3 n = {manifest['discovery_full_phq_n']}")
print(f"PASS  Insulin column detected = {insulin_col}")
print("PASS  EDA outputs written only under EDA/Results and EDA/Figures.")
