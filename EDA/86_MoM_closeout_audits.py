#!/usr/bin/env python
# =============================================================================
# 86_MoM_closeout_audits.py
#
# Final pre-presentation MoM closeout:
#
# A) EDA audit requested in MoM
#    - PHQ-9 score distribution and severity counts
#    - high-burden counts (>=10, >=15, >=20)
#    - basic sample composition by period
#
# B) Explicit component x time interaction audit
#    - formalizes whether Z -> phenotype coefficients change across periods
#    - 2005-08 is baseline
#    - interaction block tests: component x {2009-18, 2021-23}
#
# This is an association / temporal-stability audit, NOT causal.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    RESULTS / "73_frozen_phenotype_map_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
]
INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), None)
if INPUT is None:
    raise FileNotFoundError(
        "Need EDA/Results/73_frozen_phenotype_map_input.csv. Run Script 73 first."
    )

ZCOLS = [
    "SP_SHARED1", "SP_SHARED2",
    "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]

OUTCOMES = ["SOMATIC_SCORE", "COGAFF_SUM", "PHQ9_TOTAL"]
PERIODS = ["2005-2008", "2009-2018", "2021-2023"]

DISC = {"0506": "D", "0708": "E"}
TEMP = {
    "0910": "F", "1112": "G", "1314": "H",
    "1516": "I", "1718": "J", "2123": "L",
}
ALL_CYCLES = list(DISC) + list(TEMP)

DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"


def xpt_path(base: Path, stem: str) -> Path | None:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    return matches[0] if len(matches) == 1 else None


def base_for(cycle: str) -> Path:
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def suffix_for(cycle: str) -> str:
    return (DISC if cycle in DISC else TEMP)[cycle]


def wmean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    return float(np.sum(w * x) / np.sum(w))


def wsd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    mu = wmean(x, w)
    return float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))


def weighted_prop(mask, w):
    mask = np.asarray(mask, float)
    w = np.asarray(w, float)
    return float(np.sum(w * mask) / np.sum(w))


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


# -----------------------------------------------------------------------------
# Load frozen analysis scaffold.
# -----------------------------------------------------------------------------
d = pd.read_csv(INPUT)

required = (
    ["SEQN", "CYCLE", "PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG"]
    + ZCOLS + OUTCOMES
)
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing required columns from Script-73 input: {missing}")

d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce")
d["CYCLE"] = (
    d["CYCLE"].astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(4)
)

for c in ["SURVEY_WT"] + ZCOLS + OUTCOMES:
    d[c] = pd.to_numeric(d[c], errors="coerce")

# -----------------------------------------------------------------------------
# Restore core demographics from raw DEMO files where available.
# -----------------------------------------------------------------------------
demo_frames = []

for cycle in ALL_CYCLES:
    suffix = suffix_for(cycle)
    p = xpt_path(base_for(cycle), f"DEMO_{suffix}")
    if p is None:
        continue

    q = pd.read_sas(p, format="xport")
    keep = ["SEQN"]
    for c in ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3"]:
        if c in q.columns:
            keep.append(c)

    q = q[keep].copy()
    q["SEQN"] = pd.to_numeric(q["SEQN"], errors="coerce")
    q["CYCLE_RESTORED"] = cycle
    demo_frames.append(q)

if demo_frames:
    demo = pd.concat(demo_frames, ignore_index=True)

    if demo["SEQN"].duplicated().any():
        demo = demo.drop_duplicates("SEQN", keep="first")

    d = d.merge(demo, on="SEQN", how="left", validate="one_to_one")

# Unified race field.
def harmonize_race(row):
    r3 = row.get("RIDRETH3", np.nan)
    r1 = row.get("RIDRETH1", np.nan)

    if pd.notna(r3):
        code = int(r3)
        return {
            1: "Mexican American",
            2: "Other Hispanic",
            3: "Non-Hispanic White",
            4: "Non-Hispanic Black",
            6: "Non-Hispanic Asian",
            7: "Other/Multi",
        }.get(code, "Other/Multi")

    if pd.notna(r1):
        code = int(r1)
        return {
            1: "Mexican American",
            2: "Other Hispanic",
            3: "Non-Hispanic White",
            4: "Non-Hispanic Black",
            5: "Other/Multi",
        }.get(code, "Other/Multi")

    return np.nan


if "RIDRETH1" in d.columns or "RIDRETH3" in d.columns:
    d["RACE_BROAD"] = d.apply(harmonize_race, axis=1)

if "RIAGENDR" in d.columns:
    d["SEX_LABEL"] = d["RIAGENDR"].map({1: "Male", 2: "Female"})

# =============================================================================
# PART A — EDA
# =============================================================================

eda_base = d[
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d["SURVEY_WT"].notna()
    & (d["SURVEY_WT"] > 0)
].copy()

# Cohort/count audit.
count_rows = []

for period in PERIODS:
    q = eda_base[eda_base["PERIOD"].eq(period)]
    count_rows.append({
        "period": period,
        "eligible_n": len(q),
        "PHQ_nonmissing_n": int(q["PHQ9_TOTAL"].notna().sum()),
        "all_Z_nonmissing_n": int(q[ZCOLS].notna().all(axis=1).sum()),
        "PHQ_and_all_Z_n": int(
            q[["PHQ9_TOTAL"] + ZCOLS].notna().all(axis=1).sum()
        ),
    })

counts = pd.DataFrame(count_rows)
counts.to_csv(RESULTS / "86_eda_cohort_counts.csv", index=False)

# PHQ score distribution.
phq_rows = []
exact_rows = []
severity_rows = []

def phq_category(x):
    if pd.isna(x):
        return np.nan
    if x <= 4:
        return "0-4 minimal"
    if x <= 9:
        return "5-9 mild"
    if x <= 14:
        return "10-14 moderate"
    if x <= 19:
        return "15-19 moderately severe"
    return "20-27 severe"


eda_base["PHQ_CATEGORY"] = eda_base["PHQ9_TOTAL"].apply(phq_category)

for period in PERIODS:
    q = eda_base[
        eda_base["PERIOD"].eq(period)
        & eda_base["PHQ9_TOTAL"].notna()
    ].copy()

    if q.empty:
        continue

    y = q["PHQ9_TOTAL"].to_numpy(float)
    w = q["SURVEY_WT"].to_numpy(float)

    phq_rows.append({
        "period": period,
        "n": len(q),
        "mean": float(np.mean(y)),
        "weighted_mean": wmean(y, w),
        "sd": float(np.std(y, ddof=1)),
        "weighted_sd": wsd(y, w),
        "median": float(np.median(y)),
        "q25": float(np.quantile(y, 0.25)),
        "q75": float(np.quantile(y, 0.75)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "n_PHQ_ge10": int(np.sum(y >= 10)),
        "weighted_prop_PHQ_ge10": weighted_prop(y >= 10, w),
        "n_PHQ_ge15": int(np.sum(y >= 15)),
        "weighted_prop_PHQ_ge15": weighted_prop(y >= 15, w),
        "n_PHQ_ge20": int(np.sum(y >= 20)),
        "weighted_prop_PHQ_ge20": weighted_prop(y >= 20, w),
    })

    for score in range(0, 28):
        mask = y == score
        exact_rows.append({
            "period": period,
            "PHQ9_score": score,
            "n": int(mask.sum()),
            "weighted_prop": weighted_prop(mask, w),
        })

    for cat in [
        "0-4 minimal",
        "5-9 mild",
        "10-14 moderate",
        "15-19 moderately severe",
        "20-27 severe",
    ]:
        mask = q["PHQ_CATEGORY"].eq(cat).to_numpy()
        severity_rows.append({
            "period": period,
            "category": cat,
            "n": int(mask.sum()),
            "weighted_prop": weighted_prop(mask, w),
        })

phq_summary = pd.DataFrame(phq_rows)
phq_exact = pd.DataFrame(exact_rows)
phq_severity = pd.DataFrame(severity_rows)

phq_summary.to_csv(RESULTS / "86_phq_summary.csv", index=False)
phq_exact.to_csv(RESULTS / "86_phq_exact_score_distribution.csv", index=False)
phq_severity.to_csv(RESULTS / "86_phq_severity_distribution.csv", index=False)

# Demographic composition.
demo_rows = []

for period in PERIODS:
    q = eda_base[eda_base["PERIOD"].eq(period)].copy()
    if q.empty:
        continue

    w = q["SURVEY_WT"].to_numpy(float)

    if "RIDAGEYR" in q.columns:
        ok = q["RIDAGEYR"].notna().to_numpy()
        if ok.any():
            age = q.loc[ok, "RIDAGEYR"].to_numpy(float)
            ww = q.loc[ok, "SURVEY_WT"].to_numpy(float)
            demo_rows.append({
                "period": period,
                "variable": "Age",
                "level": "mean",
                "n": int(ok.sum()),
                "weighted_value": wmean(age, ww),
            })

    if "SEX_LABEL" in q.columns:
        for level in ["Male", "Female"]:
            ok = q["SEX_LABEL"].notna()
            qq = q[ok]
            if len(qq):
                mask = qq["SEX_LABEL"].eq(level).to_numpy()
                ww = qq["SURVEY_WT"].to_numpy(float)
                demo_rows.append({
                    "period": period,
                    "variable": "Sex",
                    "level": level,
                    "n": int(mask.sum()),
                    "weighted_value": weighted_prop(mask, ww),
                })

    if "RACE_BROAD" in q.columns:
        qq = q[q["RACE_BROAD"].notna()]
        if len(qq):
            ww = qq["SURVEY_WT"].to_numpy(float)
            for level in sorted(qq["RACE_BROAD"].unique()):
                mask = qq["RACE_BROAD"].eq(level).to_numpy()
                demo_rows.append({
                    "period": period,
                    "variable": "Race/ethnicity",
                    "level": level,
                    "n": int(mask.sum()),
                    "weighted_value": weighted_prop(mask, ww),
                })

demo_summary = pd.DataFrame(demo_rows)
demo_summary.to_csv(RESULTS / "86_demographic_composition.csv", index=False)

# =============================================================================
# PART B — COMPONENT x TIME AUDIT
# =============================================================================

interaction_rows = []
coefficient_rows = []

for outcome in OUTCOMES:
    q = d[
        d["ELIGIBLE_ADULT_NONPREG"].eq(1)
        & d["PERIOD"].isin(PERIODS)
        & d[ZCOLS + [outcome, "SURVEY_WT"]].notna().all(axis=1)
        & (d["SURVEY_WT"] > 0)
    ].copy()

    # Time dummies: 2005-08 baseline.
    q["P09"] = q["PERIOD"].eq("2009-2018").astype(float)
    q["P21"] = q["PERIOD"].eq("2021-2023").astype(float)

    design_cols = list(ZCOLS) + ["P09", "P21"]

    for z in ZCOLS:
        c09 = f"{z}__x__P09"
        c21 = f"{z}__x__P21"
        q[c09] = q[z] * q["P09"]
        q[c21] = q[z] * q["P21"]
        design_cols.extend([c09, c21])

    # Basic demographic adjustment only where available.
    if "RIDAGEYR" in q.columns:
        q["AGE_ADJ"] = pd.to_numeric(q["RIDAGEYR"], errors="coerce")
        if q["AGE_ADJ"].notna().all():
            design_cols.append("AGE_ADJ")

    if "RIAGENDR" in q.columns:
        q["FEMALE_ADJ"] = (pd.to_numeric(q["RIAGENDR"], errors="coerce") == 2).astype(float)
        design_cols.append("FEMALE_ADJ")

    X = sm.add_constant(q[design_cols].astype(float), has_constant="add")
    y = q[outcome].astype(float)
    w = q["SURVEY_WT"].astype(float)

    fit = sm.WLS(y, X, weights=w).fit(cov_type="HC3")

    names = list(X.columns)
    pvals_block = []

    for z in ZCOLS:
        c09 = f"{z}__x__P09"
        c21 = f"{z}__x__P21"

        beta0 = float(fit.params[z])
        b09 = float(fit.params[c09])
        b21 = float(fit.params[c21])

        coefficient_rows.extend([
            {
                "outcome": outcome,
                "component": z,
                "period": "2005-2008",
                "coefficient": beta0,
            },
            {
                "outcome": outcome,
                "component": z,
                "period": "2009-2018",
                "coefficient": beta0 + b09,
            },
            {
                "outcome": outcome,
                "component": z,
                "period": "2021-2023",
                "coefficient": beta0 + b21,
            },
        ])

        R = np.zeros((2, len(names)))
        R[0, names.index(c09)] = 1.0
        R[1, names.index(c21)] = 1.0

        wt = fit.wald_test(R, scalar=True)
        block_p = float(np.asarray(wt.pvalue).squeeze())
        pvals_block.append(block_p)

        interaction_rows.append({
            "outcome": outcome,
            "component": z,
            "beta_2005_08": beta0,
            "delta_2009_18_vs_2005_08": b09,
            "p_interaction_2009_18": float(fit.pvalues[c09]),
            "beta_2009_18": beta0 + b09,
            "delta_2021_23_vs_2005_08": b21,
            "p_interaction_2021_23": float(fit.pvalues[c21]),
            "beta_2021_23": beta0 + b21,
            "block_time_interaction_p": block_p,
            "n": len(q),
        })

    # FDR within outcome across 7 component-level block tests.
    idx = [
        i for i, row in enumerate(interaction_rows)
        if row["outcome"] == outcome
    ]
    qs = bh_fdr([interaction_rows[i]["block_time_interaction_p"] for i in idx])
    for i, qval in zip(idx, qs):
        interaction_rows[i]["block_time_interaction_q_BH"] = float(qval)

interaction = pd.DataFrame(interaction_rows)
coefficients = pd.DataFrame(coefficient_rows)

interaction.to_csv(
    RESULTS / "86_component_time_interaction_audit.csv",
    index=False,
)
coefficients.to_csv(
    RESULTS / "86_component_period_coefficients.csv",
    index=False,
)

manifest = {
    "script": "86_MoM_closeout_audits.py",
    "purpose": "final pre-presentation MoM EDA + component x time closeout",
    "EDA": {
        "PHQ_thresholds": [10, 15, 20],
        "PHQ_categories": [
            "0-4 minimal",
            "5-9 mild",
            "10-14 moderate",
            "15-19 moderately severe",
            "20-27 severe",
        ],
        "demographics": "age, sex, broad race/ethnicity where raw DEMO available",
    },
    "time_interaction": {
        "baseline": "2005-2008",
        "comparisons": ["2009-2018", "2021-2023"],
        "model": "survey-weighted WLS with HC3 robust covariance",
        "tests": "joint Wald test of both time interactions per component",
        "multiple_testing": "Benjamini-Hochberg within phenotype",
        "causal": False,
    },
}
(RESULTS / "86_MoM_closeout_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

# =============================================================================
# Concise console output for meeting prep.
# =============================================================================

print("=" * 118)
print("SCRIPT 86 — FINAL MoM CLOSEOUT AUDITS")
print("=" * 118)

print("\nA) COHORT / PHQ EDA")
print(counts.to_string(index=False))

print("\nPHQ SUMMARY")
for _, r in phq_summary.iterrows():
    print(
        f"  {r['period']}: n={int(r['n'])} | "
        f"weighted mean={r['weighted_mean']:.2f} | "
        f">=10: {int(r['n_PHQ_ge10'])} ({100*r['weighted_prop_PHQ_ge10']:.1f}%) | "
        f">=15: {int(r['n_PHQ_ge15'])} ({100*r['weighted_prop_PHQ_ge15']:.1f}%) | "
        f">=20: {int(r['n_PHQ_ge20'])} ({100*r['weighted_prop_PHQ_ge20']:.1f}%)"
    )

if not demo_summary.empty:
    print("\nBASIC DEMOGRAPHIC COMPOSITION SAVED:")
    print("  EDA/Results/86_demographic_composition.csv")
else:
    print("\nWARNING: raw demographic files were not resolved; demographic table is empty.")

print("\nB) COMPONENT x TIME AUDIT")
for outcome in OUTCOMES:
    print(f"\n[{outcome}]")
    qq = interaction[interaction["outcome"].eq(outcome)].sort_values(
        "block_time_interaction_p"
    )
    for _, r in qq.iterrows():
        flag = "*" if r["block_time_interaction_q_BH"] < 0.05 else ""
        print(
            f"  {r['component']:14s} | "
            f"beta: {r['beta_2005_08']:+.4f} -> "
            f"{r['beta_2009_18']:+.4f} -> "
            f"{r['beta_2021_23']:+.4f} | "
            f"time block p={r['block_time_interaction_p']:.4g} | "
            f"BH q={r['block_time_interaction_q_BH']:.4g} {flag}"
        )

print("\nFILES WRITTEN")
print("  86_eda_cohort_counts.csv")
print("  86_phq_summary.csv")
print("  86_phq_exact_score_distribution.csv")
print("  86_phq_severity_distribution.csv")
print("  86_demographic_composition.csv")
print("  86_component_time_interaction_audit.csv")
print("  86_component_period_coefficients.csv")
print("  86_MoM_closeout_manifest.json")

print("\nInterpretation rule:")
print("  BH q < .05 => that component shows evidence that its phenotype association changes with time.")
print("  This is a temporal association/stability test, not a causal interaction.")
print("=" * 118)
