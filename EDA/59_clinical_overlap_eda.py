#!/usr/bin/env python
# =============================================================================
# 59_clinical_overlap_eda.py
#
# Three-domain descriptive overlap analysis:
#   A = anaemia
#   D = HbA1c-defined diabetes-range glycaemia
#   P = PHQ-9 >= 10 depressive symptom burden
#
# Also reports dysglycaemia (HbA1c >= 5.7%) separately.
#
# Definitions
#   Anaemia:
#       Same sex-specific Hb threshold already used by the locked pipeline:
#       men Hb < 13 g/dL; women Hb < 12 g/dL.
#       Operationally: locked A > 0.
#
#   Diabetes-range glycaemia:
#       HbA1c >= 6.5%.
#       This is a biochemical threshold, NOT a clinical diabetes diagnosis.
#
#   Dysglycaemia:
#       HbA1c >= 5.7%.
#
#   Depressive symptom burden:
#       PHQ-9 >= 10.
#       This is a screening/severity threshold, NOT a clinical diagnosis.
#
# Cohort
#   Uses DOMAIN_PHQ9_TOTAL_X3 from Script 47 so all overlap estimates are on
#   the same locked complete analytic domain used for full-PHQ analyses.
#
# Outputs
#   EDA/Results/59_*.csv
#   EDA/Figures/59_*.png
#
# Does not modify locked Results/ or Data/Processed/.
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "Results" / "47_domain_corrected_AG_input.csv"
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"

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

DOMAIN = "DOMAIN_PHQ9_TOTAL_X3"

PERIOD_ORDER = ["2005-2008", "2009-2018", "2021-2023"]

COMBO_ORDER = [
    "None",
    "Anaemia only",
    "Diabetes-range only",
    "PHQ-9 >= 10 only",
    "Anaemia + diabetes-range",
    "Anaemia + PHQ-9 >= 10",
    "Diabetes-range + PHQ-9 >= 10",
    "All three",
]


def base_for(cycle):
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def suffix_for(cycle):
    return (DISC if cycle in DISC else TEMP)[cycle]


def xpt_path(base: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")


def weighted_pct(mask, weights):
    mask = pd.Series(mask).fillna(False).astype(bool).to_numpy()
    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(float)
    ok = np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return 100.0 * np.sum(w[ok] * mask[ok]) / np.sum(w[ok])


def weighted_n(mask, weights):
    mask = pd.Series(mask).fillna(False).astype(bool).to_numpy()
    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(float)
    ok = np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(w[ok] * mask[ok]))


def classify_combo(row):
    a = bool(row["ANAEMIA"])
    d = bool(row["DIABETES_RANGE"])
    p = bool(row["PHQ10"])

    if not a and not d and not p:
        return "None"
    if a and not d and not p:
        return "Anaemia only"
    if not a and d and not p:
        return "Diabetes-range only"
    if not a and not d and p:
        return "PHQ-9 >= 10 only"
    if a and d and not p:
        return "Anaemia + diabetes-range"
    if a and not d and p:
        return "Anaemia + PHQ-9 >= 10"
    if not a and d and p:
        return "Diabetes-range + PHQ-9 >= 10"
    return "All three"


# -----------------------------------------------------------------------------
# 1. Locked analytic scaffold
# -----------------------------------------------------------------------------

if not LOCKED.exists():
    raise FileNotFoundError(f"Missing {LOCKED}")

d = pd.read_csv(LOCKED)

required = [
    "SEQN", "CYCLE", "PERIOD", "SURVEY_WT",
    DOMAIN, "PHQ9_TOTAL", "A"
]
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Locked input missing: {missing}")

d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce")
d["CYCLE"] = d["CYCLE"].astype(str).str.zfill(4)

# -----------------------------------------------------------------------------
# 2. Restore raw HbA1c (LBXGH) from GHB files
# -----------------------------------------------------------------------------

ghb_frames = []

for cycle in ALL_CYCLES:
    base = base_for(cycle)
    s = suffix_for(cycle)
    p = xpt_path(base, f"GHB_{s}")

    g = pd.read_sas(p, format="xport")
    if "LBXGH" not in g.columns:
        raise ValueError(f"{cycle}: LBXGH not found in {p}")

    g = g[["SEQN", "LBXGH"]].copy()
    g["CYCLE"] = cycle
    ghb_frames.append(g)

ghb = pd.concat(ghb_frames, ignore_index=True)
ghb["SEQN"] = pd.to_numeric(ghb["SEQN"], errors="coerce")
ghb["CYCLE"] = ghb["CYCLE"].astype(str).str.zfill(4)

d = d.merge(ghb, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

# Exact full-PHQ X3 analytic domain
d = d[d[DOMAIN].eq(1)].copy()

# -----------------------------------------------------------------------------
# 3. Operational clinical indicators
# -----------------------------------------------------------------------------

d["ANAEMIA"] = pd.to_numeric(d["A"], errors="coerce") > 0
d["DIABETES_RANGE"] = pd.to_numeric(d["LBXGH"], errors="coerce") >= 6.5
d["DYSGLYCAEMIA"] = pd.to_numeric(d["LBXGH"], errors="coerce") >= 5.7
d["PHQ10"] = pd.to_numeric(d["PHQ9_TOTAL"], errors="coerce") >= 10

d["COMBINATION"] = d.apply(classify_combo, axis=1)

# -----------------------------------------------------------------------------
# 4. Broad prevalence and overlap table
# -----------------------------------------------------------------------------

definitions = [
    ("Anaemia", d["ANAEMIA"]),
    ("HbA1c >= 5.7% dysglycaemia", d["DYSGLYCAEMIA"]),
    ("HbA1c >= 6.5% diabetes-range", d["DIABETES_RANGE"]),
    ("PHQ-9 >= 10", d["PHQ10"]),
]

summary_rows = []

for period in PERIOD_ORDER:
    q = d[d["PERIOD"].eq(period)].copy()
    n_total = len(q)

    masks = {
        "Anaemia": q["ANAEMIA"],
        "HbA1c >= 5.7% dysglycaemia": q["DYSGLYCAEMIA"],
        "HbA1c >= 6.5% diabetes-range": q["DIABETES_RANGE"],
        "PHQ-9 >= 10": q["PHQ10"],
        "Anaemia + diabetes-range": q["ANAEMIA"] & q["DIABETES_RANGE"],
        "Anaemia + PHQ-9 >= 10": q["ANAEMIA"] & q["PHQ10"],
        "Diabetes-range + PHQ-9 >= 10": q["DIABETES_RANGE"] & q["PHQ10"],
        "All three": q["ANAEMIA"] & q["DIABETES_RANGE"] & q["PHQ10"],
        "Anaemia + dysglycaemia": q["ANAEMIA"] & q["DYSGLYCAEMIA"],
        "Dysglycaemia + PHQ-9 >= 10": q["DYSGLYCAEMIA"] & q["PHQ10"],
        "Anaemia + dysglycaemia + PHQ-9 >= 10":
            q["ANAEMIA"] & q["DYSGLYCAEMIA"] & q["PHQ10"],
    }

    for label, mask in masks.items():
        summary_rows.append({
            "period": period,
            "definition": label,
            "analytic_n": n_total,
            "n": int(mask.sum()),
            "percent": 100.0 * mask.mean(),
            "weighted_n": weighted_n(mask, q["SURVEY_WT"]),
            "weighted_percent": weighted_pct(mask, q["SURVEY_WT"]),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "59_clinical_overlap_summary.csv", index=False)

# -----------------------------------------------------------------------------
# 5. Eight mutually exclusive A / diabetes-range / PHQ states
# -----------------------------------------------------------------------------

exclusive_rows = []

for period in PERIOD_ORDER:
    q = d[d["PERIOD"].eq(period)].copy()

    for combo in COMBO_ORDER:
        mask = q["COMBINATION"].eq(combo)
        exclusive_rows.append({
            "period": period,
            "combination": combo,
            "analytic_n": len(q),
            "n": int(mask.sum()),
            "percent": 100.0 * mask.mean(),
            "weighted_n": weighted_n(mask, q["SURVEY_WT"]),
            "weighted_percent": weighted_pct(mask, q["SURVEY_WT"]),
        })

exclusive = pd.DataFrame(exclusive_rows)
exclusive.to_csv(OUT / "59_exclusive_three_domain_states.csv", index=False)

# -----------------------------------------------------------------------------
# 6. PHQ burden within physiological strata
# -----------------------------------------------------------------------------

strata_rows = []

for period in PERIOD_ORDER:
    q = d[d["PERIOD"].eq(period)].copy()

    strata = {
        "Neither anaemia nor diabetes-range":
            (~q["ANAEMIA"]) & (~q["DIABETES_RANGE"]),
        "Anaemia only":
            q["ANAEMIA"] & (~q["DIABETES_RANGE"]),
        "Diabetes-range only":
            (~q["ANAEMIA"]) & q["DIABETES_RANGE"],
        "Anaemia + diabetes-range":
            q["ANAEMIA"] & q["DIABETES_RANGE"],
    }

    for stratum, mask in strata.items():
        z = q[mask].copy()
        if len(z) == 0:
            continue

        strata_rows.append({
            "period": period,
            "physiological_stratum": stratum,
            "n": len(z),
            "PHQ9_mean": pd.to_numeric(z["PHQ9_TOTAL"], errors="coerce").mean(),
            "PHQ10_n": int(z["PHQ10"].sum()),
            "PHQ10_percent": 100.0 * z["PHQ10"].mean(),
            "PHQ10_weighted_percent": weighted_pct(
                z["PHQ10"], z["SURVEY_WT"]
            ),
        })

strata = pd.DataFrame(strata_rows)
strata.to_csv(OUT / "59_phq_within_physiology_strata.csv", index=False)

# -----------------------------------------------------------------------------
# 7. Figures with explicit axis labels
# -----------------------------------------------------------------------------

# Figure 1: weighted prevalence over time
plot_defs = [
    "Anaemia",
    "HbA1c >= 6.5% diabetes-range",
    "PHQ-9 >= 10",
]

p = summary[summary["definition"].isin(plot_defs)].copy()

x = np.arange(len(PERIOD_ORDER))
width = 0.24

fig, ax = plt.subplots(figsize=(10, 6))

for i, label in enumerate(plot_defs):
    z = (
        p[p["definition"].eq(label)]
        .set_index("period")
        .reindex(PERIOD_ORDER)
    )
    ax.bar(
        x + (i - 1) * width,
        z["weighted_percent"].to_numpy(),
        width=width,
        label=label,
    )

ax.set_xticks(x)
ax.set_xticklabels(PERIOD_ORDER)
ax.set_xlabel("NHANES analysis period")
ax.set_ylabel("Survey-weighted prevalence (%)")
ax.set_title("Anaemia, diabetes-range glycaemia and depressive symptom burden")
ax.legend(title="Operational definition")
fig.tight_layout()
fig.savefig(FIG / "59_three_domain_prevalence_by_period.png", dpi=300)
plt.close(fig)


# Figure 2: all eight mutually exclusive states
fig, ax = plt.subplots(figsize=(13, 7))

x = np.arange(len(COMBO_ORDER))
width = 0.25

for i, period in enumerate(PERIOD_ORDER):
    z = (
        exclusive[exclusive["period"].eq(period)]
        .set_index("combination")
        .reindex(COMBO_ORDER)
    )
    ax.bar(
        x + (i - 1) * width,
        z["weighted_percent"].to_numpy(),
        width=width,
        label=period,
    )

ax.set_xticks(x)
ax.set_xticklabels(COMBO_ORDER, rotation=35, ha="right")
ax.set_xlabel("Mutually exclusive anaemia / diabetes-range / PHQ-9 state")
ax.set_ylabel("Survey-weighted prevalence (%)")
ax.set_title("Three-domain clinical overlap across NHANES periods")
ax.legend(title="NHANES analysis period")
fig.tight_layout()
fig.savefig(FIG / "59_exclusive_overlap_by_period.png", dpi=300)
plt.close(fig)


# Figure 3: PHQ>=10 prevalence within physiological strata
STRATA_ORDER = [
    "Neither anaemia nor diabetes-range",
    "Anaemia only",
    "Diabetes-range only",
    "Anaemia + diabetes-range",
]

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(STRATA_ORDER))
width = 0.25

for i, period in enumerate(PERIOD_ORDER):
    z = (
        strata[strata["period"].eq(period)]
        .set_index("physiological_stratum")
        .reindex(STRATA_ORDER)
    )
    ax.bar(
        x + (i - 1) * width,
        z["PHQ10_weighted_percent"].to_numpy(),
        width=width,
        label=period,
    )

ax.set_xticks(x)
ax.set_xticklabels(STRATA_ORDER, rotation=25, ha="right")
ax.set_xlabel("Physiological stratum")
ax.set_ylabel("Survey-weighted prevalence of PHQ-9 >= 10 (%)")
ax.set_title("Depressive symptom burden within anaemia / diabetes-range strata")
ax.legend(title="NHANES analysis period")
fig.tight_layout()
fig.savefig(FIG / "59_phq10_within_physiology_strata.png", dpi=300)
plt.close(fig)

# -----------------------------------------------------------------------------
# 8. Manifest
# -----------------------------------------------------------------------------

manifest = {
    "script": "59_clinical_overlap_eda.py",
    "cohort_domain": DOMAIN,
    "anaemia_definition": "A > 0; equivalent to Hb <13 g/dL men or <12 g/dL women in locked pipeline",
    "dysglycaemia_definition": "HbA1c >= 5.7%",
    "diabetes_range_definition": "HbA1c >= 6.5%; biochemical threshold only, not clinical diagnosis",
    "depressive_burden_definition": "PHQ-9 >= 10; screening/severity threshold only, not clinical diagnosis",
    "periods": PERIOD_ORDER,
    "locked_outputs_modified": False,
}

(OUT / "59_overlap_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("PASS  Three-domain overlap analysis complete.")
print("PASS  Anaemia = locked sex-specific Hb deficit (A > 0).")
print("PASS  Diabetes-range = HbA1c >= 6.5%; dysglycaemia >= 5.7%.")
print("PASS  Depressive burden = PHQ-9 >= 10.")
print("PASS  Exact locked full-PHQ X3 domains used for all periods.")
print("PASS  Outputs written only under EDA/Results and EDA/Figures.")
