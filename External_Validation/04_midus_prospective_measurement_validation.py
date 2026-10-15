"""SCRIPT 04 | MIDUS prospective measurement and scoring validation (no model fit).

Place at <project>/External_Validation/04_midus_prospective_measurement_validation.py
Run: python External_Validation/04_midus_prospective_measurement_validation.py

Reads only local MIDUS .sav files in <project>/data/MIDUS/raw.
Exports aggregate counts, selected variable metadata, and QC diagnostics ONLY.
No participant records, identifiers, derived subject-level tables, or predictions
are written. Any aggregate count <=10 (including zero) is suppressed.

Scientific status: FEASIBILITY / SOURCE-LABEL audit, NOT full assay certification,
formal harmonization, representativeness, causal inference, or external validation.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import traceback

import numpy as np
import pandas as pd

try:
    import pyreadstat
except ImportError as exc:
    raise SystemExit("Install pyreadstat in the active environment: python -m pip install pyreadstat") from exc

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "MIDUS" / "raw"
OUT = ROOT / "External_Validation" / "Results"
OUT.mkdir(parents=True, exist_ok=True)
SMALL = "SUPPRESSED_0_TO_10"

# Explicit, audited source variables from scripts 01-03. Do not silently guess aliases.
FILES = {
    "M2_BIO": r"^M2_BIO_AGGREGATE_.*\.sav$",
    "M2_COG": r"^M2_P3_BTACT_.*\.sav$",
    "M2_SURVEY": r"^M2_P1_SURVEY_.*\.sav$",
    "M3_COG": r"^M3_P3_BTACT_.*\.sav$",
    "MR1_BIO": r"^MR1_BIO_AGGREGATE_.*\.sav$",
    "MR1_COG": r"^MR1_P3_BTACT_.*\.sav$",
    "MR1_SURVEY": r"^MR1_P1_SURVEY_.*\.sav$",
    "MR2_COG": r"^MR2_P3_BTACT.*\.sav$",
}

FIELDS = {
    "M2_BIO": {
        "HbA1c": "B4BHA1C", "Glucose": "B4BGLUC", "Insulin": "B4BINSLN",
        "CESD": "B4QCESD", "Age_clinic": "B4ZAGE", "Sex": "B1PRSEX",
        "year": "B4ZCOMPY", "month": "B4ZCOMPM",
    },
    "M2_COG": {
        "comp": "B3TCOMPZ1", "memory": "B3TEMZ1", "executive": "B3TEFZ1",
        "year": "B3PIDATE_YR", "month": "B3PIDATE_MO",
    },
    "M3_COG": {
        "comp": "C3TCOMP", "memory": "C3TEM", "executive": "C3TEF",
        "year": "C3IDATE_YR", "month": "C3IDATE_MO",
    },
    "M2_SURVEY": {"Education": "B1PB1", "BMI": "B1SBMI", "year": "B1PIDATE_YR", "month": "B1PIDATE_MO"},
    "MR1_BIO": {
        "HbA1c": "RA4BHA1C", "Glucose": "RA4BGLUC", "Insulin": "RA4BINSLN",
        "CESD": "RA4QCESD", "Age_clinic": "RA4ZAGE", "Sex": "RA1PRSEX",
        "year": "RA4ZCOMPY", "month": "RA4ZCOMPM",
    },
    "MR1_COG": {
        "comp": "RA3TCOMPZ1", "memory": "RA3TEMZ1", "executive": "RA3TEFZ1",
        "year": "RA3IDATE_YR", "month": "RA3IDATE_MO",
    },
    "MR2_COG": {
        "comp": "RB3TCOMPZ_Long", "memory": "RB3TEMZ_Long", "executive": "RB3TEFZC_Long",
        "year": "RB3IDATE_YR", "month": "RB3IDATE_MO",
    },
    "MR1_SURVEY": {"Education": "RA1PB1", "BMI": "RA1SBMI", "year": "RA1PIDATE_YR", "month": "RA1PIDATE_MO"},
}

COHORTS = {
    "CORE_M2_TO_M3": ("M2_BIO", "M2_COG", "M3_COG", "M2_SURVEY"),
    "REFRESHER_MR1_TO_MR2": ("MR1_BIO", "MR1_COG", "MR2_COG", "MR1_SURVEY"),
}

# Wide diagnostic ranges ONLY; they do not define inclusion or validate units.
RANGES = {
    "HbA1c": (2.5, 20.0, "percent, candidate"),
    "Glucose": (40.0, 600.0, "mg/dL, candidate"),
    "Insulin": (0.0, 500.0, "uIU/mL, candidate; positive needed for log"),
    "CESD": (0.0, 60.0, "CES-D score; scoring unverified"),
    "Age_clinic": (18.0, 110.0, "years at biomarker visit"),
    "BMI": (10.0, 90.0, "kg/m^2, candidate"),
    "comp": (-10.0, 10.0, "reference-standardized candidate BTACT"),
    "memory": (-10.0, 10.0, "reference-standardized candidate BTACT"),
    "executive": (-10.0, 10.0, "reference-standardized candidate BTACT"),
}


def cell(n: int) -> str:
    return str(n) if int(n) > 10 else SMALL


def file_for(label: str) -> Path:
    if not RAW.is_dir():
        raise FileNotFoundError(f"Missing data folder: {RAW}")
    pattern = re.compile(FILES[label], re.IGNORECASE)
    matches = [p for p in RAW.rglob("*") if p.is_file() and pattern.fullmatch(p.name)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one source for {label}, found {len(matches)}: {[p.name for p in matches]}")
    return matches[0]


def numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def valid_months(frame: pd.DataFrame) -> pd.Series:
    y, m = numeric(frame, "year"), numeric(frame, "month")
    ok = y.between(1990, 2030) & m.between(1, 12) & y.eq(np.floor(y)) & m.eq(np.floor(m))
    return (12 * y + m).where(ok)


def descr(s: pd.Series) -> tuple[str, str, str]:
    z = numeric(pd.DataFrame({"x": s}), "x").dropna()
    if len(z) <= 10:
        return ("NA", "NA", "NA")
    return tuple(f"{z.quantile(p):.3f}" for p in (0.05, 0.5, 0.95))


def load_one(label: str):
    p = file_for(label)
    _, md = pyreadstat.read_sav(str(p), metadataonly=True)
    lookup = {c.upper(): c for c in md.column_names}
    fields = FIELDS[label]
    id_name = "MRID" if label.startswith("MR") else "M2ID"
    missing = [v for v in [id_name, *fields.values()] if v.upper() not in lookup]
    if missing:
        raise RuntimeError(f"{label} missing expected fields: {missing}. No substitution attempted.")
    cols = [lookup[id_name.upper()], *(lookup[v.upper()] for v in fields.values())]
    # pyreadstat converts SPSS-declared user-missing codes to NaN here.
    df, _ = pyreadstat.read_sav(str(p), usecols=list(dict.fromkeys(cols)), user_missing=False,
                                apply_value_formats=False)
    renames = {lookup[v.upper()]: name for name, v in fields.items()}
    df = df.rename(columns=renames)
    original_id = lookup[id_name.upper()]
    if df[original_id].isna().any() or df[original_id].duplicated().any():
        raise RuntimeError(f"Unsafe or missing participant identifier in {label}")
    df = df.set_index(original_id)
    metadata = []
    labels = {v.upper(): (md.column_names_to_labels.get(v) or "") for v in md.column_names}
    # Automated SOURCE-LABEL checks, not instrument/assay certification.
    EXPECT = {
        "HbA1c": ("hemoglobin a1c",), "Glucose": ("fasting glucose",),
        "Insulin": ("fasting insulin",),
    }
    SCORE_REFERENCES = {
        "M2_COG": ("midus national sample",),
        "M3_COG": ("national midus 2",),
        "MR1_COG": ("midus refresher national sample",),
        "MR2_COG": ("national mr1",),
    }
    for name, var in fields.items():
        s = numeric(df, name)
        q5, q50, q95 = descr(s)
        lo, hi, unit = RANGES.get(name, (None, None, "categorical/date; not certified"))
        out = int((s.notna() & ~s.between(lo, hi)).sum()) if lo is not None else None
        declared_missing = getattr(md, "missing_ranges", {}) or {}
        expected = EXPECT.get(name, ())
        if name in ("comp", "memory", "executive"):
            expected = SCORE_REFERENCES.get(label, ())
        actual = labels[var.upper()].casefold()
        label_check = ("SOURCE_LABEL_MATCH" if expected and all(tok in actual for tok in expected)
                       else "SOURCE_LABEL_REVIEW" if expected else "NOT_APPLICABLE")
        metadata.append({
            "file_role": label, "source_variable": var, "semantic_name": name,
            "source_label": labels[var.upper()], "unit_or_scale_candidate": unit,
            "n_numeric": cell(int(s.notna().sum())),
            "p05": q5, "median": q50, "p95": q95,
            "n_outside_broad_diagnostic_range": cell(out) if out is not None else "NOT_APPLICABLE",
            "spss_declares_user_missing": bool(declared_missing.get(var)),
            "label_wording_check": label_check,
            "validation_status": "SOURCE_LABEL_ONLY__CODEBOOK_CONFIRMATION_REQUIRED",
        })
    return df, metadata


def count_rows(results, name: str, mask: pd.Series) -> None:
    results.append({"stage": name, "n_participants": cell(int(mask.sum()))})


def cohort_audit(cohort: str, datasets: dict[str, pd.DataFrame]):
    bio_name, base_name, later_name, survey_name = COHORTS[cohort]
    bio = datasets[bio_name].add_prefix("bio_")
    base = datasets[base_name].add_prefix("base_")
    later = datasets[later_name].add_prefix("later_")
    survey = datasets[survey_name].add_prefix("survey_")
    common = bio.index.intersection(base.index).intersection(later.index)
    if not common.is_unique:
        raise RuntimeError(f"Unsafe duplicate ID intersection: {cohort}")
    joined = bio.loc[common].join(base.loc[common], how="inner").join(later.loc[common], how="inner")
    # The survey is optional at the first gate; don't drop non-survey participants silently.
    joined = joined.join(survey, how="left")

    def n(col):
        return numeric(joined, col)

    physiology = n("bio_HbA1c").notna() & n("bio_Glucose").notna() & n("bio_Insulin").gt(0)
    bounds = (n("bio_HbA1c").between(2.5, 20) & n("bio_Glucose").between(40, 600)
              & n("bio_Insulin").between(0, 500, inclusive="right"))
    cesd = n("bio_CESD").notna()
    tb, t0, t1 = valid_months(joined[["bio_year", "bio_month"]].rename(columns={"bio_year":"year", "bio_month":"month"})), \
                 valid_months(joined[["base_year", "base_month"]].rename(columns={"base_year":"year", "base_month":"month"})), \
                 valid_months(joined[["later_year", "later_month"]].rename(columns={"later_year":"year", "later_month":"month"}))
    all_dates = tb.notna() & t0.notna() & t1.notna()
    # Required: baseline cognition BEFORE physiology BEFORE subsequent cognition.
    ordered = all_dates & t0.lt(tb) & tb.lt(t1)
    survey_date = valid_months(joined[["survey_year", "survey_month"]].rename(columns={"survey_year":"year", "survey_month":"month"}))
    pre_bio_survey = survey_date.notna() & tb.notna() & survey_date.lt(tb)
    basic_cov = n("bio_Age_clinic").between(18, 110) & n("bio_Sex").notna()
    plus_edu = basic_cov & n("survey_Education").notna() & pre_bio_survey
    plus_bmi = plus_edu & n("survey_BMI").between(10, 90)
    flow, notes = [], []
    count_rows(flow, "all_linked_bio+baseline_cog+later_cog", pd.Series(True, index=joined.index))
    count_rows(flow, "G3_finite_with_positive_insulin", physiology)
    count_rows(flow, "G3_and_CESD", physiology & cesd)
    count_rows(flow, "G3_and_three_valid_dates", physiology & all_dates)
    count_rows(flow, "G3_and_baseline_cognition_precedes_bio_precedes_later_cognition", physiology & ordered)
    count_rows(flow, "G3_and_ordered_and_broad_lab_range", physiology & ordered & bounds)
    count_rows(flow, "G3_and_ordered_and_age_and_sex", physiology & ordered & basic_cov)
    count_rows(flow, "G3_and_ordered_and_age_sex_and_prior_education", physiology & ordered & plus_edu)
    count_rows(flow, "G3_and_ordered_and_age_sex_prior_education_BMI", physiology & ordered & plus_bmi)
    count_rows(flow, "G3_and_ordered_and_CESD", physiology & ordered & cesd)

    for dim in ("comp", "memory", "executive"):
        both_score = n(f"base_{dim}").notna() & n(f"later_{dim}").notna()
        candidate = physiology & ordered & both_score
        score_bounds = n(f"base_{dim}").between(-10, 10) & n(f"later_{dim}").between(-10, 10)
        count_rows(flow, f"G3_and_both_{dim}_scores", physiology & both_score)
        count_rows(flow, f"G3_and_{dim}_both_scores_and_ordered", candidate)
        count_rows(flow, f"G3_and_{dim}_ordered_and_lab_score_broad_ranges", candidate & bounds & score_bounds)
        count_rows(flow, f"G3_and_{dim}_ordered_and_age_sex", candidate & basic_cov)
        count_rows(flow, f"G3_and_{dim}_ordered_and_age_sex_prior_education", candidate & plus_edu)
        count_rows(flow, f"G3_and_{dim}_ordered_and_age_sex_prior_education_BMI", candidate & plus_bmi)
        gap_0 = (tb - t0).where(candidate)
        gap_1 = (t1 - tb).where(candidate)
        gap_full = (t1 - t0).where(candidate)
        notes.append({
            "cohort": cohort, "dimension": dim, "n_candidate": cell(int(candidate.sum())),
            "baseline_cog_to_bio_median_months": descr(gap_0)[1],
            "bio_to_followup_cog_median_months": descr(gap_1)[1],
            "baseline_to_followup_cog_median_months": descr(gap_full)[1],
            "timing_rule": "baseline_cog_month < biomarker_month < followup_cog_month",
        })

    for rec in flow:
        rec["cohort"] = cohort
    return flow, notes


def main():
    inputs: dict[str, pd.DataFrame] = {}
    qc = []
    for key in FILES:
        frame, rec = load_one(key)
        inputs[key] = frame
        qc.extend(rec)
        print(f"  {key:11s} | {len(frame):4d} records | {len(FIELDS[key]):2d} variables inspected")
    flows, times = [], []
    for c in COHORTS:
        f, t = cohort_audit(c, inputs)
        flows.extend(f)
        times.extend(t)

    df_qc = pd.DataFrame(qc)
    review_labels = df_qc.loc[df_qc.label_wording_check == "SOURCE_LABEL_REVIEW",
                              ["file_role", "semantic_name", "source_variable"]]
    df_flow = pd.DataFrame(flows)
    df_time = pd.DataFrame(times)
    df_qc.to_csv(OUT / "04_variable_scoring_qc.csv", index=False, encoding="utf-8")
    df_flow.to_csv(OUT / "04_prospective_sample_flow.csv", index=False, encoding="utf-8")
    df_time.to_csv(OUT / "04_prospective_chronology.csv", index=False, encoding="utf-8")

    def stage(co, st):
        rec = df_flow.loc[(df_flow.cohort == co) & (df_flow.stage == st), "n_participants"]
        return rec.iloc[0] if len(rec) else "NOT_COMPUTED"

    lines = [
        "# Script 04 — US MIDUS prospective measurement/scoring audit",
        "",
        "**Nature:** Local SPSS metadata + participant-level checks conducted in memory; only aggregate outputs saved.",
        "**This is NOT a fitted model, proof of score invariance, or independent clinical assay certification.**",
        "**Disclosure:** All cells N <= 10 (including 0) suppressed.",
        "",
        "## Prospective sample sizes",
        "",
        "| Cohort | G3 + baseline/follow-up composite + correct dates | + clinic age/sex | + prior survey education | + prior BMI |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort in COHORTS:
        prefix = "G3_and_comp_ordered"
        lines.append("| " + " | ".join([
            cohort,
            stage(cohort,"G3_and_comp_both_scores_and_ordered"),
            stage(cohort,"G3_and_comp_ordered_and_age_sex"),
            stage(cohort,"G3_and_comp_ordered_and_age_sex_prior_education"),
            stage(cohort,"G3_and_comp_ordered_and_age_sex_prior_education_BMI"),
        ]) + " |")
    lines.extend([
        "", "## Three-domain measurement checks", "",
        "| Cohort | Composite N | Episodic memory N | Executive function N |",
        "|---|---:|---:|---:|",
    ])
    for c in COHORTS:
        lines.append(f"| {c} | {stage(c,'G3_and_comp_both_scores_and_ordered')} | "
                     f"{stage(c,'G3_and_memory_both_scores_and_ordered')} | "
                     f"{stage(c,'G3_and_executive_both_scores_and_ordered')} |")
    lines += ["", "## Timing in analyzable composite-score sample", "",
              "| Cohort | N | Baseline cognition → blood (median months) | Blood → follow-up cognition (median months) |",
              "|---|---:|---:|---:|"]
    for d in times:
        if d["dimension"] == "comp":
            lines.append(f"| {d['cohort']} | {d['n_candidate']} | {d['baseline_cog_to_bio_median_months']} | {d['bio_to_followup_cog_median_months']} |")
    lines += [
        "", "## Explicit source-score crosswalk (candidate)", "",
        "- **Core:** baseline `B3TCOMPZ1`, `B3TEMZ1`, `B3TEFZ1` (MIDUS 2 national z scores);"
        " follow-up `C3TCOMP`, `C3TEM`, `C3TEF` (labels state standardized against MIDUS 2 national mean/SD).",
        "- **Refresher:** baseline `RA3TCOMPZ1`, `RA3TEMZ1`, `RA3TEFZ1` (Refresher 1 national z scores);"
        " follow-up `RB3TCOMPZ_Long`, `RB3TEMZ_Long`, `RB3TEFZC_Long` (labels state Refresher 1 national mean/SD).",
        "- Do NOT pool raw composite values from Core and Refresher: they reference DIFFERENT national cohorts.",
        "- Longitudinal *score-reference* compatibility is suggested by labels, not proven."
        " Source scoring manuals, item definition changes, invariance/retest effects still require review.",
        "", "## Source marker meanings (not independent laboratory verification)", "",
        "- HbA1c source label: percentage; fasting glucose: mg/dL; fasting insulin: uIU/mL.",
        "- Positive insulin is required for log(Insulin). No log transform, scaling or outcome modelling was performed.",
        "- `04_variable_scoring_qc.csv` documents numeric availability, selected quantiles, provisional extreme-value flags, and the SPSS source labels.",
        "- SPSS-declared user-missing codes were parsed as missing; undocumented numeric sentinels remain possible.",
        "", "## Methodological gates before the first model", "",
        "1. **Clinical lab confirmation required:** check the official biomarker assay protocol, fasting compliance, laboratory harmonization, and exclusions.",
        "2. **BTACT manual confirmation required:** verify composite construction and whether longitudinal reference-standardization and measurement invariance support comparison.",
        "3. **Selection/attrition:** linked blood/cognition subsamples are not the population cohort; evaluate weighting and attrition strategy.",
        "4. **Chronology:** only treat strictly `baseline cognition < blood < later cognition` at month resolution as temporally ordered; same-month ambiguous.",
        "5. **Covariates:** survey education/BMI were collected before blood; clinic age is measured at blood visit. Their scientific role and missingness must be assessed.",
        "6. **NHANES:** the US G3 panel is an ADAPTED glycemic study; do not call it a replication of the frozen 7-marker A/G decomposition.",
        "7. **No causal claim:** temporally ordered associations do not identify causal physiological effects.",
        "", "## Files", "",
        "- `04_variable_scoring_qc.csv` (source labels, unit candidate, broad QC diagnostics)",
        "- `04_prospective_sample_flow.csv` (aggregate cohort gates)",
        "- `04_prospective_chronology.csv` (median gaps by cognitive domain)",
    ]
    (OUT / "04_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SCRIPT 04 | US prospective measurement/scoring feasibility | 2 cohorts | 0 errors")
    for c in COHORTS:
        print(f"  {c:21s} | strictly ordered G3 + BTACT composite: {stage(c, 'G3_and_comp_both_scores_and_ordered')}")
    print(f"OUTPUT: {OUT / '04_readout.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"SCRIPT 04 FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
