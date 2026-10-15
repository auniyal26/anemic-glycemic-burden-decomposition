"""SCRIPT 03 | MIDUS/MIDJA data atlas and temporal feasibility audit.

Place in <project-root>/External_Validation/ and run:
    python External_Validation/03_midus_data_atlas.py

Requires: pandas, numpy, pyreadstat.
All participant-level operations are local and in-memory. Outputs contain only
variable metadata and aggregate statistics; no individual IDs or rows.

NOT a harmonization or modelling script. Instrument equivalence, clinical
units, sample design, missing codes, and visit timing require independent review.
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
    raise SystemExit("Install pyreadstat: python -m pip install pyreadstat") from exc

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "MIDUS" / "raw"
OUT = ROOT / "External_Validation" / "Results"
OUT.mkdir(parents=True, exist_ok=True)
SUPPRESSED = "SUPPRESSED_0_TO_10"

# Explicit source-variable IDs, verified against Script 01 SPSS label catalog.
# These codes are candidates; lab units and outcome definitions remain unverified.
FEATURES = {
    "M2_BIO": {
        "HbA1c": "B4BHA1C", "Glucose": "B4BGLUC", "Insulin": "B4BINSLN",
        "CRP": "B4BCRP", "Fibrinogen": "B4BFGN", "Creatinine": "B4BSCREA",
        "Total_cholesterol": "B4BCHOL", "HDL": "B4BHDL", "LDL": "B4BLDL", "Triglycerides": "B4BTRIGL",
        "CESD_total": "B4QCESD", "CESD_somatic": "B4QCESDSC", "CESD_affect": "B4QCESDDA",
    },
    "M3_BIO": {
        "HbA1c": "C4BHA1C", "Glucose": "C4BGLUC", "Insulin": "C4BINSLN",
        "CRP": "C4BCRP", "Fibrinogen": "C4BFGN", "Creatinine": "C4BSCREA",
        "Total_cholesterol": "C4BCHOL", "HDL": "C4BHDL", "LDL": "C4BLDL", "Triglycerides": "C4BTRIGL",
        "CESD_total": "C4QCESD", "CESD_somatic": "C4QCESDSC", "CESD_affect": "C4QCESDDA",
    },
    "MR1_BIO": {
        "HbA1c": "RA4BHA1C", "Glucose": "RA4BGLUC", "Insulin": "RA4BINSLN",
        "CRP": "RA4BCRP", "Fibrinogen": "RA4BFGN", "Creatinine": "RA4BSCREA",
        "Total_cholesterol": "RA4BCHOL", "HDL": "RA4BHDL", "LDL": "RA4BLDL", "Triglycerides": "RA4BTRIGL",
        "CESD_total": "RA4QCESD", "CESD_somatic": "RA4QCESDSC", "CESD_affect": "RA4QCESDDA",
    },
    "MIDJA_1_BIO": {
        "Hb": "J2CHGB", "RBC": "J2CRBC", "MCV": "J2CMCV", "Hematocrit": "J2CHTC",
        "MCH": "J2CMCH", "MCHC": "J2CMCHC", "WBC": "J2CWBC",
        "HbA1c_raw": "J2CHBA1C", "HbA1c_adjusted": "J2BHA1CA",
        "CRP": "J2BCRP", "Fibrinogen": "J2BFGN", "Creatinine": "J2BSCREA",
        "Total_cholesterol": "J2BCHOL", "HDL_original": "J2BHDL", "HDL_adjusted": "J2BHDLA",
        "CESD_total": "J2QCESD", "CESD_somatic": "J2QCESDSC", "CESD_affect": "J2QCESDDA",
    },
    "MIDJA_2_BIO": {
        "Hb": "K2CHGB", "RBC": "K2CRBC", "MCV": "K2CMCV", "Hematocrit": "K2CHTC",
        "MCH": "K2CMCH", "MCHC": "K2CMCHC", "WBC": "K2CWBC",
        "HbA1c_raw": "K2CHBA1C", "HbA1c_adjusted": "K2BHA1CA",
        "CRP": "K2BCRP", "Fibrinogen": "K2BFGN", "Creatinine": "K2BSCREA",
        "Total_cholesterol": "K2BCHOL", "HDL_original": "K2BHDL", "HDL_adjusted": "K2BHDLA",
        "CESD_total": "K2QCESD",
    },
    "M2_COG": {
        "BTACT_composite_national": "B3TCOMPZ1", "BTACT_composite_all": "B3TCOMPZ3",
        "BTACT_memory": "B3TEMZ1", "BTACT_executive": "B3TEFZ1",
    },
    "M3_COG": {
        "BTACT_composite": "C3TCOMP", "BTACT_memory": "C3TEM", "BTACT_executive": "C3TEF",
    },
    "MR1_COG": {
        "BTACT_composite_national": "RA3TCOMPZ1", "BTACT_composite_all": "RA3TCOMPZ3",
        "BTACT_memory": "RA3TEMZ1", "BTACT_executive": "RA3TEFZ1",
    },
    "MR2_COG": {
        "BTACT_composite_longitudinal": "RB3TCOMPZ_Long",
        "BTACT_memory_longitudinal": "RB3TEMZ_Long",
        "BTACT_executive_longitudinal": "RB3TEFZC_Long",
    },
    "M2_SURVEY": {"Age": "B1PAGE_M2", "Sex": "B1PRSEX", "BMI": "B1SBMI", "Education": "B1PB1", "Two_week_sadness": "B1PA60"},
    "M3_SURVEY": {"Age": "C1PRAGE", "Sex": "C1PRSEX", "BMI": "C1SBMI", "Education": "C1PB1", "Two_week_sadness": "C1PA60"},
    "MR1_SURVEY": {"Age": "RA1PRAGE", "Sex": "RA1PRSEX", "BMI": "RA1SBMI", "Education": "RA1PB1", "Two_week_sadness": "RA1PA60"},
    "MR2_SURVEY": {"Age": "RB1PRAGE", "Sex": "RB1PRSEX", "BMI": "RB1SBMI", "Education": "RB1PB1", "Two_week_sadness": "RB1PA60"},
    "MIDJA_1_SURVEY": {"Sex": "J1SQ1", "Education": "J1SQ3", "Self_report_anxiety_depression": "J1SA8T"},
    "MIDJA_2_SURVEY": {"Sex": "K1SQ1", "Education": "K1SQ3", "Self_report_anxiety_depression": "K1SA9T", "Household_income": "K1SF8"},
}

DATES = {
    "M2_SURVEY": ("B1PIDATE_YR", "B1PIDATE_MO"),
    "M3_SURVEY": ("C1PIDATE_YR", "C1PIDATE_MO"),
    "MR1_SURVEY": ("RA1PIDATE_YR", "RA1PIDATE_MO"),
    "MR2_SURVEY": ("RB1PIDATE_YR", "RB1PIDATE_MO"),
    "M2_BIO": ("B4ZCOMPY", "B4ZCOMPM"),
    "M2_COG": ("B3PIDATE_YR", "B3PIDATE_MO"),
    "M3_BIO": ("C4ZCOMPY", "C4ZCOMPM"),
    "M3_COG": ("C3IDATE_YR", "C3IDATE_MO"),
    "MR1_BIO": ("RA4ZCOMPY", "RA4ZCOMPM"),
    "MR1_COG": ("RA3IDATE_YR", "RA3IDATE_MO"),
    "MR2_COG": ("RB3IDATE_YR", "RB3IDATE_MO"),
    "MIDJA_1_BIO": ("J2CYEAR", "J2CMONTH"),
    "MIDJA_2_BIO": ("K2CYEAR", "K2CMONTH"),
}

PANELS = {
    "M2_BIO": {
        "G3": ("HbA1c", "Glucose", "Insulin"),
        "G3+inflammation": ("HbA1c", "Glucose", "Insulin", "CRP", "Fibrinogen"),
        "G3+metabolic": ("HbA1c", "Glucose", "Insulin", "HDL", "Triglycerides", "Creatinine"),
        "G3+CESD": ("HbA1c", "Glucose", "Insulin", "CESD_total"),
    },
    "M3_BIO": {
        "G3": ("HbA1c", "Glucose", "Insulin"),
        "G3+inflammation": ("HbA1c", "Glucose", "Insulin", "CRP", "Fibrinogen"),
        "G3+metabolic": ("HbA1c", "Glucose", "Insulin", "HDL", "Triglycerides", "Creatinine"),
        "G3+CESD": ("HbA1c", "Glucose", "Insulin", "CESD_total"),
    },
    "MR1_BIO": {
        "G3": ("HbA1c", "Glucose", "Insulin"),
        "G3+inflammation": ("HbA1c", "Glucose", "Insulin", "CRP", "Fibrinogen"),
        "G3+metabolic": ("HbA1c", "Glucose", "Insulin", "HDL", "Triglycerides", "Creatinine"),
        "G3+CESD": ("HbA1c", "Glucose", "Insulin", "CESD_total"),
    },
    "MIDJA_1_BIO": {
        "H3+HbA1c_raw": ("Hb", "RBC", "MCV", "HbA1c_raw"),
        "H7+HbA1c_raw": ("Hb", "RBC", "MCV", "Hematocrit", "MCH", "MCHC", "WBC", "HbA1c_raw"),
        "H3+HbA1c_raw+CESD": ("Hb", "RBC", "MCV", "HbA1c_raw", "CESD_total"),
        "H7+HbA1c_raw+inflammation": ("Hb", "RBC", "MCV", "Hematocrit", "MCH", "MCHC", "WBC", "HbA1c_raw", "CRP", "Fibrinogen"),
    },
    "MIDJA_2_BIO": {
        "H3+HbA1c_raw": ("Hb", "RBC", "MCV", "HbA1c_raw"),
        "H7+HbA1c_raw": ("Hb", "RBC", "MCV", "Hematocrit", "MCH", "MCHC", "WBC", "HbA1c_raw"),
        "H3+HbA1c_raw+CESD": ("Hb", "RBC", "MCV", "HbA1c_raw", "CESD_total"),
        "H7+HbA1c_raw+inflammation": ("Hb", "RBC", "MCV", "Hematocrit", "MCH", "MCHC", "WBC", "HbA1c_raw", "CRP", "Fibrinogen"),
    },
}

# Pairwise chronological checks. These do NOT assume outcome happens after exposure.
PAIRS = (
    ("Core: M2 survey vs M2 biomarker", "M2_SURVEY", "M2_BIO"),
    ("Core: M3 survey vs M3 biomarker", "M3_SURVEY", "M3_BIO"),
    ("Refresher: MR1 survey vs MR1 biomarker", "MR1_SURVEY", "MR1_BIO"),
    ("Core: M2 biomarker vs M2 cognition", "M2_BIO", "M2_COG"),
    ("Core: M2 biomarker vs M3 cognition", "M2_BIO", "M3_COG"),
    ("Core: M3 biomarker vs M3 cognition", "M3_BIO", "M3_COG"),
    ("Core: M2 biomarker vs M3 biomarker", "M2_BIO", "M3_BIO"),
    ("Core: M2 cognition vs M3 cognition", "M2_COG", "M3_COG"),
    ("Refresher: MR1 biomarker vs MR1 cognition", "MR1_BIO", "MR1_COG"),
    ("Refresher: MR1 biomarker vs MR2 cognition", "MR1_BIO", "MR2_COG"),
    ("Refresher: MR1 cognition vs MR2 cognition", "MR1_COG", "MR2_COG"),
    ("Japan: MIDJA1 biomarker vs MIDJA2 biomarker", "MIDJA_1_BIO", "MIDJA_2_BIO"),
)


def suppress(n: int) -> str:
    return str(n) if n > 10 else SUPPRESSED


def pct(n: int, den: int) -> str:
    if den <= 10 or n <= 10 or 0 <= den-n <= 10:
        return "WITHHELD_SMALL_CELL"
    return f"{100 * n / den:.1f}%"


def summary_num(s: pd.Series) -> tuple[str, str, str]:
    a = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) <= 10:
        return "NA", "NA", "NA"
    return tuple(f"{float(a.quantile(q)):.4g}" for q in (0.05, 0.5, 0.95))


def locate() -> dict[str, Path]:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing input directory: {RAW}")
    files = [p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() == ".sav"]
    found: dict[str, Path] = {}
    patterns = {
        "M2_BIO": r"^M2_BIO_AGGREGATE_", "M2_COG": r"^M2_P3_BTACT_", "M2_SURVEY": r"^M2_P1_SURVEY_",
        "M3_BIO": r"^M3_BIO_AGGREGATE_", "M3_COG": r"^M3_P3_BTACT_", "M3_SURVEY": r"^M3_P1_SURVEY_",
        "MR1_BIO": r"^MR1_BIO_AGGREGATE_", "MR1_COG": r"^MR1_P3_BTACT_", "MR1_SURVEY": r"^MR1_P1_SURVEY_",
        "MR2_COG": r"^MR2_P3_BTACT", "MR2_SURVEY": r"^MR2_P1_SURVEY_", "MIDJA_1_BIO": r"^MIDJA_1_Biomarker_", "MIDJA_1_SURVEY": r"^MIDJA_ SURVEY_ DATA_",
        "MIDJA_2_BIO": r"^MIDJA 2_Biomarker_", "MIDJA_2_SURVEY": r"^MIDJA_2_SURVEY_DATA_",
    }
    for key, pattern in patterns.items():
        matches = [p for p in files if re.search(pattern, p.name, re.I)]
        if len(matches) == 1:
            found[key] = matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous files for {key}: {[p.name for p in matches]}")
        else:
            print(f"WARNING: No file for {key}")
    return found


def read_fields(path: Path, fields: dict[str, str], y_m: tuple[str, str] | None):
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    col_lookup = {x.upper(): x for x in meta.column_names}
    requested = set(fields.values())
    if y_m:
        requested.update(y_m)
    requested.add("MIDJA_IDs" if "MIDJA" in path.name.upper() else ("MRID" if path.name.upper().startswith("MR") else "M2ID"))
    present = [col_lookup[x.upper()] for x in requested if x.upper() in col_lookup]
    data, _ = pyreadstat.read_sav(str(path), usecols=present, user_missing=False, apply_value_formats=False)
    data.columns = [x.upper() for x in data.columns]
    labels = {c.upper(): (meta.column_names_to_labels.get(c) or "") for c in meta.column_names}
    return data, labels


def finite_numeric(frame, column):
    if column is None or column.upper() not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column.upper()], errors="coerce").replace([np.inf, -np.inf], np.nan)


def months(frame, ym):
    if not ym:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    year = finite_numeric(frame, ym[0]); month = finite_numeric(frame, ym[1])
    good = year.between(1990, 2030) & month.between(1, 12) & (year % 1 == 0) & (month % 1 == 0)
    return (year * 12 + month).where(good)


def complete(frame, mapping, names):
    wanted = [mapping.get(n) for n in names]
    if any(c is None or c.upper() not in frame for c in wanted):
        return pd.Series(False, index=frame.index)
    vals = [finite_numeric(frame, c).notna() for c in wanted]
    # Positive insulin is required for later log transform; this is not full lab QC.
    if "Insulin" in names:
        vals.append(finite_numeric(frame, mapping["Insulin"]) > 0)
    ans = vals[0]
    for a in vals[1:]:
        ans = ans & a
    return ans


def analysis() -> None:
    paths = locate()
    if not paths:
        raise RuntimeError("No relevant SPSS files were found")
    frames: dict[str, pd.DataFrame] = {}
    mappings: dict[str, dict[str, str]] = {}
    atlas = []; panels = []; chronology = []
    for key, path in paths.items():
        mapping = FEATURES[key]
        frame, labels = read_fields(path, mapping, DATES.get(key))
        id_col = "MIDJA_IDS" if key.startswith("MIDJA") else ("MRID" if key.startswith("MR") else "M2ID")
        if id_col not in frame:
            print(f"SKIPPING {key}: identifier {id_col} not found")
            continue
        ids = frame[id_col]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"Unsafe linkage: duplicate or missing IDs in {key}")
        frame = frame.set_index(id_col, verify_integrity=True)
        frames[key] = frame; mappings[key] = mapping
        for name, col in mapping.items():
            exists = col.upper() in frame
            s = finite_numeric(frame, col)
            q05, median, q95 = summary_num(s)
            atlas.append({
                "dataset": key, "role": key.rsplit("_", 1)[-1],
                "measure": name, "variable": col, "label": labels.get(col.upper(), ""),
                "present": exists, "n_rows": len(frame), "n_numeric": suppress(int(s.notna().sum())),
                "p05": q05, "median": median, "p95": q95,
                "unit_validated": False, "score_validated": False,
            })
        for panel_name, measures in PANELS.get(key, {}).items():
            all_found = all(mapping.get(n, "").upper() in frame for n in measures)
            n = int(complete(frame, mapping, measures).sum()) if all_found else 0
            panels.append({
                "dataset": key, "panel": panel_name, "all_fields_present": all_found,
                "n_numeric_complete": suppress(n), "denominator": len(frame),
                "definition": " + ".join(measures),
                "note": "Numeric availability only. Not fully screened or harmonized."
            })
        print(f"  {key:12s} | {len(frame):4d} records | {len(mapping):2d} curated measures")

    for label, source, target in PAIRS:
        if source not in frames or target not in frames:
            continue
        a = frames[source]; b = frames[target]
        common = a.index.intersection(b.index)
        a = a.loc[common]; b = b.loc[common]
        date_a = months(a, DATES.get(source)); date_b = months(b, DATES.get(target))
        both = date_a.notna() & date_b.notna()
        gaps = (date_b[both] - date_a[both]).astype(float)
        earlier = int((gaps > 0).sum())
        same = int((gaps == 0).sum())
        later = int((gaps < 0).sum())
        valid = int(both.sum())
        gap_stats = summary_num(gaps)
        physiology_rule = None
        if source.endswith("BIO"):
            phys = ("Hb", "RBC", "MCV", "HbA1c_raw") if source.startswith("MIDJA") else ("HbA1c", "Glucose", "Insulin")
            src_valid = complete(a, mappings[source], phys)
            if target.endswith("COG"):
                comp = next((k for k in mappings[target] if k.startswith("BTACT_composite")), None)
                dst_valid = complete(b, mappings[target], (comp,)) if comp else pd.Series(False, index=b.index)
            elif target.endswith("BIO"):
                dstphys = ("Hb", "RBC", "MCV", "HbA1c_raw") if target.startswith("MIDJA") else ("HbA1c", "Glucose", "Insulin")
                dst_valid = complete(b, mappings[target], dstphys)
            else:
                dst_valid = pd.Series(True, index=b.index)
            joint = src_valid & dst_valid & both
            n_joint = int(joint.sum())
            before_joint = int((joint & (date_a < date_b)).sum())
            physiology_rule = f"{suppress(n_joint)} measured-and-dated; {suppress(before_joint)} source-before-target"
        chronology.append({
            "comparison": label, "source": source, "target": target,
            "n_shared_ids": suppress(len(common)), "n_both_valid_months": suppress(valid),
            "n_source_precedes_target": suppress(earlier),
            "n_same_month": suppress(same), "n_source_follows_target": suppress(later),
            "pct_source_precedes": pct(earlier, valid),
            "gap_months_p05": gap_stats[0], "gap_months_median": gap_stats[1],
            "gap_months_p95": gap_stats[2],
            "candidate_measured_and_dated": physiology_rule or "NOT_EVALUATED",
            "note": "Month-level chronology. Same-month order unresolved. Not a causal design.",
        })

    pd.DataFrame(atlas).to_csv(OUT / "03_measurement_atlas.csv", index=False, encoding="utf-8")
    pd.DataFrame(panels).to_csv(OUT / "03_panel_feasibility.csv", index=False, encoding="utf-8")
    pd.DataFrame(chronology).to_csv(OUT / "03_visit_chronology.csv", index=False, encoding="utf-8")

    lines = [
        "# Script 03 — MIDUS/MIDJA data atlas and visit chronology", "",
        "**Purpose:** Understand available marker domains, how much they overlap, and when assessments occurred.",
        "**Not yet done:** Clinical assay/unit verification, missing-code validation, BTACT/CES-D invariance, NHANES transfer, modeling, or causal inference.",
        "**Privacy:** Counts from 0 to 10 suppressed; no individual records or identifiers exported.", "",
        "## What is measured (selected variables, not an exhaustive catalog)", "",
        "The detailed `03_measurement_atlas.csv` gives the source label, numeric coverage and unvalidated distribution for every curated variable.", "",
        "- **US MIDUS:** HbA1c, fasting glucose, fasting insulin, CRP, fibrinogen, creatinine, lipids; CES-D in biomarker files; BTACT in cognitive files.",
        "- **Japan MIDJA:** Hb, RBC, MCV, hematocrit, MCH, MCHC, WBC, raw/adjusted HbA1c, CRP, fibrinogen, lipids; CES-D; no equivalent BTACT download in this batch.",
        "- **Survey files (all waves):** sampled age, sex, education, BMI, and self-reported depressive history, where available. These are questionnaire fields, not validated confounder definitions.",
        "- Do **not** presume shared measurement units from shared variable names; verify original and adjusted HbA1c independently.", "",
        "## Panel availability — numeric-only (not screened valid)", "",
        "| Dataset | Proposed marker panel | N with numeric measurements | Source N |",
        "|---|---|---:|---:|",
    ]
    for r in panels:
        lines.append(f"| {r['dataset']} | {r['panel']} | {r['n_numeric_complete']} | {r['denominator']} |")
    lines.extend(["", "## Visit chronology (source → target)", "",
                  "Positive date gap means source was assessed earlier than target. Same month does not establish ordering.", "",
                  "| Comparison | Linked IDs | Dated pairs | Source earlier | Same month | Source later | Median target−source (months) |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for r in chronology:
        lines.append(f"| {r['comparison']} | {r['n_shared_ids']} | {r['n_both_valid_months']} | {r['n_source_precedes_target']} | {r['n_same_month']} | {r['n_source_follows_target']} | {r['gap_months_median']} |")
    lines.extend(["", "## Joint date + measurement diagnostic", ""])
    for r in chronology:
        lines.append(f"- **{r['comparison']}:** {r['candidate_measured_and_dated']}.")
    lines.extend(["", "## Hypotheses to choose from AFTER this audit", "",
        "1. US glycemic physiological signatures and later BTACT, controlling for baseline BTACT, age and measured confounders (temporal ordering required).",
        "2. Stability of the US 3-marker glycemic representation across M2/M3 and Refresher; report construct changes rather than relabeling as the full A/G decomposition.",
        "3. Japan Hb/RBC/MCV/HbA1c profiles and CES-D, within-person changes; calibrate laboratory units and inspect collinearity before decomposition.",
        "4. Sensitivity to CRP/fibrinogen or lipid markers as prespecified contextual variables, not an unbounded feature search.",
        "", "## Decision gates", "",
        "- Verify full metadata, laboratory units, assay harmonization and missingness (especially Japanese RBC and HbA1c adjusted vs raw).",
        "- Check CES-D subscale scoring and whether BTACT standardizations are on a common longitudinal reference.",
        "- Study attrition/selection, causal ordering and potential confounders; no causal conclusions from pairwise correlations.",
        "- Do not train or evaluate on wave M3 physiological measurements to predict earlier M3 cognition; use participant-specific chronology.",
        "- Preserve all frozen NHANES maps; adapted US and Japan representations must be labeled separately.",
        "", "## Outputs", "", "- `03_readout.md`", "- `03_measurement_atlas.csv`", "- `03_panel_feasibility.csv`", "- `03_visit_chronology.csv`", "",
    ])
    (OUT / "03_readout.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SCRIPT 03 | {len(paths)} local files | {len(atlas)} selected variables | {len(chronology)} date comparisons")
    print(f"OUTPUT: {OUT / '03_readout.md'}")


if __name__ == "__main__":
    try:
        analysis()
    except Exception as exc:
        print(f"SCRIPT 03 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
