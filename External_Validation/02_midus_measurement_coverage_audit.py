#!/usr/bin/env python3
"""Script 02 | MIDUS/MIDJA measurement coverage and longitudinal feasibility.

DATA STAYS LOCAL. Reads SPSS files in the project root's data/MIDUS/raw folder;
exports ONLY variable-level metadata, aggregate counts and aggregate distributions.
No patient/participant data, IDs, joined row-level tables or individual dates are saved.

Run: python External_Validation/02_midus_measurement_coverage_audit.py
Dependency: python -m pip install pandas pyreadstat

Script 02 is a feasibility audit, NOT phenotype-model replication or verified
harmonization. Variables chosen from the SPSS labels discovered in Script 01.
Revision 2026-10-15: MIDJA RBC raw-scale audit no longer applies an
unjustified 1-9 cutoff before verifying Tokyo-laboratory units.
Consult MIDUS codebooks before treating any field as analytically harmonized.

Outputs: External_Validation/Results/02_readout.md,
         02_variable_map.csv, 02_measurement_coverage.csv,
         02_joint_coverage.csv, 02_longitudinal_coverage.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


# Explicitly mapped SPSS codes. MIDJA 'HbA1c_adj' is an alternative; it is NOT
# silently substituted for the Tokyo-laboratory HbA1c value.
BIO = {
    "M2": {
        "HbA1c": "B4BHA1C", "Glucose": "B4BGLUC", "Insulin": "B4BINSLN",
        "CESD_total": "B4QCESD", "CESD_somatic": "B4QCESDSC",
        "CESD_affect": "B4QCESDDA", "Visit_year": "B4ZCOMPY", "Visit_month": "B4ZCOMPM",
    },
    "M3": {
        "HbA1c": "C4BHA1C", "Glucose": "C4BGLUC", "Insulin": "C4BINSLN",
        "CESD_total": "C4QCESD", "CESD_somatic": "C4QCESDSC",
        "CESD_affect": "C4QCESDDA", "Visit_year": "C4ZCOMPY", "Visit_month": "C4ZCOMPM",
    },
    "MR1": {
        "HbA1c": "RA4BHA1C", "Glucose": "RA4BGLUC", "Insulin": "RA4BINSLN",
        "CESD_total": "RA4QCESD", "CESD_somatic": "RA4QCESDSC",
        "CESD_affect": "RA4QCESDDA",
    },
    "MIDJA_1": {
        "Hb": "J2CHGB", "RBC": "J2CRBC", "MCV": "J2CMCV",
        "HbA1c": "J2CHBA1C", "HbA1c_adj": "J2BHA1CA",
        "CESD_total": "J2QCESD", "CESD_somatic": "J2QCESDSC",
        "CESD_affect": "J2QCESDDA", "Visit_year": "J2CYEAR", "Visit_month": "J2CMONTH",
    },
    "MIDJA_2": {
        "Hb": "K2CHGB", "RBC": "K2CRBC", "MCV": "K2CMCV",
        "HbA1c": "K2CHBA1C", "HbA1c_adj": "K2BHA1CA",
        "CESD_total": "K2QCESD", "Visit_year": "K2CYEAR", "Visit_month": "K2CMONTH",
    },
}

COG = {
    "M2": {
        "BTACT_comp": "B3TCOMPZ1",  # MIDUS national-only standardization
        "BTACT_comp_all": "B3TCOMPZ3",  # separate alternative, not merged
        "BTACT_memory": "B3TEMZ1", "BTACT_exec": "B3TEFZ1",
        "Cog_year": "B3PIDATE_YR", "Cog_month": "B3PIDATE_MO",
    },
    "M3": {
        "BTACT_comp": "C3TCOMP", "BTACT_memory": "C3TEM", "BTACT_exec": "C3TEF",
        "Cog_year": "C3IDATE_YR", "Cog_month": "C3IDATE_MO",
    },
    "MR1": {
        "BTACT_comp": "RA3TCOMPZ1", "BTACT_comp_all": "RA3TCOMPZ3",
        "BTACT_memory": "RA3TEMZ1", "BTACT_exec": "RA3TEFZ1",
    },
    "MR2": {
        "BTACT_comp": "RB3TCOMPZ_Long", "BTACT_memory": "RB3TEMZ_Long",
        "BTACT_exec": "RB3TEFZC_Long",
    },
}

# Deliberately wide *provisional* biological screens, not dataset-defined
# reference intervals or standardized clinical exclusion criteria. Unscreened
# counts are reported separately. These rules can be adjusted in a later,
# documented harmonization step; we do NOT impute missing physiology.
SCREENS = {
    "Hb": (3, 25, "g/dL assumed; verify Tokyo lab codebook"),
    "MCV": (40, 140, "fL assumed; verify Tokyo lab codebook"),
    "HbA1c": (2, 25, "% reported by SPSS label"),
    "HbA1c_adj": (2, 25, "% reported by SPSS label; adjusted alternative"),
    "Glucose": (30, 700, "mg/dL reported by SPSS label"),
    "Insulin": (0, 1000, "uIU/mL label; strictly >0 for log compatibility"),
}
FAMILIES = {"M2": "MIDUS_CORE", "M3": "MIDUS_CORE",
            "MR1": "MIDUS_REFRESHER", "MR2": "MIDUS_REFRESHER",
            "MIDJA_1": "MIDJA", "MIDJA_2": "MIDJA"}
IDS = {"MIDUS_CORE": "M2ID", "MIDUS_REFRESHER": "MRID", "MIDJA": "MIDJA_IDs"}

# Sets always operate on real participant IDs only, in memory, and are never
# written to disk. This layout prevents accidentally mixing independent cohorts.
JOINT = [
    ("M2", "BIO", "G3", ("HbA1c", "Glucose", "Insulin")),
    ("M3", "BIO", "G3", ("HbA1c", "Glucose", "Insulin")),
    ("MR1", "BIO", "G3", ("HbA1c", "Glucose", "Insulin")),
    ("M2", "BIO", "G3+CESD_total", ("HbA1c", "Glucose", "Insulin", "CESD_total")),
    ("M3", "BIO", "G3+CESD_total", ("HbA1c", "Glucose", "Insulin", "CESD_total")),
    ("MR1", "BIO", "G3+CESD_total", ("HbA1c", "Glucose", "Insulin", "CESD_total")),
    ("M2", "BIO", "G3+CESD_somatic+affect", ("HbA1c", "Glucose", "Insulin", "CESD_somatic", "CESD_affect")),
    ("M3", "BIO", "G3+CESD_somatic+affect", ("HbA1c", "Glucose", "Insulin", "CESD_somatic", "CESD_affect")),
    ("MR1", "BIO", "G3+CESD_somatic+affect", ("HbA1c", "Glucose", "Insulin", "CESD_somatic", "CESD_affect")),
    ("MIDJA_1", "BIO", "H3+A1c", ("Hb", "RBC", "MCV", "HbA1c")),
    ("MIDJA_2", "BIO", "H3+A1c", ("Hb", "RBC", "MCV", "HbA1c")),
    ("MIDJA_1", "BIO", "H3+A1c_adjusted", ("Hb", "RBC", "MCV", "HbA1c_adj")),
    ("MIDJA_2", "BIO", "H3+A1c_adjusted", ("Hb", "RBC", "MCV", "HbA1c_adj")),
    ("MIDJA_1", "BIO", "H3+A1c+CESD_total", ("Hb", "RBC", "MCV", "HbA1c", "CESD_total")),
    ("MIDJA_2", "BIO", "H3+A1c+CESD_total", ("Hb", "RBC", "MCV", "HbA1c", "CESD_total")),
]

# Longitudinal/linked cohorts: each prerequisite is (wave, file-role, measure
# group or single field). Exact cohort intersection is recomputed from actual
# valid measurements, never inferred from row-count or ID-only overlap.
LINKED = [
    ("CORE_M2", "G3 + same-wave CESD + BTACT",
     [("M2", "BIO", "G3"), ("M2", "BIO", "CESD_total"), ("M2", "COG", "BTACT_comp")]),
    ("CORE_M3", "G3 + same-wave CESD + BTACT",
     [("M3", "BIO", "G3"), ("M3", "BIO", "CESD_total"), ("M3", "COG", "BTACT_comp")]),
    ("CORE_M2_M3", "G3 at both biomarker waves",
     [("M2", "BIO", "G3"), ("M3", "BIO", "G3")]),
    ("CORE_M2_M3", "M2 G3 -> M3 BTACT",
     [("M2", "BIO", "G3"), ("M3", "COG", "BTACT_comp")]),
    ("CORE_M2_M3", "M2 G3 + M2 BTACT -> M3 BTACT",
     [("M2", "BIO", "G3"), ("M2", "COG", "BTACT_comp"), ("M3", "COG", "BTACT_comp")]),
    ("CORE_M2_M3", "M2 G3 + M2 CESD -> M3 BTACT",
     [("M2", "BIO", "G3"), ("M2", "BIO", "CESD_total"), ("M3", "COG", "BTACT_comp")]),
    ("CORE_M2_M3", "G3 + BTACT measured at both waves",
     [("M2", "BIO", "G3"), ("M2", "COG", "BTACT_comp"),
      ("M3", "BIO", "G3"), ("M3", "COG", "BTACT_comp")]),
    ("REFRESHER_MR1", "MR1 G3 + CESD + BTACT",
     [("MR1", "BIO", "G3"), ("MR1", "BIO", "CESD_total"), ("MR1", "COG", "BTACT_comp")]),
    ("REFRESHER_MR1_MR2", "MR1 G3 -> MR2 BTACT",
     [("MR1", "BIO", "G3"), ("MR2", "COG", "BTACT_comp")]),
    ("REFRESHER_MR1_MR2", "MR1 G3 + MR1 BTACT -> MR2 BTACT",
     [("MR1", "BIO", "G3"), ("MR1", "COG", "BTACT_comp"), ("MR2", "COG", "BTACT_comp")]),
    ("MIDJA_1", "H3+A1c + same-wave CESD",
     [("MIDJA_1", "BIO", "H3+A1c"), ("MIDJA_1", "BIO", "CESD_total")]),
    ("MIDJA_2", "H3+A1c + same-wave CESD",
     [("MIDJA_2", "BIO", "H3+A1c"), ("MIDJA_2", "BIO", "CESD_total")]),
    ("MIDJA_1_2", "H3+A1c measured at both waves",
     [("MIDJA_1", "BIO", "H3+A1c"), ("MIDJA_2", "BIO", "H3+A1c")]),
    ("MIDJA_1_2", "H3+A1c + CESD at both waves",
     [("MIDJA_1", "BIO", "H3+A1c"), ("MIDJA_1", "BIO", "CESD_total"),
      ("MIDJA_2", "BIO", "H3+A1c"), ("MIDJA_2", "BIO", "CESD_total")]),
]


def classify(path: Path) -> tuple[str, str]:
    s = path.name.upper()
    if s.startswith("MIDJA"):
        wave = "MIDJA_2" if s.startswith(("MIDJA 2", "MIDJA_2")) else "MIDJA_1"
    elif s.startswith(("M2_", "M3_", "MR1_", "MR2_")):
        wave = s.split("_", 1)[0]
    else:
        return "UNKNOWN", "OTHER"
    if "BIO" in s:
        return wave, "BIO"
    if "BTACT" in s or "MOCA" in s or "COGN" in s:
        return wave, "COG"
    if "SURVEY" in s:
        return wave, "SURVEY"
    return wave, "OTHER"


def normalize_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        s = str(value).strip()
        return s or None
    if not math.isfinite(x):
        return None
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return str(value).strip() or None


def disclosed(n: int | None) -> str:
    if n is None:
        return "NOT_COMPUTED"
    return "SUPPRESSED_0_TO_10" if n <= 10 else str(n)


def safe_stats(series, valid_mask):
    """Aggregate distribution only when 11+ screened nonmissing records."""
    s = series.loc[valid_mask]
    if len(s) <= 10:
        return ("SUPPRESSED_0_TO_10",) * 4
    return tuple(f"{float(x):.5g}" for x in (s.mean(), s.median(), s.quantile(.05), s.quantile(.95)))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                   help="Project root with data/MIDUS/raw and External_Validation")
    a = p.parse_args()
    root = a.root.resolve()
    src = root / "data" / "MIDUS" / "raw"
    out = root / "External_Validation" / "Results"
    out.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import pyreadstat
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install using: python -m pip install pandas pyreadstat", file=sys.stderr)
        return 2

    if not src.is_dir():
        print(f"Dataset directory does not exist: {src}", file=sys.stderr)
        return 1
    files = sorted(x for x in src.rglob("*") if x.is_file() and x.suffix.lower() == ".sav")
    if not files:
        print(f"No SPSS files in {src}", file=sys.stderr)
        return 1

    slots: dict[tuple[str, str], Path] = {}
    errors = []
    for path in files:
        slot = classify(path)
        if slot[0] == "UNKNOWN" or slot[1] == "OTHER":
            errors.append(f"Unrecognized file, not processed: {path.name}")
            continue
        if slot in slots:
            errors.append(f"Duplicate file for {slot}; do not guess which version to read: {path.name}")
            continue
        slots[slot] = path

    measured: dict[tuple[str, str, str], set[str]] = {}
    # Use sets of valid IDs only in RAM. Never write these sets, individual rows,
    # unique subject tokens or joined participant-level data to output files.
    raw_present: dict[tuple[str, str], set[str]] = {}
    mapping = []
    coverage = []
    joint_coverage = []
    prospective = []
    all_valid = {}
    duplicate_id_file = []

    for (wave, role), path in slots.items():
        family = FAMILIES[wave]
        id_col = IDS[family]
        wanted = (BIO if role == "BIO" else COG if role == "COG" else {}).get(wave, {})
        try:
            _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
            names = set(meta.column_names)
            if id_col not in names:
                raise ValueError(f"Required participant ID {id_col} absent")
            usecols = [id_col] + [code for code in wanted.values() if code in names]
            # Explicitly retain SPSS user-missing definitions as NaN (default),
            # do not apply value labels, do not convert to strings or infer codes.
            df, _ = pyreadstat.read_sav(str(path), usecols=usecols,
                                         user_missing=False, apply_value_formats=False)
            ids = [normalize_id(v) for v in df[id_col]]
            non_null_ids = [v for v in ids if v is not None]
            unique_ids = set(non_null_ids)
            if len(unique_ids) != len(non_null_ids):
                # Safety: A single ID repeated across rows would make unions of
                # measurements from *different* records look like complete cases.
                duplicate_id_file.append(path.name)
                raise ValueError("Duplicate nonmissing participant IDs; results from this file withheld")
            raw_present[(wave, role)] = unique_ids
            row_n = len(df)
            missing_def = getattr(meta, "missing_ranges", None) or {}
            names_list = list(meta.column_names)
            labels_list = list(meta.column_labels or [])
            label_map = dict(zip(names_list, labels_list))
            for measure, code in wanted.items():
                exists = code in df.columns
                bounds = SCREENS.get(measure)
                mapping.append({
                    "wave": wave, "role": role, "file": path.name,
                    "measure": measure, "variable": code,
                    "spss_label": str(label_map.get(code, "") or "") if exists else "",
                    "status": "FOUND_UNVERIFIED" if exists else "MISSING_VARIABLE",
                    "screen": (f"{'>' if measure=='Insulin' else '>='}{bounds[0]}, <={bounds[1]}" if bounds else ("unit_unverified_nonmissing_only" if measure == "RBC" else "none")),
                    "unit_note": bounds[2] if bounds else ("Tokyo Lab RBC raw scale not verified; no numerical plausibility screen or unit conversion applied" if measure == "RBC" else "check coding/standardization in official codebook"),
                    "spss_user_missing_declared": "yes" if code in missing_def else "no_or_unavailable",
                })
                if not exists:
                    coverage.append({"wave": wave, "role": role, "measure": measure,
                        "variable": code, "status": "MISSING_VARIABLE", "rows": disclosed(row_n),
                        "nonmissing": "NOT_AVAILABLE", "screened_valid": "NOT_AVAILABLE",
                        "outside_screen": "NOT_AVAILABLE", "mean": "", "median": "", "p05": "", "p95": ""})
                    continue
                s = pd.to_numeric(df[code], errors="coerce")
                finite = s.notna() & s.map(lambda v: bool(math.isfinite(v)) if pd.notna(v) else False)
                screened = finite.copy()
                if bounds:
                    lo, hi, _ = bounds
                    screened &= (s > lo) if measure == "Insulin" else (s >= lo)
                    screened &= s <= hi
                found_ids = {pid for pid, ok in zip(ids, screened) if pid is not None and ok}
                measured[(wave, role, measure)] = found_ids
                all_valid[(wave, role, measure)] = screened
                # All counts of <=10 are explicitly suppressed by disclosure rule.
                mean, med, p05, p95 = safe_stats(s, screened)
                coverage.append({"wave": wave, "role": role, "measure": measure,
                    "variable": code, "status": "PROVISIONAL_SCREENED" if bounds else ("RBC_UNIT_UNVERIFIED_NONMISSING_ONLY" if measure == "RBC" else "NONMISSING_ONLY"),
                    "rows": disclosed(row_n), "nonmissing": disclosed(int(finite.sum())),
                    "screened_valid": disclosed(int(screened.sum())),
                    "outside_screen": disclosed(int((finite & ~screened).sum())),
                    "mean": mean, "median": med, "p05": p05, "p95": p95})
            print(f"  {wave:8s} {role:6s} | {len(wanted):2d} mapped measures | {row_n} records examined locally")
        except Exception as exc:
            errors.append(f"{wave} {role} {path.name}: {type(exc).__name__}: {exc}")
            # Withhold partial results if a file encounters a problem.
            raw_present.pop((wave, role), None)
            for k in list(measured):
                if k[0] == wave and k[1] == role:
                    del measured[k]
            for k in list(all_valid):
                if k[0] == wave and k[1] == role:
                    del all_valid[k]
            mapping = [r for r in mapping if not (r["wave"] == wave and r["role"] == role)]
            coverage = [r for r in coverage if not (r["wave"] == wave and r["role"] == role)]
            print(f"  ERROR {wave} {role}: {exc}", file=sys.stderr)
        finally:
            if "df" in locals():
                del df

    def create_joint(wave, role, title, req):
        if not all((wave, role, x) in measured for x in req):
            return None
        s = set.intersection(*(measured[(wave, role, x)] for x in req))
        measured[(wave, role, title)] = s
        return s

    for wave, role, title, req in JOINT:
        got = create_joint(wave, role, title, req)
        joint_coverage.append({"wave": wave, "role": role, "panel": title,
                               "requirements": "+".join(req),
                               "valid_joint_n": disclosed(len(got)) if got is not None else "NOT_AVAILABLE",
                               "status": "PROVISIONAL" if got is not None else "MISSING_REQUIRED_VARIABLE"})

    for wave in ("M2", "M3", "MR1"):
        for panel in ("G3", "G3+CESD_total"):
            components = [(wave, "BIO", panel), (wave, "COG", "BTACT_comp")]
            title = f"{panel}+BTACT_comp"
            if all(q in measured for q in components):
                s = set.intersection(*(measured[q] for q in components))
                measured[(wave, "BIO+COG", title)] = s
                count, status = disclosed(len(s)), "PROVISIONAL"
            else:
                count, status = "NOT_AVAILABLE", "MISSING_REQUIRED_VARIABLE"
            joint_coverage.append({"wave": wave, "role": "BIO+COG", "panel": title,
                "requirements": "+".join(f"{role}:{part}" for _, role, part in components),
                "valid_joint_n": count, "status": status})

    for cohort, name, parts in LINKED:
        # Verify that requested wave membership is in the same participant-ID
        # namespace before computing any cohort intersections.
        family_set = {FAMILIES[w] for w, _role, _marker in parts}
        if len(family_set) != 1:
            errors.append(f"Invalid cross-family intersection request: {cohort}/{name}")
            continue
        if all(p in measured for p in parts):
            subset = set.intersection(*(measured[p] for p in parts))
            status = "PROVISIONAL_COMPLETE_CASES"
            n = disclosed(len(subset))
        else:
            status, n = "MISSING_REQUIRED_VARIABLE", "NOT_AVAILABLE"
        prospective.append({"cohort": cohort, "analysis_candidate": name,
                            "requirements": " & ".join(f"{w}.{r}.{v}" for w, r, v in parts),
                            "joint_n": n, "status": status})

    write_csv(out / "02_variable_map.csv", mapping,
        ["wave", "role", "file", "measure", "variable", "spss_label", "status", "screen",
         "unit_note", "spss_user_missing_declared"])
    write_csv(out / "02_measurement_coverage.csv", coverage,
        ["wave", "role", "measure", "variable", "status", "rows", "nonmissing", "screened_valid",
         "outside_screen", "mean", "median", "p05", "p95"])
    write_csv(out / "02_joint_coverage.csv", joint_coverage,
        ["wave", "role", "panel", "requirements", "valid_joint_n", "status"])
    write_csv(out / "02_longitudinal_coverage.csv", prospective,
        ["cohort", "analysis_candidate", "requirements", "joint_n", "status"])

    lines = [
        "# Script 02 — MIDUS measurement and longitudinal coverage audit", "",
        "**Scope:** Local SPSS reads of explicit candidate variables; aggregate-only outputs. No subject records exported.",
        "**Status:** Feasibility audit only. No complete clinical harmonization, frozen NHANES transfer, modelling, or causal inference.",
        "**Disclosure:** All computed table cells with N <= 10, including zero, are suppressed.", "",
        "## Candidate measurements, by study", "",
        "| Wave | Profile | Complete physiology | Physiology + CES-D | Physiology + BTACT |",
        "|---|---|---:|---:|---:|",
    ]
    for wave in ("M2", "M3", "MR1", "MIDJA_1", "MIDJA_2"):
        is_us = wave in ("M2", "M3", "MR1")
        profile = "G3 (HbA1c, glucose, insulin)" if is_us else "H3+A1c (Hb, RBC, MCV, HbA1c)"
        main_title = "G3" if is_us else "H3+A1c"
        dep_title = "G3+CESD_total" if is_us else "H3+A1c+CESD_total"
        def pick(panel):
            return next((r["valid_joint_n"] for r in joint_coverage
                         if r["wave"] == wave and r["panel"] == panel), "NOT_AVAILABLE")
        bcomp = pick("G3+BTACT_comp") if is_us else "NO_PERFORMANCE_COGNITION_FILE"
        lines.append(f"| {wave} | {profile} | {pick(main_title)} | {pick(dep_title)} | {bcomp} |")
    lines += [
        "", "## Additional within-wave profiles", "",
        "| Wave | Panel | Valid participant N |", "|---|---|---:|",
    ]
    for r in joint_coverage:
        if r["panel"] not in ("G3", "G3+CESD_total", "H3+A1c", "H3+A1c+CESD_total", "G3+BTACT_comp"):
            lines.append(f"| {r['wave']} | {r['panel']} | {r['valid_joint_n']} |")
    lines += ["", "**MIDJA panel note:** H3+A1c counts reflect RBC numeric availability (units unverified), not certified physiological validity. Do not use these raw RBC values for modelling until unit harmonization.",
              "", "## Longitudinal/linkage complete-case candidates", "",
              "| Cohort | Prerequisites | Valid participant N |", "|---|---|---:|"]
    for r in prospective:
        lines.append(f"| {r['cohort']} | {r['analysis_candidate']} | {r['joint_n']} |")
    lines += [
        "", "## Important interpretation limits", "",
        "- These are **valid-measurement ID intersections**, not an approved analytical cohort. No exclusions for medications, pregnancy, diagnoses, fasting verification, age, sampling design, or time gaps have been applied.",
        "- Physiology uses provisional bounds where units are reasonably identifiable. Both the nonmissing count and the screened-valid count appear in `02_measurement_coverage.csv`.",
        "- **MIDJA RBC correction:** the original 1-9 numeric screen was inappropriate without confirmation of Japanese RBC units. RBC is now audited on finite nonmissing data only, with **no unit conversion**. Japanese panel counts are **measurement-available**, NOT clinically validated; do not fit models using this raw RBC scale until laboratory units and sentinel values are verified.",
        "- SPSS user-missing values are parsed as NaN by `pyreadstat` (`user_missing=False`); unflagged sentinel codes require separate codebook review.",
        "- For insulin, positive screened values are **log-transform-compatible**; no log transformation or model transfer has occurred.",
        "- US waves provide a possible glycemic triad, not the full frozen seven-marker NHANES A/G space. Japan provides a distinct provisional Hb/RBC/MCV/HbA1c panel, not an exact NHANES transfer.",
        "- `CESD_total`, `CESD_somatic`, and `CESD_affect` are separate measures. CES-D is not PHQ-9; depressive affect is not interchangeable with PHQ-derived CogAff.",
        "- BTACT score standardizations vary by wave and sample (national-only, full-sample, longitudinal). Explicit mappings are candidates only; audit the source scoring and measurement invariance before calculating change.",
        "- MIDJA lacks downloaded performance-based cognitive assessments. A `Cognition Control` questionnaire response is not interchangeable with BTACT.",
        "- Two observations per person do not in themselves identify a causal effect. Check visit dates and temporal ordering before building longitudinal equations.",
        "- `HbA1c_adj` in MIDJA is a separately reported adjusted assay; no substitution is made without source documentation.",
        "", "## Diagnostic details", "",
        f"- SPSS files staged: {len(files)}; recognized cohort/role files: {len(slots)}.",
        f"- Mapped variables reviewed: {len(mapping)}; linkage candidates evaluated: {len(prospective)}.",
        f"- Files with duplicate IDs (unsafe for row-level intersections): {len(duplicate_id_file)}.",
        f"- Errors: {len(errors)}.",
    ]
    if errors:
        lines += ["", "### Errors requiring resolution", ""]
        lines.extend(f"- {str(e).replace('|', '/')}" for e in errors)
    lines += ["", "**Next scientific gate:** Confirm assay units, sentinel codes, questionnaire scoring and timing against codebooks before attempting an adapted decomposition or frozen transfer.", ""]
    (out / "02_readout.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"SCRIPT 02 | MIDUS measurement coverage (RBC-scale correction) | {len(slots)} study files | {len(errors)} errors")
    print("  MIDJA RBC: finite/nonmissing only; units unverified, no conversion; diagnostic counts only")
    for r in prospective:
        if r["analysis_candidate"] in ("G3 + BTACT measured at both waves", "MR1 G3 + MR1 BTACT -> MR2 BTACT", "H3+A1c measured at both waves"):
            print(f"  {r['cohort']:17s} {r['analysis_candidate']}: {r['joint_n']}")
    print(f"OUTPUT: {out / '02_readout.md'}")
    print("        02_variable_map.csv | 02_measurement_coverage.csv | 02_joint_coverage.csv | 02_longitudinal_coverage.csv")
    # Errors cause nonzero exit, but the diagnostic readout is still written.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
