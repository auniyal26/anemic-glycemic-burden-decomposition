#!/usr/bin/env python3
"""Script 01 | MIDUS/MIDJA label, biomarker compatibility, and ID-linkage audit.

Local-only. Reads SPSS schemas and, for aggregate linkage counts, participant ID
columns only. Never writes participant-level data, identifiers, or measurements.

Install: python -m pip install pyreadstat
Run from project root: python External_Validation/01_midus_schema_linkage_audit.py
Optional: --schema-only (does not read IDs)

Outputs under External_Validation/Results:
  01_variable_catalog.csv       full SPSS metadata (variable labels, NOT rows)
  01_candidate_variables.csv    search hits requiring scientific verification
  01_linkage_counts.csv         aggregate ID overlap (small cells suppressed)
  01_audit.json                 machine-readable metadata report
  01_readout.md                 human-readable report

Important: candidate matches are *not* harmonized variables, not codebook-verified
measurements, and never a claim that frozen NHANES maps can be transferred.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

DATASET_KEYS = {"MIDUS_CORE": "M2ID", "MIDUS_REFRESHER": "MRID", "MIDJA": "MIDJA_IDs"}
# These are search queries over SPSS variable labels and names, not validated aliases.
SEARCH_RULES = {
    "Hb": [r"\b(?:ha?emoglobin|hgb)\b"],
    "RBC": [r"\brbc(?:\s+(?:count|number))?\b", r"\berythrocyte(?:s)?\s+(?:count|number)\b", r"\bred(?:\s+blood)?\s+cell(?:s)?\s+(?:count|number)\b"],
    "MCV": [r"\bmcv\b", r"mean\s+corpuscular\s+volume"],
    "RDW": [r"\brdw\b", r"red\s+(?:blood\s+)?cell\s+distribution\s+width"],
    "HbA1c": [r"\b(?:hba1c|a1c|glycated\s+ha?emoglobin|glycosylated\s+ha?emoglobin)\b"],
    "Glucose": [r"\bglucose\b", r"\bglyc(?:a|e)emia\b"],
    "Insulin": [r"\binsulin\b"],
    "CESD": [r"\bces[-_ ]?d\b", r"center\s+for\s+epidemiolog(?:ic|ical)\s+studies\s+depression"],
    "Depression": [r"\bdepress(?:ion|ed|ive)\b", r"\bmasq\b", r"mood\s+and\s+anxiety\s+symptom"],
    "Cognition": [r"\b(?:btact|moca|cognit(?:ive|ion))\b", r"executive\s+function", r"episodic\s+memory", r"working\s+memory"],
    "Age": [r"\bage\b", r"age\s+at"],
    "Sex": [r"\bsex\b", r"\bgender\b"],
    "BMI": [r"\bbmi\b", r"body\s+mass\s+index"],
}
RX = {k: [re.compile(t, re.I) for t in rules] for k, rules in SEARCH_RULES.items()}
PHYSIOLOGY = ["Hb", "RBC", "MCV", "RDW", "HbA1c", "Glucose", "Insulin"]


def classify(p: Path) -> tuple[str, str, str]:
    s = p.name.upper()
    if s.startswith("MIDJA"):
        family = "MIDJA"
        wave = "MIDJA_2" if ("MIDJA 2" in s or "MIDJA_2" in s) else "MIDJA_1"
    elif s.startswith("MR2_"):
        family, wave = "MIDUS_REFRESHER", "MR2"
    elif s.startswith("MR1_"):
        family, wave = "MIDUS_REFRESHER", "MR1"
    elif s.startswith("M3_"):
        family, wave = "MIDUS_CORE", "M3"
    elif s.startswith("M2_"):
        family, wave = "MIDUS_CORE", "M2"
    else:
        family, wave = "UNKNOWN", "UNKNOWN"
    if "BIO" in s:
        kind = "BIO"
    elif "BTACT" in s or "COGN" in s or "MOCA" in s:
        kind = "COG"
    elif "SURVEY" in s:
        kind = "SURVEY"
    else:
        kind = "OTHER"
    return family, wave, kind


def matches(name: str, label: str) -> list[str]:
    query = f"{name} {label}"
    found = [group for group, rules in RX.items() if any(r.search(query) for r in rules)]
    # Avoid classifying HbA1c variables as simple Hb measurements.
    if "HbA1c" in found and "Hb" in found:
        found.remove("Hb")
    return found


def canonical_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        s = str(value).strip()
        return s or None
    if not math.isfinite(f):
        return None
    if f.is_integer() and abs(f) < 10**15:
        return str(int(f))
    return str(value).strip() or None


def masked(n: int | None) -> str:
    """MIDUS portal prohibits publishing tabulations with 1-10 observations."""
    if n is None:
        return "not_computed"
    return "SUPPRESSED_0_TO_10" if n <= 10 else str(n)


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--schema-only", action="store_true", help="skip participant-ID reads and overlap counts")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data" / "MIDUS" / "raw"
    result_dir = root / "External_Validation" / "Results"
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyreadstat
    except ImportError:
        print("Missing dependency. Run: python -m pip install pyreadstat", file=sys.stderr)
        return 2

    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".sav") if source.exists() else []
    if not files:
        print(f"No .sav files found: {source}", file=sys.stderr)
        return 1
    print(f"SCRIPT 01 | {len(files)} MIDUS/MIDJA SPSS files | all variable labels")
    catalog, candidates, inventory, failures = [], [], [], []
    groups: dict[tuple[str, str, str], dict] = {}
    id_sets: dict[tuple[str, str, str], set[str]] = {}

    for path in files:
        family, wave, kind = classify(path)
        relpath = str(path.relative_to(root))
        try:
            _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
            names = list(meta.column_names)
            labels = list(meta.column_labels or [])
            if len(labels) < len(names):
                labels += [""] * (len(names) - len(labels))
            id_target = DATASET_KEYS.get(family, "")
            id_col = next((n for n in names if n.casefold() == id_target.casefold()), "")
            data_count = getattr(meta, "number_rows", None)
            item = {"file": path.name, "path": relpath, "family": family, "wave": wave,
                    "kind": kind, "column_count": len(names), "metadata_row_count": data_count,
                    "person_id_column": id_col, "matched_variable_count": 0,
                    "candidate_groups": [], "id_status": "schema_only" if args.schema_only else "pending"}
            this_categories = set()
            for col, label in zip(names, labels):
                label = "" if label is None else str(label)
                got = matches(col, label)
                entry = {"file": path.name, "wave": wave, "kind": kind,
                         "variable": col, "label": label,
                         "measurement_type": str((getattr(meta, "variable_measure", None) or {}).get(col, "")),
                         "storage_type": str((getattr(meta, "original_variable_types", None) or {}).get(col, ""))}
                catalog.append(entry)
                for group in got:
                    candidates.append({**entry, "candidate_group": group,
                                       "warning": "UNVERIFIED: check codebook, units, timing, missing codes"})
                    this_categories.add(group)
            item["matched_variable_count"] = len(set(x["variable"] for x in candidates if x["file"] == path.name))
            item["candidate_groups"] = sorted(this_categories)
            if not args.schema_only and id_col:
                # Read only the person-ID column; never export IDs.
                df, _ = pyreadstat.read_sav(str(path), usecols=[id_col], apply_value_formats=False)
                raw_ids = [canonical_id(v) for v in df[id_col]]
                seen = [s for s in raw_ids if s is not None]
                key = (family, wave, kind)
                if key in id_sets:
                    raise ValueError(f"Two files map to the same wave/kind: {key}; audit required")
                id_sets[key] = set(seen)
                item["id_status"] = "read_locally"
                item["n_unique_id"] = len(id_sets[key])
                item["n_null_id"] = len(raw_ids) - len(seen)
                item["n_duplicate_id_rows"] = len(seen) - len(id_sets[key])
                del df, raw_ids, seen
            elif not id_col:
                item["id_status"] = "missing_person_key"
            inventory.append(item)
            groups[(family, wave, kind)] = item
            print(f"  {wave:8s} {kind:6s} {len(names):5d} variables | {len(this_categories)} candidate groups | {item['id_status']}")
        except Exception as exc:
            failures.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR {path.name}: {exc}", file=sys.stderr)

    linkage_rows = []
    # Within-wave BIO/SURVEY/COG, across-wave matched BIO and COG, and baseline BIO->follow-up COG.
    combos = [
        ("MIDUS_CORE", "M2", "BIO", "M2", "SURVEY"),
        ("MIDUS_CORE", "M2", "BIO", "M2", "COG"),
        ("MIDUS_CORE", "M2", "BIO", "M3", "BIO"),
        ("MIDUS_CORE", "M2", "BIO", "M3", "COG"),
        ("MIDUS_CORE", "M2", "COG", "M3", "COG"),
        ("MIDUS_CORE", "M3", "BIO", "M3", "SURVEY"),
        ("MIDUS_CORE", "M3", "BIO", "M3", "COG"),
        ("MIDUS_REFRESHER", "MR1", "BIO", "MR1", "SURVEY"),
        ("MIDUS_REFRESHER", "MR1", "BIO", "MR1", "COG"),
        ("MIDUS_REFRESHER", "MR1", "BIO", "MR2", "COG"),
        ("MIDUS_REFRESHER", "MR1", "COG", "MR2", "COG"),
        ("MIDJA", "MIDJA_1", "BIO", "MIDJA_1", "SURVEY"),
        ("MIDJA", "MIDJA_1", "BIO", "MIDJA_2", "BIO"),
        ("MIDJA", "MIDJA_1", "BIO", "MIDJA_2", "SURVEY"),
        ("MIDJA", "MIDJA_2", "BIO", "MIDJA_2", "SURVEY"),
    ]
    for family, w1, t1, w2, t2 in combos:
        k1, k2 = (family, w1, t1), (family, w2, t2)
        if k1 not in id_sets or k2 not in id_sets:
            linkage_rows.append({"family": family, "source": f"{w1}_{t1}", "target": f"{w2}_{t2}",
                                 "shared_ids": "not_computed", "note": "missing source, target or schema-only"})
            continue
        n = len(id_sets[k1] & id_sets[k2])
        linkage_rows.append({"family": family, "source": f"{w1}_{t1}", "target": f"{w2}_{t2}",
                             "shared_ids": masked(n), "note": "aggregate count; records not exported"})
    # Core triple overlaps, cautious about ID-only overlap not complete physiology/cognition.
    triples = [
        ("MIDUS_CORE", ("M2", "BIO"), ("M2", "COG"), ("M3", "COG")),
        ("MIDUS_CORE", ("M2", "BIO"), ("M3", "BIO"), ("M3", "COG")),
        ("MIDUS_REFRESHER", ("MR1", "BIO"), ("MR1", "COG"), ("MR2", "COG")),
    ]
    for family, a, b, c in triples:
        keys = [(family, *v) for v in (a, b, c)]
        if all(k in id_sets for k in keys):
            n = len(id_sets[keys[0]] & id_sets[keys[1]] & id_sets[keys[2]])
            val, note = masked(n), "three-way aggregate; not complete-case N"
        else:
            val, note = "not_computed", "missing component"
        linkage_rows.append({"family": family,
                             "source": "+".join(f"{x[0]}_{x[1]}" for x in (a, b)),
                             "target": f"{c[0]}_{c[1]}", "shared_ids": val, "note": note})

    save_csv(result_dir / "01_variable_catalog.csv", catalog,
             ["file", "wave", "kind", "variable", "label", "measurement_type", "storage_type"])
    save_csv(result_dir / "01_candidate_variables.csv", candidates,
             ["file", "wave", "kind", "candidate_group", "variable", "label", "measurement_type", "storage_type", "warning"])
    save_csv(result_dir / "01_linkage_counts.csv", linkage_rows,
             ["family", "source", "target", "shared_ids", "note"])

    # Publish only aggregate counts that satisfy the source's disclosure restriction.
    safe_inventory = []
    for item in inventory:
        x = dict(item)
        for key in ("metadata_row_count", "n_unique_id", "n_null_id", "n_duplicate_id_rows"):
            if key in x:
                v = x[key]
                x[key] = masked(v) if isinstance(v, int) else ("not_reported" if v is None else v)
        safe_inventory.append(x)
    payload = {"script": "01_midus_schema_linkage_audit.py", "input_root": str(source),
               "files_inspected": len(inventory), "candidate_matches": len(candidates),
               "disclosure_rule": "counts 0 to 10 suppressed (MIDUS portal policy)",
               "candidate_warning": "variable-label searches, NOT verified harmonized measures",
               "inventory": safe_inventory, "linkage": linkage_rows, "errors": failures}
    (result_dir / "01_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = ["# Script 01 — MIDUS/MIDJA label and linkage audit", "",
             "**Scope:** All SPSS variable labels + local ID-column joins. No individual records exported.",
             "**Interpretation:** Hits are provisional candidates, NOT validated measurements. Review codebooks and units.",
             "**Privacy:** Source-policy cell counts <=10 suppressed, including zero.", "",
             "## File coverage", "",
             "| Wave | Kind | Variables | Candidate marker groups | Unique IDs |",
             "|---|---|---:|---|---:|"]
    for item in safe_inventory:
        available = ", ".join(k for k in PHYSIOLOGY if k in item["candidate_groups"]) or "none matched"
        lines.append(f"| {item['wave']} | {item['kind']} | {item['column_count']} | {available} | {item.get('n_unique_id','not_read')} |")
    lines += ["", "## Candidate physiological variables by file", ""]
    byfile = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        if row["candidate_group"] in PHYSIOLOGY:
            byfile[row["file"]][row["candidate_group"]].append(row)
    for item in inventory:
        if item["kind"] != "BIO":
            continue
        lines.append(f"### {item['file']}")
        if not byfile[item['file']]:
            lines.append("No physiological-label matches. Check codebook before concluding absence.")
        else:
            for cat in PHYSIOLOGY:
                if cat not in byfile[item["file"]]:
                    continue
                sample = byfile[item["file"]][cat]
                parts = [f"`{x['variable']}` ({x['label'][:100].replace('|','/').replace(chr(10),' ')})" for x in sample[:6]]
                lines.append(f"- **{cat}** ({len(sample)} possible): " + "; ".join(parts) + ("; ..." if len(sample) > 6 else ""))
        lines.append("")
    lines += ["## Candidate depression and cognitive outcomes", "",
              "These are label-based leads, not confirmed equivalent scales across waves.", ""]
    for item in inventory:
        hits = [r for r in candidates if r["file"] == item["file"]
                and r["candidate_group"] in {"CESD", "Depression", "Cognition"}]
        if not hits:
            continue
        bygroup = defaultdict(list)
        for r in hits:
            bygroup[r["candidate_group"]].append(r)
        lines.append(f"### {item['wave']} — {item['kind']}")
        for group in ("CESD", "Depression", "Cognition"):
            vals = bygroup.get(group, [])
            if vals:
                ex = "; ".join(f"`{r['variable']}` ({r['label'][:75].replace('|','/')})" for r in vals[:4])
                lines.append(f"- **{group}:** {len(vals)} candidate variables; {ex}" + ("; ..." if len(vals) > 4 else ""))
        lines.append("")
    lines += ["## Aggregate participant linkage (not complete-case samples)", "",
             "| Family | From | To | Shared participant IDs |", "|---|---|---|---:|"]
    for row in linkage_rows:
        lines.append(f"| {row['family']} | {row['source']} | {row['target']} | {row['shared_ids']} |")
    lines += ["", "## Decision gates", "",
              "1. Verify every biomarker variable against the downloaded SPSS/PDF codebook, including laboratory units, fasting status, timing and missing-value codes.",
              "2. Determine seven-biomarker complete cases on-device; ID overlap alone is not enough.",
              "3. Keep the original MIDUS core, Refresher and Japanese MIDJA cohorts distinct.",
              "4. For longitudinal models confirm actual visit time/order and repeated measurements; don't assume biomarker and cognitive measurement dates coincide.",
              "5. Report exact seven-marker transfer only if all original markers can be harmonized. Otherwise prespecify reduced-variable tests and call them adaptations.",
              "6. Preserve frozen NHANES scripts and maps. Do not refit them on external outcomes."]
    if failures:
        lines += ["", "## Files needing attention", ""] + [f"- {x['file']}: {x['error']}" for x in failures]
    (result_dir / "01_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OUTPUTS: {result_dir / '01_readout.md'} | {result_dir / '01_candidate_variables.csv'}")
    print(f"          {result_dir / '01_linkage_counts.csv'} | {result_dir / '01_variable_catalog.csv'}")
    if failures:
        print(f"CAUTION: {len(failures)} input files failed; see 01_readout.md")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
