#!/usr/bin/env python3
"""Script 00 | MIDUS-family staged dataset inventory (metadata only).

Inputs:
    <project root>/data/MIDUS/raw/*.[sS][aA][vV]
Outputs:
    <project root>/External_Validation/Results/00_inventory.md
    <project root>/External_Validation/Results/00_dataset_manifest.json

This script reads NO participant observations. It lists files, waves, project
roles, schema sizes, available join-key *column names*, and dataset coverage.
Script 01 performs full variable-label and aggregate participant-linkage audits.
KNHANES is deliberately not part of the current workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ID_KEYS = {"MIDUS_CORE": "M2ID", "MIDUS_REFRESHER": "MRID", "MIDJA": "MIDJA_IDs"}
WAVES = ["M2", "M3", "MR1", "MR2", "MIDJA_1", "MIDJA_2"]
WAVE_ROLES = {
    "M2": ("SURVEY", "BIO", "COG"),
    "M3": ("SURVEY", "BIO", "COG"),
    "MR1": ("SURVEY", "BIO", "COG"),
    "MR2": ("SURVEY", "COG"),
    "MIDJA_1": ("SURVEY", "BIO"),
    "MIDJA_2": ("SURVEY", "BIO"),
}


def classify(path: Path) -> tuple[str, str, str]:
    name = path.name.upper()
    if name.startswith("MIDJA"):
        family = "MIDJA"
        wave = "MIDJA_2" if ("MIDJA 2" in name or "MIDJA_2" in name) else "MIDJA_1"
    elif name.startswith("M2_"):
        family, wave = "MIDUS_CORE", "M2"
    elif name.startswith("M3_"):
        family, wave = "MIDUS_CORE", "M3"
    elif name.startswith("MR1_"):
        family, wave = "MIDUS_REFRESHER", "MR1"
    elif name.startswith("MR2_"):
        family, wave = "MIDUS_REFRESHER", "MR2"
    else:
        family, wave = "UNKNOWN", "UNKNOWN"
    if "BIO" in name:
        role = "BIO"
    elif "BTACT" in name or "COGN" in name or "MOCA" in name:
        role = "COG"
    elif "SURVEY" in name:
        role = "SURVEY"
    else:
        role = "OTHER"
    return family, wave, role


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--sha256", action="store_true", help="hash local source files without exporting data")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data" / "MIDUS" / "raw"
    results = root / "External_Validation" / "Results"
    results.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        print(f"ERROR: missing dataset folder: {source}", file=sys.stderr)
        return 1

    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".sav")
    if not files:
        print(f"ERROR: no staged SPSS .sav files under {source}", file=sys.stderr)
        return 1
    try:
        import pyreadstat
    except ImportError:
        print("Install metadata reader: python -m pip install pyreadstat", file=sys.stderr)
        return 2

    rows = []
    errors = []
    for path in files:
        family, wave, role = classify(path)
        info = {
            "file": path.name,
            "relative_path": str(path.relative_to(root)),
            "family": family,
            "wave": wave,
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path) if args.sha256 else None,
            "schema_status": "pending",
            "column_count": None,
            "row_count_metadata": None,
            "id_column": None,
            "first_columns": [],
        }
        try:
            _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
            cols = list(meta.column_names)
            wanted = ID_KEYS.get(family, "")
            info.update({
                "schema_status": "ok",
                "column_count": len(cols),
                "row_count_metadata": getattr(meta, "number_rows", None),
                "id_column": next((c for c in cols if c.casefold() == wanted.casefold()), None) if wanted else None,
                "first_columns": cols[:12],
            })
        except Exception as exc:
            info["schema_status"] = "error"
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(info)

    coverage = {}
    for wave in WAVES:
        matching = [x for x in rows if x["wave"] == wave]
        roles = {x["role"] for x in matching if x["schema_status"] == "ok"}
        coverage[wave] = {
            "expected_roles": list(WAVE_ROLES[wave]),
            "observed_roles": sorted(roles),
            "missing_roles": [r for r in WAVE_ROLES[wave] if r not in roles],
            "file_count": len(matching),
        }
    duplicates = [f"{w}:{r} ({n} files)" for (w, r), n in
                  sorted(Counter((x["wave"], x["role"]) for x in rows).items()) if n > 1]

    manifest = {
        "script": "00_dataset_inventory.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "source": str(source),
        "scope": "MIDUS and MIDJA only; KNHANES will be added later",
        "privacy": "schema metadata only; no participant observations or IDs read",
        "file_count": len(rows),
        "inspected_schema_count": sum(x["schema_status"] == "ok" for x in rows),
        "files": rows,
        "coverage": coverage,
        "duplicates": duplicates,
        "errors": errors,
    }
    (results / "00_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    md = [
        "# Script 00 — MIDUS local file and schema inventory", "",
        "**Scope:** American MIDUS Core/Refresher + Japanese MIDJA; KNHANES deferred.",
        "**Method:** SPSS metadata only. No participant rows read, no variable harmonization claimed.", "",
        f"- SPSS files staged: **{len(rows)}**",
        f"- Schemas inspected: **{manifest['inspected_schema_count']}**", "",
        "## Study coverage", "",
        "| Wave | Survey | Biomarker | Cognitive | Notes |",
        "|---|---|---|---|---|",
    ]
    for wave, status in coverage.items():
        cells = ["present" if r in status["observed_roles"] else
                 ("missing" if r in status["expected_roles"] else "not part of downloaded wave")
                 for r in ("SURVEY", "BIO", "COG")]
        md.append(f"| {wave} | {' | '.join(cells)} | {status['file_count']} files |")
    md += ["", "## File inventory", "",
           "| Wave | Role | File | Columns | Participant ID field | Status |",
           "|---|---|---|---:|---|---|"]
    for x in rows:
        md.append(f"| {x['wave']} | {x['role']} | `{x['file']}` | "
                  f"{x['column_count'] if x['column_count'] is not None else '—'} | "
                  f"{('`' + x['id_column'] + '`') if x['id_column'] else 'not detected'} | "
                  f"{x['schema_status']} |")
    md += ["", "## Next action", "",
           "Run `01_midus_schema_linkage_audit.py` to inspect complete SPSS variable labels,",
           "candidate blood markers, depressive/cognitive phenotypes and aggregate ID overlap.",
           "Presence of files or linkage IDs is **not** evidence of seven-marker complete cases."]
    if duplicates:
        md += ["", "## Duplicate role warnings", ""] + [f"- {d}" for d in duplicates]
    if errors:
        md += ["", "## Schema errors", ""] + [f"- {e['file']}: {e['error']}" for e in errors]
    (results / "00_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"SCRIPT 00 | MIDUS/MIDJA file inventory | {len(rows)} SPSS files")
    for wave in WAVES:
        s = coverage[wave]
        print(f"  {wave:8s} | staged roles: {', '.join(s['observed_roles']) or 'none'}"
              f" | missing: {', '.join(s['missing_roles']) or 'none'}")
    print(f"OUTPUTS: {results / '00_inventory.md'} | {results / '00_dataset_manifest.json'}")
    if errors:
        print(f"ERRORS: {len(errors)} file schemas could not be inspected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
