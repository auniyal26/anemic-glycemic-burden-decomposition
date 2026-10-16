#!/usr/bin/env python3
"""Safely repair GitHub math formatting and navigation in Logs/Experiment_Log.md.

NO research text is deleted, summarized, reworded, or numerically changed.
Only 26 math delimiters are normalized, and navigational lines are inserted.

Run from the root of the research Git repository:
    python fix_experiment_log.py
Then inspect:
    git diff -- Logs/Experiment_Log.md
"""
from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

EXPECTED_GIT_BLOB = "1a523b4d5cab8e2e491cf39ad862af76d7833978"
RELATIVE_FILE = Path("Logs") / "Experiment_Log.md"

ANCHORS = [
    ("log-aug-20", "# Research Log — 20 August 2026"),
    ("log-aug-21", "# Research Log — 21 August 2026"),
    ("log-sep-14", "# Research Log — 14 September 2026"),
    ("log-sep-16", "# Research Log — 16 September 2026"),
    ("log-backfill", "# Research Log Backfill — Work Completed After Script 24"),
    ("log-sep-17", "# 17 September — Synthetic Longitudinal Ground-Truth Experiment"),
    ("log-sep-21", "# 21 September — Formal Lock-In / Falsification Suite"),
    ("log-sep-24", "# 24 September — Mechanistic / Interpretability Audit"),
    ("log-sep-25", "# 25 Sep 2026 — Research Log"),
    ("log-sep-30", "## 30 Sep 2026 — Shared/Private Decomposition + Phenotype-Map Transfer"),
    ("log-oct-02", "## 02 Oct 2026 — Bidirectional Gradient Audit + Functional Dependence"),
    ("log-oct-03", "# 03 October 2026 — Phenotype-Map Stability, Reverse Inference, Longitudinal Proof-of-Concept, and MoM Closeout"),
    ("log-oct-12", "## Research Log — 12 October 2026"),
    ("log-oct-13", "# Research Log — 13 October 2026"),
    ("log-oct-14", "## 14 October 2026 — Research Log: Scripts 96–100"),
    ("log-oct-15", "## 2026-10-15 — External Validation and Causal Identification"),
]

GUIDE = """> **Research integrity:** This is the complete experimental record. No findings, estimates,
> methodology, limitations, negative results, or interpretations have been removed.
>
> **Reading path:** NHANES feasibility → seven-marker decomposition/reconstruction →
> phenotype mapping → temporal transfer → encoding/reverse-inference limits →
> external validation and causal-identification audit.

### Quick navigation

| Stage | Research entries |
| --- | --- |
| Initial feasibility | [20 Aug](#log-aug-20) · [21 Aug](#log-aug-21) |
| Representation and replication | [14 Sep](#log-sep-14) · [16 Sep](#log-sep-16) · [Scripts 25+ backfill](#log-backfill) |
| Simulation and falsification | [17 Sep](#log-sep-17) · [21 Sep](#log-sep-21) |
| Decomposition interpretation | [24 Sep](#log-sep-24) · [25 Sep](#log-sep-25) · [30 Sep](#log-sep-30) |
| Bidirectional / temporal analysis | [2 Oct](#log-oct-02) · [3 Oct](#log-oct-03) |
| Encoding and phenotype equations | [12 Oct](#log-oct-12) · [13 Oct](#log-oct-13) · [14 Oct](#log-oct-14) |
| External / causal stress tests | [15 Oct](#log-oct-15) |

---

"""


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def main() -> None:
    path = Path.cwd() / RELATIVE_FILE
    if not path.is_file():
        raise SystemExit(f"Not found: {path}\nRun this script from the repository root.")
    raw = path.read_bytes()
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    clean_raw = raw[len(bom):]
    eol = "\r\n" if b"\r\n" in clean_raw else "\n"
    normalized = clean_raw.replace(b"\r\n", b"\n")
    sha = git_blob_sha(normalized)
    if sha != EXPECTED_GIT_BLOB:
        raise SystemExit(
            "The log differs from the GitHub snapshot reviewed on 16 Oct 2026.\n"
            f"Expected blob: {EXPECTED_GIT_BLOB}\nFound: {sha}\n"
            "No changes made; re-audit the latest log before applying."
        )

    original = normalized.decode("utf-8")
    lines = original.split("\n")
    repaired = []
    changes = 0
    for line in lines:
        if line.strip() in (r"\[", r"\]"):
            new_line = "$$"
            changes += 1
        else:
            new_line = line.replace(r"\(", "$").replace(r"\)", "$")
            changes += line.count(r"\(") + line.count(r"\)")
        repaired.append(new_line)

    if changes != 26:
        # 10 display delimiters + 14 inline delimiters (seven pairs).
        raise SystemExit(f"Unexpected number of math delimiter edits: {changes}. No changes made.")
    result = "\n".join(repaired)
    for ident, heading in ANCHORS:
        if result.count(heading) != 1:
            raise SystemExit(f"Expected exactly one heading for {ident}; no changes made.")
        result = result.replace(heading, f'<a id="{ident}"></a>\n\n{heading}', 1)
    result = GUIDE + result

    # Independent losslessness check: undo ONLY formatting additions and verify that
    # all original text is byte-identical after normalizing delimiters.
    stripped = result[len(GUIDE):]
    for ident, heading in ANCHORS:
        stripped = stripped.replace(f'<a id="{ident}"></a>\n\n{heading}', heading, 1)
    if stripped != "\n".join(repaired):
        raise SystemExit("Integrity check failed. No changes made.")
    if result.count("\\[") or result.count("\\]") or result.count("\\(") or result.count("\\)"):
        raise SystemExit("Some math delimiters remain. No changes made.")

    backup = path.with_name(f"Experiment_Log.preformat_{datetime.now():%Y%m%d_%H%M%S}.bak")
    shutil.copy2(path, backup)
    rendered = result.replace("\n", eol).encode("utf-8")
    path.write_bytes(bom + rendered)
    print(f"Updated: {path}")
    print(f"Backup:  {backup}")
    print(f"Math delimiters normalized: {changes}")
    print(f"Added navigation links: {len(ANCHORS)}")
    print("All original experiment text preserved. Check 'git diff' before committing.")


if __name__ == "__main__":
    main()
