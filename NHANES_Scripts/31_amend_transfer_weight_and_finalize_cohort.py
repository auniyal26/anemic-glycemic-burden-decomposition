from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES_Transfer"
PROCESSED = ROOT / "Data" / "Processed"
AUDIT = ROOT / "Audit"
RESULTS = ROOT / "Results"

PREMODEL = PROCESSED / "30_transfer_harmonized_PREMODEL.parquet"
FROZEN = AUDIT / "28_temporal_transfer_protocol_FROZEN.json"

if not PREMODEL.exists():
    raise FileNotFoundError(PREMODEL)
if not FROZEN.exists():
    raise FileNotFoundError(FROZEN)

print()
print("TRANSFER PROTOCOL WEIGHT AMENDMENT")
print("==================================")

df = pd.read_parquet(PREMODEL)

# NHANES Aug 2021-Aug 2023 blood-analyte analyses require WTPH2YR,
# not the generic MEC weight. Pull it from CBC_L before outcome modeling.
cbc_l = pd.read_sas(DATA / "2123" / "CBC_L.XPT", format="xport")

if "WTPH2YR" not in cbc_l.columns:
    raise RuntimeError("WTPH2YR not found in CBC_L.XPT.")

phw = cbc_l[["SEQN", "WTPH2YR"]].copy()

df = df.merge(phw, on="SEQN", how="left", validate="many_to_one")

# Explicit weight columns for future validation models.
df["WEIGHT_0918_POOLED"] = np.where(
    df["CYCLE"].isin(["0910", "1112", "1314", "1516", "1718"]),
    df["WTMEC2YR"] / 5.0,
    np.nan
)

df["WEIGHT_2123"] = np.where(
    df["CYCLE"] == "2123",
    df["WTPH2YR"],
    np.nan
)

# Prefix strata/PSU for pooled 2009-2018 analysis.
df["STRATUM_TRANSFER"] = (
    df["CYCLE"].astype(str) + "_" + df["SDMVSTRA"].astype("Int64").astype(str)
)
df["PSU_TRANSFER"] = (
    df["CYCLE"].astype(str) + "_" +
    df["SDMVSTRA"].astype("Int64").astype(str) + "_" +
    df["SDMVPSU"].astype("Int64").astype(str)
)

out = PROCESSED / "31_transfer_harmonized_FINAL_PREMODEL.parquet"
df.to_parquet(out, index=False)

# Audit counts of valid weights.
rows = []
for cycle in ["0910", "1112", "1314", "1516", "1718", "2123"]:
    q = df[df["CYCLE"] == cycle].copy()
    weight_col = "WEIGHT_2123" if cycle == "2123" else "WTMEC2YR"
    rows.append({
        "cycle": cycle,
        "n": len(q),
        "weight_variable": "WTPH2YR" if cycle == "2123" else "WTMEC2YR",
        "positive_weight_n": int((pd.to_numeric(q[weight_col], errors="coerce") > 0).sum())
    })

pd.DataFrame(rows).to_csv(
    RESULTS / "31_transfer_weight_audit.csv",
    index=False
)

# Preserve original freeze; add a transparent pre-outcome amendment.
addendum = {
    "amendment_number": 1,
    "date": "2026-08-22",
    "timing": "Before any transfer outcome model or coefficient was fit/viewed",
    "reason": (
        "Official NHANES August 2021-August 2023 analytic guidance states that "
        "analyses using blood analytes must use the phlebotomy weight WTPH2YR."
    ),
    "change": {
        "old_2123_weight": "WTMEC2YR",
        "new_2123_weight": "WTPH2YR",
        "unchanged_2009_2018": "WTMEC2YR; divide by 5 when pooling five 2-year cycles"
    },
    "unchanged_frozen_elements": [
        "A definition",
        "G definition",
        "A×G definition",
        "somatic PHQ outcome",
        "PHQ9 total outcome",
        "X adjustment sets",
        "success/failure rules",
        "no tuning on validation outcomes"
    ],
    "final_premodel_file": str(out.relative_to(ROOT))
}

addendum_path = AUDIT / "31_transfer_protocol_addendum_weight.json"
addendum_path.write_text(json.dumps(addendum, indent=2), encoding="utf-8")

md = """# Temporal Transfer Protocol Addendum 1 — Survey Weight

**Date:** 2026-08-22  
**Timing:** Before any validation outcome model or coefficient was fit or viewed.

Official NHANES August 2021–August 2023 guidance requires the **phlebotomy 2-year weight (`WTPH2YR`)** for analyses using blood analytes.

Therefore:

- 2009–2018 pooled temporal validation: `WTMEC2YR / 5`
- 2021–2023 modern holdout: `WTPH2YR`

No exposure definition, outcome definition, confounder set, interaction definition, or replication criterion has been changed.

The original frozen protocol is preserved unchanged; this file records the pre-outcome methodological correction transparently.
"""
(AUDIT / "31_transfer_protocol_addendum_weight.md").write_text(md, encoding="utf-8")

print("PASS  Original frozen protocol preserved unchanged.")
print("PASS  Pre-outcome weight correction recorded as an addendum.")
print("PASS  2021-2023 blood analyses will use WTPH2YR.")
print("PASS  2009-2018 pooled analysis will use WTMEC2YR / 5.")
print(f"PASS  Final premodel cohort: {out}")
print()
print("NO VALIDATION OUTCOME MODEL HAS BEEN FIT.")
