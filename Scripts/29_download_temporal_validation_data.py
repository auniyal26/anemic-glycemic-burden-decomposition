from pathlib import Path
import hashlib
import json
import requests
from tqdm import tqdm

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES_Transfer"
AUDIT = ROOT / "Audit"

PROTOCOL = AUDIT / "28_temporal_transfer_protocol_FROZEN.json"

if not PROTOCOL.exists():
    raise FileNotFoundError(
        "Run 28_freeze_temporal_transfer_protocol.py first."
    )

CYCLES = {
    "0910": {"year": "2009", "suffix": "F"},
    "1112": {"year": "2011", "suffix": "G"},
    "1314": {"year": "2013", "suffix": "H"},
    "1516": {"year": "2015", "suffix": "I"},
    "1718": {"year": "2017", "suffix": "J"},
    "2123": {"year": "2021", "suffix": "L"},
}

COMPONENTS = [
    "DEMO",
    "DPQ",
    "CBC",
    "GHB",
    "GLU",
    "BMX",
    "BIOPRO",
    "SMQ",
    "DIQ",
]

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"

DATA.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = []

print()
print("NHANES TEMPORAL-VALIDATION DOWNLOAD")
print("===================================")
print("This script downloads raw files only. It does not fit or inspect outcome models.")
print()

jobs = []
for cycle, meta in CYCLES.items():
    for component in COMPONENTS:
        name = f"{component}_{meta['suffix']}"
        url = BASE.format(year=meta["year"], name=name)
        out = DATA / cycle / f"{name}.XPT"
        jobs.append((cycle, component, url, out))

for cycle, component, url, out in tqdm(jobs, desc="Files", unit="file", dynamic_ncols=True):
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        status = "existing"
    else:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                manifest.append({
                    "cycle": cycle,
                    "component": component,
                    "url": url,
                    "path": str(out),
                    "status": f"HTTP_{r.status_code}",
                    "bytes": 0,
                    "sha256": None
                })
                continue

            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        status = "downloaded"

    manifest.append({
        "cycle": cycle,
        "component": component,
        "url": url,
        "path": str(out.relative_to(ROOT)),
        "status": status,
        "bytes": out.stat().st_size,
        "sha256": sha256(out)
    })

manifest_path = AUDIT / "29_transfer_raw_download_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

failed = [x for x in manifest if str(x["status"]).startswith("HTTP_")]

print()
print(f"PASS  Manifest written: {manifest_path}")
print(f"Downloaded/existing: {len(manifest)-len(failed)}/{len(manifest)}")

if failed:
    print("WARNING  Some files were unavailable at expected URLs:")
    for x in failed:
        print(f"  {x['cycle']} {x['component']} {x['status']}")
else:
    print("PASS  All requested validation files available.")

print()
print("NEXT  Build a frozen harmonized validation cohort WITHOUT changing the protocol.")
