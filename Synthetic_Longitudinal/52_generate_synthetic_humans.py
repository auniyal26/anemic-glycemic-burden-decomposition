from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "Synthetic_Longitudinal"
SIM_RESULTS = SIM_DIR / "Results"
SIM_RESULTS.mkdir(parents=True, exist_ok=True)

CALIBRATION_FILE = SIM_DIR / "51_simulation_calibration_2005_2008.parquet"
CALIBRATION_MANIFEST = SIM_DIR / "51_simulation_calibration_manifest.json"

OUT_PARQUET = SIM_DIR / "52_synthetic_baseline_population.parquet"
OUT_MANIFEST = SIM_DIR / "52_synthetic_baseline_manifest.json"
OUT_BALANCE = SIM_RESULTS / "52_baseline_balance.csv"

N_SYNTH = 10_000
SEED = 20260917

CONTINUOUS = [
    "RIDAGEYR", "INDFMPIR", "BMXBMI", "EGFR_2021",
    "A_SCALAR", "G_SCALAR",
    "LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW",
    "LBXGH", "LBXGLU", "LBXIN", "LOG_IN",
    "A_OI_PC1", "A_OI_PC2", "A_OI_PC3",
    "G3_OI_PC1", "G3_OI_PC2", "G3_OI_PC3",
    "SOMATIC_SCORE",
]
CATEGORICAL = ["RIAGENDR", "RACE", "EDUC3", "SMOKING3"]

print()
print("SYNTHETIC LONGITUDINAL — BASELINE HUMANS")
print("========================================")
print("Generator source: 2005-06 calibration-fit rows ONLY.")
print("2007-08 is used only as a frozen distributional check.")
print("2009+ data are not read.")
print()

if not CALIBRATION_FILE.exists() or not CALIBRATION_MANIFEST.exists():
    raise FileNotFoundError("Run Script 51 first.")

manifest51 = json.loads(CALIBRATION_MANIFEST.read_text(encoding="utf-8"))
actual_sha = hashlib.sha256(CALIBRATION_FILE.read_bytes()).hexdigest()
if actual_sha != manifest51["output_sha256"]:
    raise RuntimeError("Calibration file SHA256 does not match Script 51 manifest.")

d = pd.read_parquet(CALIBRATION_FILE)

allowed_cycles = set(d["CYCLE"].astype(str).unique())
if not allowed_cycles.issubset({"0506", "0708"}):
    raise RuntimeError(f"Leakage guard failed: unexpected cycles {sorted(allowed_cycles)}")

fit = d[
    d["CYCLE"].astype(str).eq("0506")
    & d["SIM_CALIBRATION_COMPLETE"].eq(1)
].copy()

check = d[
    d["CYCLE"].astype(str).eq("0708")
    & d["SIM_CALIBRATION_COMPLETE"].eq(1)
].copy()

if len(fit) == 0 or len(check) == 0:
    raise RuntimeError("Expected non-empty 0506 fit and 0708 check cohorts.")

w = pd.to_numeric(fit["SURVEY_WT_2YR"], errors="coerce").to_numpy(float)
if not np.all(np.isfinite(w)) or np.any(w <= 0):
    raise RuntimeError("Invalid 0506 survey weights.")
p = w / w.sum()

rng = np.random.default_rng(SEED)
draw = rng.choice(np.arange(len(fit)), size=N_SYNTH, replace=True, p=p)

# Baseline humans are survey-weighted pseudo-individuals sampled only from 2005-06.
# No longitudinal dynamics are introduced here; that begins in Script 53.
synth = fit.iloc[draw].reset_index(drop=True).copy()
synth.insert(0, "SYNTH_ID", [f"SYN_{i:06d}" for i in range(1, N_SYNTH + 1)])
synth["DONOR_SOURCE_ID"] = synth["SOURCE_ID"].astype(str)
synth["BASELINE_SOURCE_CYCLE"] = "0506"
synth["SIM_YEAR"] = 0
synth["SIM_AGE"] = pd.to_numeric(synth["RIDAGEYR"], errors="coerce")
synth["IS_SYNTHETIC"] = 1

# Remove fields that could accidentally make later code treat these rows as NHANES records.
drop_cols = [
    "SEQN", "SOURCE_ID", "CYCLE", "SIM_ROLE",
    "SURVEY_WT_2YR", "SURVEY_WT_4YR",
    "SDMVPSU", "SDMVSTRA", "STRATUM", "PSU",
    "ELIGIBLE_ADULT_NONPREG", "SIM_CALIBRATION_COMPLETE",
]
synth = synth.drop(columns=[c for c in drop_cols if c in synth.columns])

synth.to_parquet(OUT_PARQUET, index=False)


def wmean(frame, col, weight="SURVEY_WT_2YR"):
    x = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
    wt = pd.to_numeric(frame[weight], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(wt) & (wt > 0)
    return float(np.sum(x[ok] * wt[ok]) / np.sum(wt[ok]))


def wsd(frame, col, weight="SURVEY_WT_2YR"):
    x = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
    wt = pd.to_numeric(frame[weight], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(wt) & (wt > 0)
    mu = np.sum(x[ok] * wt[ok]) / np.sum(wt[ok])
    return float(np.sqrt(np.sum(wt[ok] * (x[ok] - mu) ** 2) / np.sum(wt[ok])))


rows = []
for col in CONTINUOUS:
    fit_mu = wmean(fit, col)
    fit_sd = wsd(fit, col)
    syn_mu = float(pd.to_numeric(synth[col], errors="coerce").mean())
    check_mu = wmean(check, col)
    denom = fit_sd if np.isfinite(fit_sd) and fit_sd > 0 else np.nan
    rows.append({
        "variable": col,
        "metric": "mean",
        "fit_0506": fit_mu,
        "synthetic": syn_mu,
        "check_0708": check_mu,
        "smd_synth_vs_fit": (syn_mu - fit_mu) / denom,
        "smd_check_vs_fit": (check_mu - fit_mu) / denom,
    })

for col in CATEGORICAL:
    levels = sorted(set(fit[col].dropna().tolist()) | set(check[col].dropna().tolist()))
    for level in levels:
        fit_prop = float(np.sum(w[fit[col].to_numpy() == level]) / np.sum(w))

        cw = pd.to_numeric(check["SURVEY_WT_2YR"], errors="coerce").to_numpy(float)
        check_mask = check[col].to_numpy() == level
        check_prop = float(np.sum(cw[check_mask]) / np.sum(cw))

        syn_prop = float((synth[col] == level).mean())
        rows.append({
            "variable": f"{col}={level}",
            "metric": "proportion",
            "fit_0506": fit_prop,
            "synthetic": syn_prop,
            "check_0708": check_prop,
            "smd_synth_vs_fit": np.nan,
            "smd_check_vs_fit": np.nan,
        })

balance = pd.DataFrame(rows)
balance.to_csv(OUT_BALANCE, index=False)

max_abs_smd = float(balance["smd_synth_vs_fit"].abs().max(skipna=True))
sha256 = hashlib.sha256(OUT_PARQUET.read_bytes()).hexdigest()

manifest = {
    "script": "52_generate_synthetic_humans.py",
    "purpose": "Generate baseline synthetic pseudo-individuals before longitudinal life simulation.",
    "generator_source_cycle": "0506",
    "distributional_check_cycle": "0708",
    "future_cycles_read": False,
    "method": "survey-weighted donor resampling with replacement",
    "n_synthetic": N_SYNTH,
    "seed": SEED,
    "important_rule": "2007-08 may be inspected but must not be used to tune the generator in this script.",
    "longitudinal_dynamics_added": False,
    "input_sha256": actual_sha,
    "output_sha256": sha256,
    "output": str(OUT_PARQUET.relative_to(ROOT)),
    "balance_file": str(OUT_BALANCE.relative_to(ROOT)),
    "max_abs_continuous_smd_synthetic_vs_0506": max_abs_smd,
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"PASS 0506 donor rows: {len(fit):,}")
print(f"PASS 0708 frozen check rows: {len(check):,}")
print(f"PASS synthetic humans: {len(synth):,}")
print(f"PASS max |SMD| synthetic vs 0506: {max_abs_smd:.4f}")
print(f"PASS saved: {OUT_PARQUET}")
print(f"PASS balance: {OUT_BALANCE}")
print(f"PASS manifest: {OUT_MANIFEST}")
print(f"PASS SHA256: {sha256}")
