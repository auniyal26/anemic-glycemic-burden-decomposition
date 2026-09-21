from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "Synthetic_Longitudinal"
SIM_RESULTS = SIM_DIR / "Results"
SIM_RESULTS.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = SIM_DIR / "52_synthetic_baseline_population.parquet"
BASELINE_MANIFEST = SIM_DIR / "52_synthetic_baseline_manifest.json"

OBS_FILE = SIM_DIR / "57_synthetic_longitudinal_observed_v2.parquet"
TRUTH_FILE = SIM_DIR / "57_synthetic_longitudinal_truth_v2.parquet"
MANIFEST_FILE = SIM_DIR / "57_synthetic_longitudinal_v2_manifest.json"
SUMMARY_FILE = SIM_RESULTS / "57_scenario_summary_v2.csv"

SEED = 20260917
YEARS = 15
RHO_PHENOTYPE = 0.42

SCENARIOS = [
    "stable_additive",
    "mapping_drift",
    "ag_interaction",
    "measurement_shift",
]

print()
print("SYNTHETIC LONGITUDINAL — LIFE TRAJECTORIES V2")
print("=============================================")
print("Fix: phenotype dynamics preserve each person's baseline set-point.")
print("Physiology transitions are unchanged from Script 53.")
print("No NHANES 2007-08 or 2009+ data are read.")
print()

if not BASELINE_FILE.exists() or not BASELINE_MANIFEST.exists():
    raise FileNotFoundError("Run Script 52 first.")

m52 = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
baseline_sha = hashlib.sha256(BASELINE_FILE.read_bytes()).hexdigest()
if baseline_sha != m52["output_sha256"]:
    raise RuntimeError("Baseline SHA256 does not match Script 52 manifest.")

base = pd.read_parquet(BASELINE_FILE).copy()

required = [
    "SYNTH_ID", "SIM_AGE", "RIDAGEYR", "RIAGENDR", "RACE", "EDUC3", "SMOKING3",
    "INDFMPIR", "BMXBMI", "EGFR_2021",
    "LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW",
    "LBXGH", "LBXGLU", "LBXIN", "LOG_IN", "SOMATIC_SCORE",
]
missing = [c for c in required if c not in base.columns]
if missing:
    raise ValueError(f"Baseline missing required columns: {missing}")

rng = np.random.default_rng(SEED)

marker_cols = [
    "LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW",
    "LBXGH", "LBXGLU", "LOG_IN",
]
mu = {c: float(pd.to_numeric(base[c], errors="coerce").mean()) for c in marker_cols}
sd = {c: float(pd.to_numeric(base[c], errors="coerce").std(ddof=0)) for c in marker_cols}

for c in marker_cols:
    if not np.isfinite(sd[c]) or sd[c] <= 0:
        raise RuntimeError(f"Invalid baseline SD for {c}")

def z(frame, c):
    return (pd.to_numeric(frame[c], errors="coerce").to_numpy(float) - mu[c]) / sd[c]

# Hidden physiological truth remains deliberately different from framework PCA.
A0 = (
    -0.60 * z(base, "LBXHGB")
    -0.20 * z(base, "LBXRBCSI")
    -0.10 * z(base, "LBXMCVSI")
    +0.45 * z(base, "LBXRDW")
)
G0 = (
     0.55 * z(base, "LBXGH")
    +0.35 * z(base, "LBXGLU")
    +0.30 * z(base, "LOG_IN")
)
A0 = (A0 - A0.mean()) / A0.std(ddof=0)
G0 = (G0 - G0.mean()) / G0.std(ddof=0)

n = len(base)
all_obs = []
all_truth = []
summary_rows = []

# Baseline observed phenotype and covariates.
y0 = pd.to_numeric(base["SOMATIC_SCORE"], errors="coerce").to_numpy(float)
age0 = pd.to_numeric(base["SIM_AGE"], errors="coerce").to_numpy(float)
bmi0 = pd.to_numeric(base["BMXBMI"], errors="coerce").to_numpy(float)

# Baseline coefficients shared by all worlds.
BETA_A0 = 0.22
BETA_G0 = 0.18
BETA_AG0 = 0.0

# Individual intercept chosen so the latent set-point at t=0 equals observed baseline.
base_nonlin = 0.035 * np.maximum(A0, 0) ** 2
base_bmi = 0.025 * ((bmi0 - 25.0) / 5.0)
base_age = 0.012 * ((age0 - 45.0) / 10.0)

subject_intercept = (
    y0
    - BETA_A0 * A0
    - BETA_G0 * G0
    - BETA_AG0 * A0 * G0
    - base_nonlin
    - base_bmi
    - base_age
)

for scenario in SCENARIOS:
    age = age0.copy()
    bmi = bmi0.copy()
    egfr = pd.to_numeric(base["EGFR_2021"], errors="coerce").to_numpy(float).copy()

    A = A0.copy()
    G = G0.copy()

    hb0 = pd.to_numeric(base["LBXHGB"], errors="coerce").to_numpy(float)
    rbc0 = pd.to_numeric(base["LBXRBCSI"], errors="coerce").to_numpy(float)
    mcv0 = pd.to_numeric(base["LBXMCVSI"], errors="coerce").to_numpy(float)
    rdw0 = pd.to_numeric(base["LBXRDW"], errors="coerce").to_numpy(float)
    hba1c0 = pd.to_numeric(base["LBXGH"], errors="coerce").to_numpy(float)
    fpg0 = pd.to_numeric(base["LBXGLU"], errors="coerce").to_numpy(float)
    login0 = pd.to_numeric(base["LOG_IN"], errors="coerce").to_numpy(float)

    prev_y_latent = y0.copy()

    for t in range(YEARS + 1):
        beta_a = BETA_A0
        beta_g = BETA_G0
        beta_ag = BETA_AG0

        if scenario == "mapping_drift":
            beta_g = BETA_G0 + 0.018 * t
            beta_a = BETA_A0 - 0.006 * t

        if scenario == "ag_interaction" and t >= 5:
            beta_ag = 0.16

        # Observed biomarker layer: unchanged from Script 53.
        hb = hb0 - 0.32 * (A - A0) + rng.normal(0, 0.18, n)
        rbc = rbc0 - 0.08 * (A - A0) + rng.normal(0, 0.07, n)
        mcv = mcv0 - 0.55 * (A - A0) + rng.normal(0, 0.75, n)
        rdw = rdw0 + 0.28 * (A - A0) + rng.normal(0, 0.20, n)

        hba1c = hba1c0 + 0.28 * (G - G0) + rng.normal(0, 0.08, n)
        fpg = fpg0 + 7.5 * (G - G0) + rng.normal(0, 3.5, n)
        login = login0 + 0.16 * (G - G0) + rng.normal(0, 0.08, n)

        assay_shift = 0.0
        if scenario == "measurement_shift" and t >= 8:
            assay_shift = 0.20 + 0.015 * (t - 8)
            hba1c = hba1c + assay_shift

        insulin = np.exp(login)

        # Corrected phenotype dynamics:
        # 1) each person has a baseline latent set-point;
        # 2) current physiology/covariates move that set-point;
        # 3) AR persistence acts around the set-point rather than shrinking to zero.
        current_setpoint = (
            subject_intercept
            + beta_a * A
            + beta_g * G
            + beta_ag * A * G
            + 0.035 * np.maximum(A, 0) ** 2
            + 0.025 * ((bmi - 25.0) / 5.0)
            + 0.012 * ((age - 45.0) / 10.0)
        )

        y_mean = (
            RHO_PHENOTYPE * prev_y_latent
            + (1.0 - RHO_PHENOTYPE) * current_setpoint
        )

        y_latent = y_mean + rng.normal(0, 0.85, n)
        y_obs = np.clip(np.rint(y_latent), 0, 9).astype(int)

        obs = pd.DataFrame({
            "SYNTH_ID": base["SYNTH_ID"].to_numpy(),
            "SCENARIO": scenario,
            "SIM_YEAR": t,
            "SIM_AGE": age,
            "RIAGENDR": base["RIAGENDR"].to_numpy(),
            "RACE": base["RACE"].to_numpy(),
            "EDUC3": base["EDUC3"].to_numpy(),
            "SMOKING3": base["SMOKING3"].to_numpy(),
            "INDFMPIR": base["INDFMPIR"].to_numpy(),
            "BMXBMI": bmi,
            "EGFR_2021": egfr,
            "LBXHGB": hb,
            "LBXRBCSI": rbc,
            "LBXMCVSI": mcv,
            "LBXRDW": rdw,
            "LBXGH": hba1c,
            "LBXGLU": fpg,
            "LBXIN": insulin,
            "LOG_IN": login,
            "SOMATIC_SCORE": y_obs,
        })
        all_obs.append(obs)

        truth = pd.DataFrame({
            "SYNTH_ID": base["SYNTH_ID"].to_numpy(),
            "SCENARIO": scenario,
            "SIM_YEAR": t,
            "TRUE_A_STATE": A,
            "TRUE_G_STATE": G,
            "TRUE_BETA_A": beta_a,
            "TRUE_BETA_G": beta_g,
            "TRUE_BETA_AG": beta_ag,
            "TRUE_HBA1C_ASSAY_SHIFT": assay_shift,
            "TRUE_SOMATIC_SETPOINT": current_setpoint,
            "TRUE_SOMATIC_MEAN": y_mean,
            "TRUE_SOMATIC_LATENT": y_latent,
        })
        all_truth.append(truth)

        summary_rows.append({
            "scenario": scenario,
            "sim_year": t,
            "n": n,
            "mean_age": float(np.mean(age)),
            "mean_bmi": float(np.mean(bmi)),
            "mean_egfr": float(np.mean(egfr)),
            "mean_true_A": float(np.mean(A)),
            "mean_true_G": float(np.mean(G)),
            "mean_somatic": float(np.mean(y_obs)),
            "mean_somatic_setpoint": float(np.mean(current_setpoint)),
            "beta_A": beta_a,
            "beta_G": beta_g,
            "beta_AG": beta_ag,
            "hba1c_assay_shift": assay_shift,
        })

        prev_y_latent = y_latent

        if t == YEARS:
            continue

        # Physiology transitions intentionally unchanged from Script 53.
        age_next = age + 1.0

        bmi = (
            bmi
            + 0.04
            + 0.025 * G
            + rng.normal(0, 0.35, n)
        )
        bmi = np.clip(bmi, 15, 60)

        egfr = (
            egfr
            - 0.70
            - 0.10 * np.maximum(G, 0)
            + rng.normal(0, 1.6, n)
        )
        egfr = np.clip(egfr, 10, 150)

        age_term = (age_next - 45.0) / 10.0
        bmi_term = (bmi - 25.0) / 5.0

        A = (
            0.94 * A
            + 0.018 * age_term
            + 0.018 * bmi_term
            + rng.normal(0, 0.18, n)
        )

        G = (
            0.95 * G
            + 0.030 * age_term
            + 0.050 * bmi_term
            + rng.normal(0, 0.16, n)
        )

        if scenario == "ag_interaction":
            G = G + 0.018 * np.maximum(A, 0)

        age = age_next

observed = pd.concat(all_obs, ignore_index=True)
truth = pd.concat(all_truth, ignore_index=True)
summary = pd.DataFrame(summary_rows)

key = ["SYNTH_ID", "SCENARIO", "SIM_YEAR"]
if observed.duplicated(key).any():
    raise RuntimeError("Duplicate observed trajectory keys.")
if truth.duplicated(key).any():
    raise RuntimeError("Duplicate truth trajectory keys.")
if len(observed) != len(truth):
    raise RuntimeError("Observed and truth row counts differ.")

observed.to_parquet(OBS_FILE, index=False)
truth.to_parquet(TRUTH_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)

obs_sha = hashlib.sha256(OBS_FILE.read_bytes()).hexdigest()
truth_sha = hashlib.sha256(TRUTH_FILE.read_bytes()).hexdigest()

manifest = {
    "script": "57_simulate_longitudinal_lives_v2.py",
    "replaces": "Script 53 for v2 experiments; Script 53 outputs remain preserved.",
    "change_from_v1": (
        "Phenotype process now evolves around an individual baseline set-point. "
        "Physiology transition equations are unchanged."
    ),
    "seed": SEED,
    "baseline_input_sha256": baseline_sha,
    "n_humans": int(n),
    "years": YEARS,
    "timepoints_per_human": YEARS + 1,
    "phenotype_persistence_rho": RHO_PHENOTYPE,
    "scenarios": SCENARIOS,
    "future_nhanes_read": False,
    "generator_uses_0708": False,
    "generator_uses_2009_plus": False,
    "simulator_truth_is_not_framework_pca": True,
    "observed_output": str(OBS_FILE.relative_to(ROOT)),
    "truth_output": str(TRUTH_FILE.relative_to(ROOT)),
    "summary_output": str(SUMMARY_FILE.relative_to(ROOT)),
    "observed_sha256": obs_sha,
    "truth_sha256": truth_sha,
    "important_rule": (
        "Downstream framework tests must use the observed file only. "
        "Truth remains unavailable until predictions are frozen."
    ),
    "interpretation": (
        "Controlled synthetic worlds for method validation, not estimates "
        "of true individual NHANES longitudinal dynamics."
    ),
}
MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"PASS humans per world: {n:,}")
print(f"PASS scenarios: {len(SCENARIOS)}")
print(f"PASS years: 0-{YEARS}")
print(f"PASS baseline mean somatic: {y0.mean():.3f}")
for scenario in SCENARIOS:
    q = summary[(summary["scenario"] == scenario) & (summary["sim_year"] == YEARS)]
    print(f"PASS year {YEARS} mean somatic [{scenario}]: {q['mean_somatic'].iloc[0]:.3f}")
print(f"PASS observed rows: {len(observed):,}")
print(f"PASS truth rows: {len(truth):,}")
print(f"PASS observed: {OBS_FILE}")
print(f"PASS truth: {TRUTH_FILE}")
print(f"PASS summary: {SUMMARY_FILE}")
print(f"PASS manifest: {MANIFEST_FILE}")
print(f"PASS observed SHA256: {obs_sha}")
print(f"PASS truth SHA256: {truth_sha}")
