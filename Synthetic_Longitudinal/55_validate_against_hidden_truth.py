from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "Synthetic_Longitudinal"
SIM_RESULTS = SIM_DIR / "Results"
PARAMETERS = ROOT / "Parameters"
SIM_RESULTS.mkdir(parents=True, exist_ok=True)

OBS_FILE = SIM_DIR / "53_synthetic_longitudinal_observed.parquet"
TRUTH_FILE = SIM_DIR / "53_synthetic_longitudinal_truth.parquet"
M53_FILE = SIM_DIR / "53_synthetic_longitudinal_manifest.json"
PRED_FILE = SIM_DIR / "54_frozen_longitudinal_predictions.parquet"
M54_FILE = SIM_DIR / "54_longitudinal_framework_manifest.json"
A_PARAM = PARAMETERS / "46_outcome_independent_A_transform.json"
G_PARAM = PARAMETERS / "46_outcome_independent_G3_transform.json"

OUT_PRED = SIM_RESULTS / "55_truth_prediction_metrics.csv"
OUT_DELTA = SIM_RESULTS / "55_model_increment_diagnostics.csv"
OUT_REP = SIM_RESULTS / "55_representation_truth_recovery.csv"
OUT_MANIFEST = SIM_DIR / "55_truth_validation_manifest.json"

print()
print("SYNTHETIC LONGITUDINAL — HIDDEN TRUTH VALIDATION")
print("================================================")
print("Predictions were frozen in Script 54 before hidden truth is opened here.")
print()

for p in [OBS_FILE, TRUTH_FILE, M53_FILE, PRED_FILE, M54_FILE, A_PARAM, G_PARAM]:
    if not p.exists():
        raise FileNotFoundError(f"Missing prerequisite: {p}")

m53 = json.loads(M53_FILE.read_text(encoding="utf-8"))
m54 = json.loads(M54_FILE.read_text(encoding="utf-8"))

obs_sha = hashlib.sha256(OBS_FILE.read_bytes()).hexdigest()
truth_sha = hashlib.sha256(TRUTH_FILE.read_bytes()).hexdigest()
pred_sha = hashlib.sha256(PRED_FILE.read_bytes()).hexdigest()

if obs_sha != m53["observed_sha256"]:
    raise RuntimeError("Observed SHA mismatch.")
if truth_sha != m53["truth_sha256"]:
    raise RuntimeError("Truth SHA mismatch.")
if pred_sha != m54["prediction_sha256"]:
    raise RuntimeError("Prediction SHA mismatch: predictions may have changed after truth was available.")

obs = pd.read_parquet(OBS_FILE)
truth = pd.read_parquet(TRUTH_FILE)
pred = pd.read_parquet(PRED_FILE)

# ------------------------------------------------------------------
# 1) Evaluate frozen predictions against the noise-free hidden mean
#    at the TARGET year.
# ------------------------------------------------------------------
truth_target = truth[
    ["SYNTH_ID", "SCENARIO", "SIM_YEAR", "TRUE_SOMATIC_MEAN",
     "TRUE_A_STATE", "TRUE_G_STATE", "TRUE_BETA_A", "TRUE_BETA_G",
     "TRUE_BETA_AG", "TRUE_HBA1C_ASSAY_SHIFT"]
].rename(columns={"SIM_YEAR": "TARGET_YEAR"})

j = pred.merge(
    truth_target,
    on=["SYNTH_ID", "SCENARIO", "TARGET_YEAR"],
    how="left",
    validate="many_to_one",
)

if j["TRUE_SOMATIC_MEAN"].isna().any():
    raise RuntimeError("Truth join failed for some prediction rows.")

rows = []
for keys, q in j.groupby(["SCENARIO", "MODEL", "EVAL_SET"], sort=True):
    y = q["TRUE_SOMATIC_MEAN"].to_numpy(float)
    p = q["PREDICTED_NEXT_SOMATIC"].to_numpy(float)
    rows.append({
        "scenario": keys[0],
        "model": keys[1],
        "eval_set": keys[2],
        "n": len(q),
        "rmse_vs_true_mean": float(np.sqrt(mean_squared_error(y, p))),
        "r2_vs_true_mean": float(r2_score(y, p)),
        "mean_error_vs_true_mean": float(np.mean(p - y)),
    })

truth_metrics = pd.DataFrame(rows)
truth_metrics.to_csv(OUT_PRED, index=False)

# Model increments on the most important future tests.
pivot = truth_metrics.pivot_table(
    index=["scenario", "eval_set"],
    columns="model",
    values="r2_vs_true_mean",
)
delta_rows = []
for (scenario, eval_set), r in pivot.iterrows():
    delta_rows.append({
        "scenario": scenario,
        "eval_set": eval_set,
        "component_minus_scalar":
            float(r["component_additive"] - r["scalar_additive"]),
        "time_minus_component":
            float(r["component_time"] - r["component_additive"]),
        "ag_minus_time":
            float(r["component_time_ag"] - r["component_time"]),
    })

deltas = pd.DataFrame(delta_rows)
deltas.to_csv(OUT_DELTA, index=False)

# ------------------------------------------------------------------
# 2) Representation recovery against hidden physiological truth.
#    Fit a truth-decoder only on years 0-4, then freeze it.
#    This decoder is an audit instrument, NOT part of the framework.
# ------------------------------------------------------------------
def load_transform(path):
    x = json.loads(path.read_text(encoding="utf-8"))
    vars_ = x["variables"]
    mu = np.array([x["standardization"]["mean"][v] for v in vars_], float)
    sd = np.array([x["standardization"]["scale"][v] for v in vars_], float)
    L = np.array(x["pca"]["loadings_columns_are_components"], float)
    k = int(x["pca"]["retained_components"])
    return vars_, mu, sd, L, k

A_VARS, A_MU, A_SD, A_L, A_K = load_transform(A_PARAM)
G_VARS, G_MU, G_SD, G_L, G_K = load_transform(G_PARAM)

ZA = (obs[A_VARS].to_numpy(float) - A_MU) / A_SD
ZG = (obs[G_VARS].to_numpy(float) - G_MU) / G_SD
SA = ZA @ A_L[:, :A_K]
SG = ZG @ G_L[:, :G_K]

rep = obs[["SYNTH_ID", "SCENARIO", "SIM_YEAR"]].copy()
for i in range(A_K):
    rep[f"A_PC{i+1}"] = SA[:, i]
for i in range(G_K):
    rep[f"G_PC{i+1}"] = SG[:, i]

rep = rep.merge(
    truth[["SYNTH_ID", "SCENARIO", "SIM_YEAR", "TRUE_A_STATE", "TRUE_G_STATE",
           "TRUE_HBA1C_ASSAY_SHIFT"]],
    on=["SYNTH_ID", "SCENARIO", "SIM_YEAR"],
    how="left",
    validate="one_to_one",
)

rep_rows = []
for scenario, q in rep.groupby("SCENARIO", sort=True):
    train = q[q["SIM_YEAR"].between(0, 4)].copy()
    A_cols = [f"A_PC{i}" for i in range(1, A_K + 1)]
    G_cols = [f"G_PC{i}" for i in range(1, G_K + 1)]

    decA = LinearRegression().fit(train[A_cols], train["TRUE_A_STATE"])
    decG = LinearRegression().fit(train[G_cols], train["TRUE_G_STATE"])

    q = q.copy()
    q["A_TRUTH_HAT"] = decA.predict(q[A_cols])
    q["G_TRUTH_HAT"] = decG.predict(q[G_cols])

    for year, yy in q.groupby("SIM_YEAR", sort=True):
        a_true = yy["TRUE_A_STATE"].to_numpy(float)
        g_true = yy["TRUE_G_STATE"].to_numpy(float)
        a_hat = yy["A_TRUTH_HAT"].to_numpy(float)
        g_hat = yy["G_TRUTH_HAT"].to_numpy(float)

        rep_rows.append({
            "scenario": scenario,
            "sim_year": int(year),
            "n": len(yy),
            "A_corr_truth": float(np.corrcoef(a_true, a_hat)[0, 1]),
            "A_mean_error": float(np.mean(a_hat - a_true)),
            "G_corr_truth": float(np.corrcoef(g_true, g_hat)[0, 1]),
            "G_mean_error": float(np.mean(g_hat - g_true)),
            "true_hba1c_assay_shift": float(yy["TRUE_HBA1C_ASSAY_SHIFT"].mean()),
        })

rep_audit = pd.DataFrame(rep_rows)
rep_audit.to_csv(OUT_REP, index=False)

# ------------------------------------------------------------------
# Concise terminal summary: unseen-person future test.
# ------------------------------------------------------------------
focus = deltas[deltas["eval_set"].eq("test_unseen_people_10_14")].copy()

print("Future unseen-person truth recovery (R² increments):")
for _, r in focus.sort_values("scenario").iterrows():
    print(
        f"  {r['scenario']}: "
        f"components-scalar={r['component_minus_scalar']:+.4f}, "
        f"time={r['time_minus_component']:+.4f}, "
        f"A×G={r['ag_minus_time']:+.4f}"
    )

gshift = rep_audit[
    rep_audit["scenario"].eq("measurement_shift")
    & rep_audit["sim_year"].isin([0, 7, 8, 15])
][["sim_year", "G_corr_truth", "G_mean_error", "true_hba1c_assay_shift"]]

print()
print("Measurement-shift G recovery:")
for _, r in gshift.iterrows():
    print(
        f"  year {int(r['sim_year']):2d}: "
        f"corr={r['G_corr_truth']:.4f}, "
        f"bias={r['G_mean_error']:+.4f}, "
        f"assay_shift={r['true_hba1c_assay_shift']:+.3f}"
    )

manifest = {
    "script": "55_validate_against_hidden_truth.py",
    "prediction_sha256_verified_before_truth_use": True,
    "truth_sha256_verified": True,
    "predictions_frozen_before_truth_opened": True,
    "truth_prediction_metrics": str(OUT_PRED.relative_to(ROOT)),
    "model_increment_diagnostics": str(OUT_DELTA.relative_to(ROOT)),
    "representation_truth_recovery": str(OUT_REP.relative_to(ROOT)),
    "truth_decoder_role": "audit only; not part of the predictive framework",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print()
print(f"PASS truth metrics: {OUT_PRED}")
print(f"PASS increments: {OUT_DELTA}")
print(f"PASS representation audit: {OUT_REP}")
print(f"PASS manifest: {OUT_MANIFEST}")
