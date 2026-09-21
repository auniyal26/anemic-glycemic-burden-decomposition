from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "Synthetic_Longitudinal"
SIM_RESULTS = SIM_DIR / "Results"
PARAMETERS = ROOT / "Parameters"
SIM_RESULTS.mkdir(parents=True, exist_ok=True)

OBS_FILE = SIM_DIR / "53_synthetic_longitudinal_observed.parquet"
M53 = SIM_DIR / "53_synthetic_longitudinal_manifest.json"

A_PARAM = PARAMETERS / "46_outcome_independent_A_transform.json"
G_PARAM = PARAMETERS / "46_outcome_independent_G3_transform.json"

PRED_FILE = SIM_DIR / "54_frozen_longitudinal_predictions.parquet"
METRIC_FILE = SIM_RESULTS / "54_longitudinal_prediction_metrics.csv"
SPLIT_FILE = SIM_RESULTS / "54_person_split.csv"
MANIFEST_FILE = SIM_DIR / "54_longitudinal_framework_manifest.json"

SEED = 20260917
HOLDOUT_FRAC = 0.20
RIDGE_ALPHA = 1.0

print()
print("SYNTHETIC LONGITUDINAL — FROZEN FRAMEWORK TEST")
print("==============================================")
print("Observed synthetic trajectories only.")
print("Hidden truth is NOT read.")
print("Frozen Script 46 A/G transforms are projected without refitting.")
print("Chronological fitting: years 0-4 only.")
print("Validation: years 5-9. Test: years 10-14.")
print("20% of people are never used for model fitting.")
print()

for p in [OBS_FILE, M53, A_PARAM, G_PARAM]:
    if not p.exists():
        raise FileNotFoundError(f"Missing prerequisite: {p}")

m53 = json.loads(M53.read_text(encoding="utf-8"))
obs_sha = hashlib.sha256(OBS_FILE.read_bytes()).hexdigest()
if obs_sha != m53["observed_sha256"]:
    raise RuntimeError("Observed trajectory SHA256 does not match Script 53 manifest.")

d = pd.read_parquet(OBS_FILE).copy()

if any(str(c).startswith("TRUE_") for c in d.columns):
    raise RuntimeError("Leakage guard failed: TRUE_* columns present in observed data.")

# ------------------------------------------------------------------
# Frozen representation projection from Script 46 parameters.
# ------------------------------------------------------------------
def load_transform(path):
    x = json.loads(path.read_text(encoding="utf-8"))
    vars_ = x["variables"]
    mu = np.array([x["standardization"]["mean"][v] for v in vars_], dtype=float)
    sd = np.array([x["standardization"]["scale"][v] for v in vars_], dtype=float)
    L = np.array(x["pca"]["loadings_columns_are_components"], dtype=float)
    k = int(x["pca"]["retained_components"])
    return vars_, mu, sd, L, k

A_VARS, A_MU, A_SD, A_L, A_K = load_transform(A_PARAM)
G_VARS, G_MU, G_SD, G_L, G_K = load_transform(G_PARAM)

missing = [c for c in A_VARS + G_VARS if c not in d.columns]
if missing:
    raise ValueError(f"Observed synthetic data missing frozen-transform variables: {missing}")

ZA = (d[A_VARS].to_numpy(float) - A_MU) / A_SD
ZG = (d[G_VARS].to_numpy(float) - G_MU) / G_SD
SA = ZA @ A_L
SG = ZG @ G_L

for j in range(A_K):
    d[f"A_FROZEN_PC{j+1}"] = SA[:, j]
for j in range(G_K):
    d[f"G_FROZEN_PC{j+1}"] = SG[:, j]

# Conventional scalar comparators.
d["HB_THRESHOLD"] = np.where(d["RIAGENDR"] == 1, 13.0, 12.0)
d["A_SCALAR"] = (d["HB_THRESHOLD"] - d["LBXHGB"]).clip(lower=0)
d["G_SCALAR"] = (d["LBXGH"] - 5.7).clip(lower=0)

# ------------------------------------------------------------------
# Next-year target. Current time t predicts observed somatic score t+1.
# ------------------------------------------------------------------
d = d.sort_values(["SCENARIO", "SYNTH_ID", "SIM_YEAR"]).reset_index(drop=True)
grp = d.groupby(["SCENARIO", "SYNTH_ID"], sort=False)
d["TARGET_NEXT_SOMATIC"] = grp["SOMATIC_SCORE"].shift(-1)
d["TARGET_YEAR"] = d["SIM_YEAR"] + 1
d = d[d["SIM_YEAR"] <= 14].copy()

# ------------------------------------------------------------------
# Person holdout: same split for every scenario.
# ------------------------------------------------------------------
ids = np.array(sorted(d["SYNTH_ID"].unique()))
rng = np.random.default_rng(SEED)
rng.shuffle(ids)
n_hold = int(round(len(ids) * HOLDOUT_FRAC))
holdout_ids = set(ids[:n_hold])
d["PERSON_SET"] = np.where(d["SYNTH_ID"].isin(holdout_ids), "HOLDOUT", "DEVELOPMENT")

split_df = pd.DataFrame({
    "SYNTH_ID": sorted(d["SYNTH_ID"].unique()),
})
split_df["PERSON_SET"] = np.where(split_df["SYNTH_ID"].isin(holdout_ids), "HOLDOUT", "DEVELOPMENT")
split_df.to_csv(SPLIT_FILE, index=False)

# ------------------------------------------------------------------
# Prespecified feature sets.
# No model/hyperparameter is selected using validation or test results.
# ------------------------------------------------------------------
A_PCS = [f"A_FROZEN_PC{i}" for i in range(1, A_K + 1)]
G_PCS = [f"G_FROZEN_PC{i}" for i in range(1, G_K + 1)]

for c in A_PCS + G_PCS:
    d[f"{c}_X_TIME"] = d[c] * d["SIM_YEAR"]

AG_TERMS = []
for a in A_PCS:
    for g in G_PCS:
        name = f"{a}_X_{g}"
        d[name] = d[a] * d[g]
        AG_TERMS.append(name)

NUM_BASE = [
    "SOMATIC_SCORE",
    "SIM_AGE",
    "INDFMPIR",
    "BMXBMI",
    "EGFR_2021",
    "SIM_YEAR",
]
CAT_BASE = ["RIAGENDR", "RACE", "EDUC3", "SMOKING3"]

MODELS = {
    "scalar_additive": {
        "num": NUM_BASE + ["A_SCALAR", "G_SCALAR"],
        "cat": CAT_BASE,
    },
    "component_additive": {
        "num": NUM_BASE + A_PCS + G_PCS,
        "cat": CAT_BASE,
    },
    "component_time": {
        "num": NUM_BASE + A_PCS + G_PCS + [f"{c}_X_TIME" for c in A_PCS + G_PCS],
        "cat": CAT_BASE,
    },
    "component_time_ag": {
        "num": NUM_BASE + A_PCS + G_PCS + [f"{c}_X_TIME" for c in A_PCS + G_PCS] + AG_TERMS,
        "cat": CAT_BASE,
    },
}

def make_model(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])

def metric_row(y, pred):
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }

pred_parts = []
metric_rows = []

for scenario in sorted(d["SCENARIO"].unique()):
    s = d[d["SCENARIO"].eq(scenario)].copy()

    train = s[
        s["PERSON_SET"].eq("DEVELOPMENT")
        & s["SIM_YEAR"].between(0, 4)
    ].copy()

    eval_sets = {
        "validation_time_5_9": s[
            s["PERSON_SET"].eq("DEVELOPMENT")
            & s["SIM_YEAR"].between(5, 9)
        ].copy(),
        "test_future_10_14": s[
            s["PERSON_SET"].eq("DEVELOPMENT")
            & s["SIM_YEAR"].between(10, 14)
        ].copy(),
        "test_unseen_people_10_14": s[
            s["PERSON_SET"].eq("HOLDOUT")
            & s["SIM_YEAR"].between(10, 14)
        ].copy(),
    }

    # Persistence baseline needs no fitting.
    for set_name, ev in eval_sets.items():
        y = ev["TARGET_NEXT_SOMATIC"].to_numpy(float)
        pred = ev["SOMATIC_SCORE"].to_numpy(float)
        r = {
            "scenario": scenario,
            "model": "persistence",
            "eval_set": set_name,
            **metric_row(y, pred),
        }
        metric_rows.append(r)

        pred_parts.append(pd.DataFrame({
            "SYNTH_ID": ev["SYNTH_ID"].to_numpy(),
            "SCENARIO": scenario,
            "SIM_YEAR": ev["SIM_YEAR"].to_numpy(),
            "TARGET_YEAR": ev["TARGET_YEAR"].to_numpy(),
            "PERSON_SET": ev["PERSON_SET"].to_numpy(),
            "EVAL_SET": set_name,
            "MODEL": "persistence",
            "OBSERVED_NEXT_SOMATIC": y,
            "PREDICTED_NEXT_SOMATIC": pred,
        }))

    for model_name, spec in MODELS.items():
        needed = spec["num"] + spec["cat"] + ["TARGET_NEXT_SOMATIC"]
        tr = train.dropna(subset=needed).copy()
        if len(tr) == 0:
            raise RuntimeError(f"No training rows for {scenario} / {model_name}")

        model = make_model(spec["num"], spec["cat"])
        model.fit(tr[spec["num"] + spec["cat"]], tr["TARGET_NEXT_SOMATIC"])

        for set_name, ev0 in eval_sets.items():
            ev = ev0.dropna(subset=needed).copy()
            y = ev["TARGET_NEXT_SOMATIC"].to_numpy(float)
            pred = model.predict(ev[spec["num"] + spec["cat"]])

            metric_rows.append({
                "scenario": scenario,
                "model": model_name,
                "eval_set": set_name,
                **metric_row(y, pred),
            })

            pred_parts.append(pd.DataFrame({
                "SYNTH_ID": ev["SYNTH_ID"].to_numpy(),
                "SCENARIO": scenario,
                "SIM_YEAR": ev["SIM_YEAR"].to_numpy(),
                "TARGET_YEAR": ev["TARGET_YEAR"].to_numpy(),
                "PERSON_SET": ev["PERSON_SET"].to_numpy(),
                "EVAL_SET": set_name,
                "MODEL": model_name,
                "OBSERVED_NEXT_SOMATIC": y,
                "PREDICTED_NEXT_SOMATIC": pred,
            }))

metrics = pd.DataFrame(metric_rows)
preds = pd.concat(pred_parts, ignore_index=True)

metrics.to_csv(METRIC_FILE, index=False)
preds.to_parquet(PRED_FILE, index=False)

pred_sha = hashlib.sha256(PRED_FILE.read_bytes()).hexdigest()

manifest = {
    "script": "54_test_dynamic_framework.py",
    "observed_input_sha256": obs_sha,
    "hidden_truth_read": False,
    "future_nhanes_read": False,
    "frozen_representation_source": [
        str(A_PARAM.relative_to(ROOT)),
        str(G_PARAM.relative_to(ROOT)),
    ],
    "time_split": {
        "train_current_years": [0, 4],
        "validation_current_years": [5, 9],
        "test_current_years": [10, 14],
    },
    "person_holdout_fraction": HOLDOUT_FRAC,
    "person_split_seed": SEED,
    "ridge_alpha_prespecified": RIDGE_ALPHA,
    "models": list(MODELS.keys()) + ["persistence"],
    "no_validation_based_model_selection": True,
    "prediction_output": str(PRED_FILE.relative_to(ROOT)),
    "prediction_sha256": pred_sha,
    "metrics_output": str(METRIC_FILE.relative_to(ROOT)),
    "important_rule": (
        "Predictions are frozen here before Script 55 is allowed to read "
        "53_synthetic_longitudinal_truth.parquet."
    ),
}
MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"PASS development people: {len(ids) - n_hold:,}")
print(f"PASS held-out people: {n_hold:,}")
print(f"PASS scenarios tested: {d['SCENARIO'].nunique()}")
print(f"PASS frozen predictions: {len(preds):,}")
print(f"PASS metrics rows: {len(metrics):,}")
print(f"PASS predictions: {PRED_FILE}")
print(f"PASS metrics: {METRIC_FILE}")
print(f"PASS manifest: {MANIFEST_FILE}")
print(f"PASS prediction SHA256: {pred_sha}")
print()
print("Hidden truth remains unopened.")
