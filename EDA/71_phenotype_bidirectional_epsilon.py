#!/usr/bin/env python
# =============================================================================
# 71_phenotype_bidirectional_epsilon.py
#
# Prof. Saeed-style bidirectional epsilon experiment with the ACTUAL phenotype.
#
# Here:
#   X = raw physiology
#       Hb, RBC, MCV, RDW, HbA1c, fasting glucose, log fasting insulin
#
#   Y = observed phenotype score
#       primary: somatic PHQ domain
#       secondary: total PHQ-9 (if available)
#
# Exact held-out chain:
#
#   X0 --f--> Y1 --g--> X1 --f--> Y2 --g--> X2
#
# Main cycle scores:
#   epsilon_f = d(Y1, Y2)
#   epsilon_g = d(X1, X2)
#
# Crucial phenotype scores:
#   d(Y0, Y1) = initial phenotype prediction error
#   d(Y0, Y2) = full-cycle phenotype error
#
# Physiology reconstruction:
#   d(X0, X1)
#   d(X0, X2)
#
# f and g are learned ONLY on 70% of NHANES 2005-08.
# All epsilons are evaluated on untouched 30% holdout.
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "EDA" / "Results"
RES.mkdir(parents=True, exist_ok=True)

MASTER_CANDIDATES = [
    RES / "63_deep_ddx_input.csv",
    ROOT / "Results" / "63_deep_ddx_input.csv",
]
RAWBANK_CANDIDATES = [
    RES / "69_temporal_raw_bank.csv",
    ROOT / "Results" / "69_temporal_raw_bank.csv",
]

MASTER = next((p for p in MASTER_CANDIDATES if p.exists()), None)
RAWBANK = next((p for p in RAWBANK_CANDIDATES if p.exists()), None)

if MASTER is None:
    raise FileNotFoundError("Need 63_deep_ddx_input.csv")
if RAWBANK is None:
    raise FileNotFoundError(
        "Need 69_temporal_raw_bank.csv. Run Script 69 first."
    )

m = pd.read_csv(MASTER)
r = pd.read_csv(RAWBANK)

for df, name in [(m, "master"), (r, "raw bank")]:
    if "SEQN" not in df.columns:
        raise ValueError(f"{name} has no SEQN.")

m["SEQN"] = pd.to_numeric(m["SEQN"], errors="coerce")
r["SEQN"] = pd.to_numeric(r["SEQN"], errors="coerce")
m["PERIOD"] = m["PERIOD"].astype(str)

# Raw physiology.
X_COLS = [
    "LBXHGB",
    "LBXRBCSI",
    "LBXMCVSI",
    "LBXRDW",
    "LBXGH",
    "LBXGLU",
    "LOG_IN",
]

missing_x = [c for c in X_COLS if c not in r.columns]
if missing_x:
    raise ValueError(f"69 raw bank missing: {missing_x}")

# Candidate observed phenotypes.
OUTCOME_CANDIDATES = [
    ("SOMATIC", "DOMAIN_SOMATIC_SCORE_X3"),
    ("PHQ9_TOTAL", "DOMAIN_PHQ9_TOTAL_X3"),
    ("PHQ9_TOTAL_RAW", "PHQ9_TOTAL"),
]
OUTCOMES = [(label, col) for label, col in OUTCOME_CANDIDATES if col in m.columns]

if not OUTCOMES:
    raise ValueError(
        "Could not find somatic or PHQ-9 phenotype columns in Script-63 input."
    )

keep_master = ["SEQN", "PERIOD", "SURVEY_WT"] + [c for _, c in OUTCOMES]
d = m[keep_master].merge(
    r[["SEQN"] + X_COLS],
    on="SEQN",
    how="left",
    validate="one_to_one",
)

d = d[d["PERIOD"].eq("2005-2008")].copy()

def weighted_mean_sd(A, w):
    A = np.asarray(A, float)
    w = np.asarray(w, float)
    mu = np.sum(w[:, None] * A, axis=0) / np.sum(w)
    sd = np.sqrt(np.sum(w[:, None] * (A - mu) ** 2, axis=0) / np.sum(w))
    return mu, sd

def wls(A, B, w):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = B[:, None]
    D = np.column_stack([np.ones(len(A)), A])
    sw = np.sqrt(w)
    return np.linalg.lstsq(D * sw[:, None], B * sw[:, None], rcond=None)[0]

def predict(A, coef):
    A = np.asarray(A, float)
    return np.column_stack([np.ones(len(A)), A]) @ coef

def epsilon(target, estimate, w):
    target = np.asarray(target, float)
    estimate = np.asarray(estimate, float)
    if target.ndim == 1:
        target = target[:, None]
    if estimate.ndim == 1:
        estimate = estimate[:, None]
    mu = np.sum(w[:, None] * target, axis=0) / np.sum(w)
    sse = np.sum(w[:, None] * (target - estimate) ** 2)
    sst = np.sum(w[:, None] * (target - mu) ** 2)
    return float(np.sqrt(sse / sst))

def wrmse(target, estimate, w):
    target = np.asarray(target, float)
    estimate = np.asarray(estimate, float)
    if target.ndim == 1:
        target = target[:, None]
    if estimate.ndim == 1:
        estimate = estimate[:, None]
    return float(np.sqrt(
        np.sum(w[:, None] * (target - estimate) ** 2)
        / (np.sum(w) * target.shape[1])
    ))

summary = []
details = []
models = {}

for label, outcome in OUTCOMES:
    q = d.copy()
    needed = X_COLS + [outcome, "SURVEY_WT"]

    for c in needed:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q.dropna(subset=needed)
    q = q[q["SURVEY_WT"] > 0].reset_index(drop=True)

    if len(q) < 100:
        continue

    # Deterministic 70/30 split.
    rng = np.random.default_rng(42)
    idx = np.arange(len(q))
    rng.shuffle(idx)
    n_train = int(np.floor(0.70 * len(q)))
    tr_idx = idx[:n_train]
    te_idx = idx[n_train:]

    tr = q.iloc[tr_idx].copy()
    te = q.iloc[te_idx].copy()

    Xtr_raw = tr[X_COLS].to_numpy(float)
    Xte_raw = te[X_COLS].to_numpy(float)

    ytr_raw = tr[outcome].to_numpy(float)
    yte_raw = te[outcome].to_numpy(float)

    wtr = tr["SURVEY_WT"].to_numpy(float)
    wte = te["SURVEY_WT"].to_numpy(float)

    # Train-only standardisation.
    xmu, xsd = weighted_mean_sd(Xtr_raw, wtr)
    Xtr = (Xtr_raw - xmu) / xsd
    X0 = (Xte_raw - xmu) / xsd

    ymu = np.sum(wtr * ytr_raw) / np.sum(wtr)
    ysd = np.sqrt(np.sum(wtr * (ytr_raw - ymu) ** 2) / np.sum(wtr))
    if ysd <= 0:
        raise RuntimeError(f"{outcome}: zero phenotype SD.")

    ytr = (ytr_raw - ymu) / ysd
    Y0 = ((yte_raw - ymu) / ysd)[:, None]

    # Learn on training data only:
    # f : physiology -> phenotype
    # g : phenotype -> physiology
    f = wls(Xtr, ytr, wtr)
    g = wls(ytr[:, None], Xtr, wtr)

    # Exact four-step holdout loop.
    Y1 = predict(X0, f)
    X1 = predict(Y1, g)
    Y2 = predict(X1, f)
    X2 = predict(Y2, g)

    tests = [
        ("MAIN_epsilon_f__Y1_vs_Y2", Y1, Y2),
        ("MAIN_epsilon_g__X1_vs_X2", X1, X2),
        ("PHENOTYPE_initial__Y0_vs_Y1", Y0, Y1),
        ("PHENOTYPE_full__Y0_vs_Y2", Y0, Y2),
        ("PHYSIOLOGY_initial__X0_vs_X1", X0, X1),
        ("PHYSIOLOGY_full__X0_vs_X2", X0, X2),
    ]

    for test, target, estimate in tests:
        ep = epsilon(target, estimate, wte)
        summary.append({
            "outcome_label": label,
            "outcome_column": outcome,
            "test": test,
            "epsilon": ep,
            "R2_equivalent": 1.0 - ep**2,
            "weighted_RMSE_standardized": wrmse(target, estimate, wte),
            "n_train": len(tr),
            "n_test": len(te),
        })

    # Outcome RMSE in original phenotype-score units.
    Y1_raw = Y1[:, 0] * ysd + ymu
    Y2_raw = Y2[:, 0] * ysd + ymu

    for tag, pred_raw in [
        ("Y0_vs_Y1", Y1_raw),
        ("Y0_vs_Y2", Y2_raw),
    ]:
        mu_test = np.sum(wte * yte_raw) / np.sum(wte)
        sse = np.sum(wte * (yte_raw - pred_raw) ** 2)
        sst = np.sum(wte * (yte_raw - mu_test) ** 2)
        details.append({
            "outcome_label": label,
            "comparison": tag,
            "variable": outcome,
            "epsilon": float(np.sqrt(sse / sst)),
            "R2_equivalent": float(1.0 - sse / sst),
            "RMSE_raw_units": float(np.sqrt(sse / np.sum(wte))),
            "n_test": len(te),
        })

    # Physiology reconstruction in original biomarker units.
    X1_raw = X1 * xsd + xmu
    X2_raw = X2 * xsd + xmu

    for j, var in enumerate(X_COLS):
        for tag, pred_raw in [
            ("X0_vs_X1", X1_raw[:, j]),
            ("X0_vs_X2", X2_raw[:, j]),
        ]:
            target = Xte_raw[:, j]
            mu_test = np.sum(wte * target) / np.sum(wte)
            sse = np.sum(wte * (target - pred_raw) ** 2)
            sst = np.sum(wte * (target - mu_test) ** 2)
            details.append({
                "outcome_label": label,
                "comparison": tag,
                "variable": var,
                "epsilon": float(np.sqrt(sse / sst)),
                "R2_equivalent": float(1.0 - sse / sst),
                "RMSE_raw_units": float(np.sqrt(sse / np.sum(wte))),
                "n_test": len(te),
            })

    models[label] = {
        "outcome_column": outcome,
        "complete_n": int(len(q)),
        "train_n": int(len(tr)),
        "test_n": int(len(te)),
        "X_dimension": len(X_COLS),
        "Y_dimension": 1,
        "X_variables": X_COLS,
        "phenotype_train_mean": float(ymu),
        "phenotype_train_sd": float(ysd),
    }

summary_df = pd.DataFrame(summary)
details_df = pd.DataFrame(details)

summary_df.to_csv(RES / "71_phenotype_bidirectional_epsilon_summary.csv", index=False)
details_df.to_csv(RES / "71_phenotype_bidirectional_epsilon_details.csv", index=False)

manifest = {
    "script": "71_phenotype_bidirectional_epsilon.py",
    "experiment": "Actual phenotype as Y",
    "chain": "X0 --f--> Y1 --g--> X1 --f--> Y2 --g--> X2",
    "X": "raw 7-variable physiology",
    "Y": "observed depressive phenotype score",
    "training": "70% of complete NHANES 2005-08 only",
    "testing": "untouched 30% NHANES 2005-08 holdout",
    "main_scores": {
        "epsilon_f": "distance(Y1,Y2)",
        "epsilon_g": "distance(X1,X2)",
        "phenotype_initial": "distance(Y0,Y1)",
        "phenotype_full": "distance(Y0,Y2)",
        "physiology_initial": "distance(X0,X1)",
        "physiology_full": "distance(X0,X2)",
    },
    "warning":
        "Because Y is one phenotype score and X is seven-dimensional physiology, "
        "g:Y->X is not a unique inverse. It estimates the physiology predictable "
        "from the phenotype, not full physiological recovery.",
    "locked_outputs_modified": False,
    "models": models,
}
(RES / "71_phenotype_bidirectional_epsilon_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

print("PASS  Script 71 phenotype bidirectional epsilon experiment complete.")
print("PASS  Here Y is the ACTUAL observed phenotype, not ICA/latent Y.")
print("PASS  f and g learned on 70% of 2005-08; evaluated on untouched 30%.")
print("PASS  Locked Results untouched.")
print()
print(summary_df.to_string(index=False))
