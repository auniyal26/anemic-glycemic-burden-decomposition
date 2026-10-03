#!/usr/bin/env python
# =============================================================================
# 83_conformal_reverse_bridge.py
#
# Goal:
#   Calibrate the probabilistic reverse bridge Y -> plausible Z region without
#   assuming Gaussian tail probabilities.
#
# Design:
#   2005-06 = train conditional mean E[Z|Y] + residual covariance
#   2007-08 = calibrate conformal Mahalanobis thresholds
#   2009-18 / 2021-23 = fully frozen evaluation
#
# Primary phenotype state:
#   Y2 = [SOMATIC_SCORE, COGAFF_SUM]
#
# This is an uncertainty/calibration test, not causal.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    RESULTS / "73_frozen_phenotype_map_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
]
INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), None)

if INPUT is None:
    raise FileNotFoundError(
        "Need EDA/Results/73_frozen_phenotype_map_input.csv. Run Script 73 first."
    )

ZCOLS = [
    "SP_SHARED1", "SP_SHARED2",
    "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]
YCOLS = ["SOMATIC_SCORE", "COGAFF_SUM"]

TARGET_LEVELS = [0.50, 0.80, 0.95]
EVAL_PERIODS = ["2009-2018", "2021-2023"]


def weighted_mean_sd(X, w):
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    sw = w.sum()
    mu = np.sum(w[:, None] * X, axis=0) / sw
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def stdize(X, mu, sd):
    return (np.asarray(X, float) - mu) / sd


def wmse(Y, P, w):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    w = np.asarray(w, float)
    return float(
        np.sum(w[:, None] * (Y - P) ** 2)
        / (np.sum(w) * Y.shape[1])
    )


def tune_ridge(A, B, w, seed=42):
    grid = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    scored = []
    for alpha in grid:
        vals = []
        for tr, va in kf.split(A):
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(A[tr], B[tr], sample_weight=w[tr])
            vals.append(wmse(B[va], m.predict(A[va]), w[va]))
        scored.append((float(np.mean(vals)), alpha))

    scored.sort(key=lambda x: x[0])
    best = scored[0][1]

    m = Ridge(alpha=best, fit_intercept=True)
    m.fit(A, B, sample_weight=w)
    return m, best


def weighted_cov(E, w):
    E = np.asarray(E, float)
    w = np.asarray(w, float)
    sw = w.sum()

    mu = np.sum(w[:, None] * E, axis=0) / sw
    Ec = E - mu
    cov = (Ec * w[:, None]).T @ Ec / sw

    eps = max(1e-8, 1e-6 * np.trace(cov) / cov.shape[0])
    cov = cov + np.eye(cov.shape[0]) * eps
    return mu, cov


def mahal2(E, mu, cov):
    Ec = np.asarray(E, float) - mu
    inv = np.linalg.inv(cov)
    return np.einsum("ni,ij,nj->n", Ec, inv, Ec)


def conformal_quantile(scores, coverage):
    """
    Split-conformal finite-sample quantile:
    k = ceil((n+1)*coverage), capped at n.
    """
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = min(n, math.ceil((n + 1) * coverage))
    return float(s[k - 1])


def weighted_rate(hit, w):
    hit = np.asarray(hit, float)
    w = np.asarray(w, float)
    return float(np.sum(w * hit) / np.sum(w))


# -----------------------------------------------------------------------------
# Load and clean.
# -----------------------------------------------------------------------------
d = pd.read_csv(INPUT)

required = (
    ["CYCLE", "PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG"]
    + ZCOLS + YCOLS
)
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

d["CYCLE"] = (
    d["CYCLE"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(4)
)

for c in ["SURVEY_WT"] + ZCOLS + YCOLS:
    d[c] = pd.to_numeric(d[c], errors="coerce")

d = d[
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[ZCOLS + YCOLS + ["SURVEY_WT"]].notna().all(axis=1)
    & (d["SURVEY_WT"] > 0)
].copy()

train = d[d["CYCLE"].eq("0506")].copy()
cal = d[d["CYCLE"].eq("0708")].copy()

if len(train) < 100 or len(cal) < 100:
    raise RuntimeError(
        f"Need adequate 0506 train and 0708 calibration samples; "
        f"got train={len(train)}, calibration={len(cal)}"
    )

# -----------------------------------------------------------------------------
# Train mean model + covariance on 2005-06 only.
# -----------------------------------------------------------------------------
Ztr = train[ZCOLS].to_numpy(float)
Ytr = train[YCOLS].to_numpy(float)
wtr = train["SURVEY_WT"].to_numpy(float)

zmu, zsd = weighted_mean_sd(Ztr, wtr)
ymu, ysd = weighted_mean_sd(Ytr, wtr)

Ztrs = stdize(Ztr, zmu, zsd)
Ytrs = stdize(Ytr, ymu, ysd)

mean_model, alpha = tune_ridge(Ytrs, Ztrs, wtr, seed=42)
Ztr_hat = mean_model.predict(Ytrs)

resid_mu, resid_cov = weighted_cov(Ztrs - Ztr_hat, wtr)

# -----------------------------------------------------------------------------
# Calibrate nonparametric score cutoffs on independent 2007-08.
# -----------------------------------------------------------------------------
Zcal = stdize(cal[ZCOLS].to_numpy(float), zmu, zsd)
Ycal = stdize(cal[YCOLS].to_numpy(float), ymu, ysd)
wcal = cal["SURVEY_WT"].to_numpy(float)

Zcal_hat = mean_model.predict(Ycal)
cal_scores = mahal2(Zcal - Zcal_hat, resid_mu, resid_cov)

thresholds = {
    level: conformal_quantile(cal_scores, level)
    for level in TARGET_LEVELS
}

# -----------------------------------------------------------------------------
# Evaluate calibration + future transfer.
# -----------------------------------------------------------------------------
rows = []

eval_sets = [
    ("2007-2008_CAL", cal),
    ("2009-2018", d[d["PERIOD"].eq("2009-2018")].copy()),
    ("2021-2023", d[d["PERIOD"].eq("2021-2023")].copy()),
]

for label, q in eval_sets:
    if len(q) < 50:
        continue

    Z = stdize(q[ZCOLS].to_numpy(float), zmu, zsd)
    Y = stdize(q[YCOLS].to_numpy(float), ymu, ysd)
    w = q["SURVEY_WT"].to_numpy(float)

    Zhat = mean_model.predict(Y)
    scores = mahal2(Z - Zhat, resid_mu, resid_cov)

    row = {
        "period": label,
        "n": len(q),
        "mean_mahalanobis2": float(np.mean(scores)),
        "weighted_mean_mahalanobis2": float(np.sum(w * scores) / np.sum(w)),
    }

    for level, cutoff in thresholds.items():
        hit = scores <= cutoff
        pct = int(level * 100)

        row[f"nominal_{pct}"] = level
        row[f"cutoff_{pct}"] = cutoff
        row[f"coverage_{pct}_unweighted"] = float(hit.mean())
        row[f"coverage_{pct}_weighted"] = weighted_rate(hit, w)

    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(
    RESULTS / "83_conformal_reverse_bridge_summary.csv",
    index=False,
)

threshold_df = pd.DataFrame([
    {
        "target_coverage": level,
        "conformal_cutoff_mahalanobis2": cutoff,
        "calibration_n": len(cal),
    }
    for level, cutoff in thresholds.items()
])
threshold_df.to_csv(
    RESULTS / "83_conformal_reverse_bridge_thresholds.csv",
    index=False,
)

manifest = {
    "script": "83_conformal_reverse_bridge.py",
    "train_cycle": "2005-2006",
    "calibration_cycle": "2007-2008",
    "future_evaluation": EVAL_PERIODS,
    "Y": YCOLS,
    "Z": ZCOLS,
    "mean_model": "weighted ridge",
    "score": "Mahalanobis squared residual using 2005-06 covariance",
    "calibration": "split conformal using 2007-08 empirical score quantiles",
    "exact_conformal_note": (
        "Finite-sample conformal guarantee applies to exchangeable unweighted "
        "calibration/test observations; survey-weighted coverage is reported "
        "descriptively."
    ),
    "causal": False,
}
(RESULTS / "83_conformal_reverse_bridge_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 108)
print("SCRIPT 83 — CONFORMAL REVERSE BRIDGE")
print("=" * 108)
print(f"Train 2005-06 n={len(train)}")
print(f"Calibrate 2007-08 n={len(cal)}")
print(f"Ridge alpha={alpha:g}")
print()

print("CONFORMAL THRESHOLDS")
for level in TARGET_LEVELS:
    print(
        f"  {int(level*100)}% region: "
        f"Mahalanobis2 cutoff={thresholds[level]:.3f}"
    )
print()

for _, r in summary.iterrows():
    print(f"[{r['period']}] n={int(r['n'])}")
    for level in TARGET_LEVELS:
        pct = int(level * 100)
        print(
            f"  nominal {pct}% -> "
            f"coverage unweighted={r[f'coverage_{pct}_unweighted']:.3f}, "
            f"weighted={r[f'coverage_{pct}_weighted']:.3f}"
        )
    print(
        f"  weighted mean Mahalanobis2="
        f"{r['weighted_mean_mahalanobis2']:.3f}"
    )
    print()

print("READING GUIDE")
print("  Future coverage close to nominal -> calibrated reverse region transfers.")
print("  Future coverage below nominal -> region is too narrow under time shift.")
print("  Future coverage above nominal -> region is conservative/wider than needed.")
print()
print("This is the first genuinely out-of-period uncertainty calibration test:")
print("2005-06 learns the reverse map; 2007-08 calibrates its region; later periods are untouched.")
print("=" * 108)
