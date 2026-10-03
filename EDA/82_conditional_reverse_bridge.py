#!/usr/bin/env python
# =============================================================================
# 82_conditional_reverse_bridge.py
#
# Goal:
#   Replace impossible deterministic Y -> Z inversion with a conditional
#   distribution p(Z | Y).
#
# Compare:
#   Y1 = PHQ-9 total
#   Y2 = [somatic, cognitive-affective]
#   Y9 = all 9 PHQ items
#
# Train ONLY on 2005-08.
# Freeze mean model + residual covariance into 2009-18 and 2021-23.
#
# Main outputs:
#   - mean prediction R2
#   - average Mahalanobis distance
#   - 50/80/95% ellipsoid coverage
#   - Gaussian negative log-likelihood
#
# This is a probabilistic reconstruction/calibration test, not causal.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import chi2

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

PHQ = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]

PERIODS = ["2005-2008", "2009-2018", "2021-2023"]

DISC = {"0506": "D", "0708": "E"}
TEMP = {
    "0910": "F",
    "1112": "G",
    "1314": "H",
    "1516": "I",
    "1718": "J",
    "2123": "L",
}
ALL_CYCLES = list(DISC) + list(TEMP)

DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"


def xpt_path(base: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")


def base_for(cycle: str) -> Path:
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def suffix_for(cycle: str) -> str:
    return (DISC if cycle in DISC else TEMP)[cycle]


def weighted_mean_sd(X, w):
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    sw = w.sum()

    mu = np.sum(w[:, None] * X, axis=0) / sw
    sd = np.sqrt(
        np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw
    )
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def standardize(X, mu, sd):
    return (np.asarray(X, float) - mu) / sd


def weighted_mse(Y, P, w):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if P.ndim == 1:
        P = P[:, None]
    w = np.asarray(w, float)

    return float(
        np.sum(w[:, None] * (Y - P) ** 2)
        / (np.sum(w) * Y.shape[1])
    )


def multivariate_r2(Y, P, w):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if P.ndim == 1:
        P = P[:, None]
    w = np.asarray(w, float)

    mu = np.sum(w[:, None] * Y, axis=0) / np.sum(w)
    sse = np.sum(w[:, None] * (Y - P) ** 2)
    sst = np.sum(w[:, None] * (Y - mu) ** 2)

    return float(1 - sse / sst) if sst > 0 else np.nan


def tune_ridge(A, B, w, seed=42):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    w = np.asarray(w, float)

    grid = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    scores = []

    for alpha in grid:
        fold_scores = []

        for tr, va in kf.split(A):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(A[tr], B[tr], sample_weight=w[tr])

            pred = model.predict(A[va])
            fold_scores.append(
                weighted_mse(B[va], pred, w[va])
            )

        scores.append((float(np.mean(fold_scores)), alpha))

    scores.sort(key=lambda x: x[0])
    best_alpha = scores[0][1]

    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(A, B, sample_weight=w)

    return model, best_alpha


def weighted_cov(E, w):
    E = np.asarray(E, float)
    w = np.asarray(w, float)

    sw = w.sum()
    mu = np.sum(w[:, None] * E, axis=0) / sw
    Ec = E - mu

    cov = (Ec * w[:, None]).T @ Ec / sw

    # Small numerical ridge only to guarantee invertibility.
    eps = max(1e-8, 1e-6 * np.trace(cov) / cov.shape[0])
    cov = cov + np.eye(cov.shape[0]) * eps

    return mu, cov


def mahalanobis_squared(E, mu, cov):
    Ec = np.asarray(E, float) - mu
    inv = np.linalg.inv(cov)
    return np.einsum("ni,ij,nj->n", Ec, inv, Ec)


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    return float(np.sum(w * x) / np.sum(w))


def gaussian_nll(E, mu, cov):
    E = np.asarray(E, float)
    p = cov.shape[0]

    d2 = mahalanobis_squared(E, mu, cov)
    sign, logdet = np.linalg.slogdet(cov)

    if sign <= 0:
        raise RuntimeError("Residual covariance is not positive definite.")

    return 0.5 * (
        p * np.log(2 * np.pi)
        + logdet
        + d2
    )


# -----------------------------------------------------------------------------
# 1. Load frozen Z scaffold
# -----------------------------------------------------------------------------

d = pd.read_csv(INPUT)

required = [
    "SEQN", "PERIOD", "SURVEY_WT",
    "ELIGIBLE_ADULT_NONPREG",
] + ZCOLS

missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Input missing required columns: {missing}")

d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce")

# Drop any existing PHQ item columns so raw DPQ is authoritative.
drop_existing = [c for c in PHQ if c in d.columns]
if drop_existing:
    d = d.drop(columns=drop_existing)

# -----------------------------------------------------------------------------
# 2. Restore raw PHQ items
# -----------------------------------------------------------------------------

frames = []

for cycle in ALL_CYCLES:
    s = suffix_for(cycle)
    p = xpt_path(base_for(cycle), f"DPQ_{s}")

    q = pd.read_sas(p, format="xport")
    have = [c for c in PHQ if c in q.columns]

    if len(have) != 9:
        raise ValueError(f"{cycle}: expected all 9 PHQ items.")

    q = q[["SEQN"] + PHQ].copy()
    q["SEQN"] = pd.to_numeric(q["SEQN"], errors="coerce")

    for c in PHQ:
        x = pd.to_numeric(q[c], errors="coerce")
        x.loc[x.abs() < 1e-10] = 0
        x.loc[~x.isin([0, 1, 2, 3])] = np.nan
        q[c] = x

    frames.append(q)

phq = pd.concat(frames, ignore_index=True)

if phq["SEQN"].duplicated().any():
    raise RuntimeError("Duplicate SEQN found in restored PHQ bank.")

d = d.merge(
    phq,
    on="SEQN",
    how="left",
    validate="one_to_one",
)

d["PHQ9_TOTAL_BRIDGE"] = d[PHQ].sum(axis=1, min_count=9)
d["SOMATIC_BRIDGE"] = d[SOMATIC].sum(axis=1, min_count=3)
d["COGAFF_BRIDGE"] = (
    d["PHQ9_TOTAL_BRIDGE"] - d["SOMATIC_BRIDGE"]
)

REPRESENTATIONS = {
    "Y1_total": ["PHQ9_TOTAL_BRIDGE"],
    "Y2_somatic_cogaff": [
        "SOMATIC_BRIDGE",
        "COGAFF_BRIDGE",
    ],
    "Y9_items": PHQ,
}

# -----------------------------------------------------------------------------
# 3. Common complete cohort
# -----------------------------------------------------------------------------

d = d[
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[PHQ + ZCOLS + ["SURVEY_WT"]].notna().all(axis=1)
    & (pd.to_numeric(d["SURVEY_WT"], errors="coerce") > 0)
].copy()

disc = d[d["PERIOD"].eq("2005-2008")].copy()

if len(disc) < 100:
    raise RuntimeError(
        f"Too few complete discovery observations: n={len(disc)}"
    )

# -----------------------------------------------------------------------------
# 4. Discovery-only p(Z|Y)
# -----------------------------------------------------------------------------

summary_rows = []
model_rows = []

coverage_levels = [0.50, 0.80, 0.95]

for rep, ycols in REPRESENTATIONS.items():

    Z0 = disc[ZCOLS].to_numpy(float)
    Y0 = disc[ycols].to_numpy(float)
    w0 = disc["SURVEY_WT"].to_numpy(float)

    zmu, zsd = weighted_mean_sd(Z0, w0)
    ymu, ysd = weighted_mean_sd(Y0, w0)

    Z0s = standardize(Z0, zmu, zsd)
    Y0s = standardize(Y0, ymu, ysd)

    # Conditional mean E[Z|Y]
    mean_model, alpha = tune_ridge(
        Y0s, Z0s, w0, seed=42
    )

    Z0_mean = mean_model.predict(Y0s)
    residual0 = Z0s - Z0_mean

    resid_mu, resid_cov = weighted_cov(
        residual0, w0
    )

    rank = int(
        np.linalg.matrix_rank(
            np.atleast_2d(mean_model.coef_)
        )
    )

    model_rows.append({
        "representation": rep,
        "Y_dimension": len(ycols),
        "Z_dimension": len(ZCOLS),
        "rank_conditional_mean": rank,
        "ridge_alpha": alpha,
        "discovery_n": len(disc),
        "residual_cov_condition_number":
            float(np.linalg.cond(resid_cov)),
    })

    for period in PERIODS:

        q = d[d["PERIOD"].eq(period)].copy()
        if len(q) < 50:
            continue

        Z = q[ZCOLS].to_numpy(float)
        Y = q[ycols].to_numpy(float)
        w = q["SURVEY_WT"].to_numpy(float)

        Zs = standardize(Z, zmu, zsd)
        Ys = standardize(Y, ymu, ysd)

        Zmean = mean_model.predict(Ys)
        E = Zs - Zmean

        d2 = mahalanobis_squared(
            E, resid_mu, resid_cov
        )
        nll = gaussian_nll(
            E, resid_mu, resid_cov
        )

        row = {
            "representation": rep,
            "period": period,
            "n": len(q),
            "conditional_mean_R2":
                multivariate_r2(Zs, Zmean, w),
            "weighted_mean_Mahalanobis2":
                weighted_mean(d2, w),
            "weighted_mean_Gaussian_NLL":
                weighted_mean(nll, w),
        }

        for level in coverage_levels:
            cutoff = chi2.ppf(level, df=len(ZCOLS))
            covered = (d2 <= cutoff).astype(float)

            row[
                f"ellipsoid_coverage_{int(level*100)}"
            ] = weighted_mean(covered, w)

        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
models = pd.DataFrame(model_rows)

summary.to_csv(
    RESULTS / "82_conditional_reverse_bridge_summary.csv",
    index=False,
)

models.to_csv(
    RESULTS / "82_conditional_reverse_bridge_models.csv",
    index=False,
)

manifest = {
    "script": "82_conditional_reverse_bridge.py",
    "training_period": "2005-2008 only",
    "future_refit": False,
    "target": "p(Z|Y)",
    "Z": ZCOLS,
    "Y_representations": REPRESENTATIONS,
    "distribution": (
        "multivariate Gaussian residual around "
        "ridge conditional mean E[Z|Y]"
    ),
    "coverage_levels": coverage_levels,
    "causal": False,
}

(RESULTS / "82_conditional_reverse_bridge_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 108)
print("SCRIPT 82 — CONDITIONAL REVERSE BRIDGE p(Z | Y)")
print("=" * 108)
print(f"Common discovery cohort n={len(disc)}")
print("Training: 2005-08 only. Future periods frozen.")
print()

for rep in [
    "Y1_total",
    "Y2_somatic_cogaff",
    "Y9_items",
]:
    print(f"[{rep}]")

    m = models[
        models["representation"].eq(rep)
    ].iloc[0]

    print(
        f"  Ydim={int(m.Y_dimension)}, "
        f"rank(mean map)={int(m.rank_conditional_mean)}, "
        f"ridge alpha={m.ridge_alpha:g}"
    )

    q = summary[
        summary["representation"].eq(rep)
    ]

    for period in PERIODS:
        z = q[q["period"].eq(period)]
        if z.empty:
            continue

        r = z.iloc[0]

        print(
            f"  {period}: "
            f"mean R2={r.conditional_mean_R2:.4f} | "
            f"Mahalanobis2={r.weighted_mean_Mahalanobis2:.3f} | "
            f"coverage50={r.ellipsoid_coverage_50:.3f} | "
            f"coverage80={r.ellipsoid_coverage_80:.3f} | "
            f"coverage95={r.ellipsoid_coverage_95:.3f} | "
            f"NLL={r.weighted_mean_Gaussian_NLL:.3f}"
        )

    print()

print("HOW TO READ THIS")
print("  Mean R2 can remain low; that is expected.")
print("  The key test is whether the TRUE Z falls inside the predicted")
print("  conditional region at about the advertised rate.")
print("  Example: coverage95 near 0.95 means p(Z|Y) is reasonably calibrated.")
print("  Mahalanobis2 near 7 is expected for a calibrated 7D Gaussian.")
print()
print("If coverage transfers, we have a usable probabilistic reverse bridge")
print("even though exact deterministic inversion is impossible.")
print("=" * 108)
