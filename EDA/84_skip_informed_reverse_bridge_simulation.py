#!/usr/bin/env python
# =============================================================================
# 84_skip_informed_reverse_bridge_simulation.py
#
# Goal:
#   Test the "DL-style skip connection" idea for the reverse bridge:
#
#       phenotype-only:      p(Z_t | Y_t)
#       prior-only:          p(Z_t | Z_{t-1})
#       informed bridge:     p(Z_t | Y_t, Z_{t-1})
#
# IMPORTANT:
#   NHANES is not longitudinal at the person level. Therefore this script is
#   explicitly a CONTROLLED SIMULATION grounded in the empirical 2005-08 Z/Y
#   distribution. It does NOT claim observed longitudinal evidence.
#
# Simulation:
#   1) Learn empirical forward phenotype map Y|Z on 2005-08.
#   2) Sample Z_{t-1} from real observed physiological states.
#   3) Generate Z_t with controlled continuity rho:
#
#          Z_t = rho * Z_{t-1} + sqrt(1-rho^2) * Z_innovation
#
#      where Z_innovation is another real observed standardized Z state.
#   4) Generate Y_t from the learned forward map plus empirically resampled
#      phenotype residuals.
#   5) Compare reverse models on held-out simulated subjects.
#
# Outputs:
#   - mean reconstruction R2
#   - conformal 95% coverage
#   - uncertainty sharpness (average residual SD + log-volume proxy)
#
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

RHOS = [0.00, 0.25, 0.50, 0.75, 0.90]
N_SIM = 30000
SEED = 84


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


def r2_multi(Y, P):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    mu = Y.mean(axis=0)
    sse = np.sum((Y - P) ** 2)
    sst = np.sum((Y - mu) ** 2)
    return float(1 - sse / sst) if sst > 0 else np.nan


def mse_multi(Y, P):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    return float(np.mean((Y - P) ** 2))


def tune_ridge(A, B, seed=42):
    grid = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scored = []

    for alpha in grid:
        vals = []
        for tr, va in kf.split(A):
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(A[tr], B[tr])
            vals.append(mse_multi(B[va], m.predict(A[va])))
        scored.append((float(np.mean(vals)), alpha))

    scored.sort(key=lambda x: x[0])
    best = scored[0][1]

    m = Ridge(alpha=best, fit_intercept=True)
    m.fit(A, B)
    return m, best


def residual_cov(E):
    E = np.asarray(E, float)
    mu = E.mean(axis=0)
    Ec = E - mu
    cov = Ec.T @ Ec / len(E)
    eps = max(1e-8, 1e-6 * np.trace(cov) / cov.shape[0])
    cov = cov + np.eye(cov.shape[0]) * eps
    return mu, cov


def mahal2(E, mu, cov):
    Ec = np.asarray(E, float) - mu
    inv = np.linalg.inv(cov)
    return np.einsum("ni,ij,nj->n", Ec, inv, Ec)


def conformal_quantile(scores, coverage=0.95):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = min(n, math.ceil((n + 1) * coverage))
    return float(s[k - 1])


def sharpness(cov, cutoff):
    """
    Two simple uncertainty-width summaries:
      avg_residual_sd : average marginal residual SD
      log_volume_proxy: proportional to log ellipsoid volume,
                        ignoring constants common to all models.
    """
    p = cov.shape[0]
    avg_sd = float(np.mean(np.sqrt(np.diag(cov))))
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise RuntimeError("Residual covariance not positive definite.")
    log_volume_proxy = float(0.5 * logdet + 0.5 * p * np.log(cutoff))
    return avg_sd, log_volume_proxy


# -----------------------------------------------------------------------------
# 1. Empirical discovery scaffold
# -----------------------------------------------------------------------------
d = pd.read_csv(INPUT)

required = (
    ["PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG"]
    + ZCOLS + YCOLS
)
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for c in ["SURVEY_WT"] + ZCOLS + YCOLS:
    d[c] = pd.to_numeric(d[c], errors="coerce")

d = d[
    d["PERIOD"].eq("2005-2008")
    & d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[ZCOLS + YCOLS + ["SURVEY_WT"]].notna().all(axis=1)
    & (d["SURVEY_WT"] > 0)
].copy()

if len(d) < 500:
    raise RuntimeError(f"Too few discovery rows: n={len(d)}")

Z = d[ZCOLS].to_numpy(float)
Y = d[YCOLS].to_numpy(float)
w = d["SURVEY_WT"].to_numpy(float)

zmu, zsd = weighted_mean_sd(Z, w)
ymu, ysd = weighted_mean_sd(Y, w)

Zs = stdize(Z, zmu, zsd)
Ys = stdize(Y, ymu, ysd)

# -----------------------------------------------------------------------------
# 2. Empirical forward phenotype generator Y|Z
# -----------------------------------------------------------------------------
forward, alpha_forward = tune_ridge(Zs, Ys, seed=841)
Yhat = forward.predict(Zs)
Yresid = Ys - Yhat
Yresid = Yresid - Yresid.mean(axis=0, keepdims=True)

# Sampling weights preserve NHANES discovery composition approximately.
prob = w / w.sum()

rng = np.random.default_rng(SEED)

# Fixed simulated subject ingredients so rho comparisons are paired/fair.
idx_prev = rng.choice(len(d), size=N_SIM, replace=True, p=prob)
idx_innov = rng.choice(len(d), size=N_SIM, replace=True, p=prob)
idx_yres = rng.choice(len(d), size=N_SIM, replace=True, p=prob)

Zprev_base = Zs[idx_prev]
Zinnov_base = Zs[idx_innov]
Ynoise_base = Yresid[idx_yres]

# Fixed train/cal/test split across all rho values.
perm = rng.permutation(N_SIM)
n_train = int(0.60 * N_SIM)
n_cal = int(0.20 * N_SIM)

itr = perm[:n_train]
ica = perm[n_train:n_train + n_cal]
ite = perm[n_train + n_cal:]

# -----------------------------------------------------------------------------
# 3. Compare reverse models across physiological continuity strengths.
# -----------------------------------------------------------------------------
rows = []
model_rows = []

for rho in RHOS:

    Zprev = Zprev_base.copy()
    Zt = (
        rho * Zprev_base
        + np.sqrt(max(0.0, 1.0 - rho ** 2)) * Zinnov_base
    )

    # Phenotype at current time generated from current physiology + empirical
    # phenotype residual. No current-Z information is supplied to reverse models.
    Yt = forward.predict(Zt) + Ynoise_base

    designs = {
        "Y_only": Yt,
        "Zprev_only": Zprev,
        "Y_plus_Zprev": np.column_stack([Yt, Zprev]),
    }

    rho_results = {}

    for model_name, X in designs.items():

        reverse, alpha = tune_ridge(
            X[itr], Zt[itr], seed=842
        )

        # Independent calibration split.
        Zcal_hat = reverse.predict(X[ica])
        Ecal = Zt[ica] - Zcal_hat
        emu, ecov = residual_cov(Ecal)

        cal_scores = mahal2(Ecal, emu, ecov)
        cutoff95 = conformal_quantile(cal_scores, 0.95)

        # Untouched test split.
        Ztest_hat = reverse.predict(X[ite])
        Etest = Zt[ite] - Ztest_hat
        test_scores = mahal2(Etest, emu, ecov)

        coverage95 = float(np.mean(test_scores <= cutoff95))
        test_r2 = r2_multi(Zt[ite], Ztest_hat)

        avg_sd, logvol = sharpness(ecov, cutoff95)

        result = {
            "rho": rho,
            "model": model_name,
            "test_n": len(ite),
            "test_R2": test_r2,
            "coverage95": coverage95,
            "avg_residual_SD": avg_sd,
            "log_volume_proxy": logvol,
            "ridge_alpha": alpha,
            "conformal_cutoff95": cutoff95,
        }
        rows.append(result)
        rho_results[model_name] = result

        model_rows.append({
            "rho": rho,
            "model": model_name,
            "input_dimension": X.shape[1],
            "ridge_alpha": alpha,
            "rank_reverse_map": int(
                np.linalg.matrix_rank(np.atleast_2d(reverse.coef_))
            ),
        })

    # Add relative sharpness versus phenotype-only baseline.
    base_logvol = rho_results["Y_only"]["log_volume_proxy"]
    for row in rows[-3:]:
        row["relative_volume_vs_Y_only"] = float(
            np.exp(row["log_volume_proxy"] - base_logvol)
        )

summary = pd.DataFrame(rows)
models = pd.DataFrame(model_rows)

summary.to_csv(
    RESULTS / "84_skip_informed_reverse_bridge_simulation.csv",
    index=False,
)
models.to_csv(
    RESULTS / "84_skip_informed_reverse_bridge_models.csv",
    index=False,
)

manifest = {
    "script": "84_skip_informed_reverse_bridge_simulation.py",
    "status": "controlled simulation, not observed longitudinal evidence",
    "empirical_source": "NHANES 2005-08 frozen Z and phenotype state",
    "Y": YCOLS,
    "Z": ZCOLS,
    "rho_values": RHOS,
    "n_simulated": N_SIM,
    "forward_generator": "ridge Y|Z + empirically resampled phenotype residuals",
    "transition": "Z_t = rho*Z_tminus1 + sqrt(1-rho^2)*empirical_Z_innovation",
    "reverse_models": [
        "p(Z_t|Y_t)",
        "p(Z_t|Z_tminus1)",
        "p(Z_t|Y_t,Z_tminus1)",
    ],
    "train_cal_test": "60/20/20 fixed split",
    "coverage": "95% split-conformal Mahalanobis region",
    "causal": False,
}

(RESULTS / "84_skip_informed_reverse_bridge_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 112)
print("SCRIPT 84 — SKIP-INFORMED REVERSE BRIDGE SIMULATION")
print("=" * 112)
print(f"Empirical discovery rows = {len(d)}")
print(f"Simulated subjects = {N_SIM}")
print(f"Forward phenotype ridge alpha = {alpha_forward:g}")
print()
print("THIS IS A CONTROLLED SIMULATION.")
print("NHANES does not observe the same individuals longitudinally.")
print()

for rho in RHOS:
    q = summary[summary["rho"].eq(rho)]
    print(f"[rho={rho:.2f}] physiological continuity")

    for model_name in ["Y_only", "Zprev_only", "Y_plus_Zprev"]:
        r = q[q["model"].eq(model_name)].iloc[0]
        print(
            f"  {model_name:14s} | "
            f"R2={r.test_R2:.4f} | "
            f"cov95={r.coverage95:.3f} | "
            f"avgSD={r.avg_residual_SD:.3f} | "
            f"relVolume={r.relative_volume_vs_Y_only:.3f}"
        )

    informed = q[q["model"].eq("Y_plus_Zprev")].iloc[0]
    yonly = q[q["model"].eq("Y_only")].iloc[0]

    print(
        f"  informed gain vs Y-only: "
        f"ΔR2={informed.test_R2 - yonly.test_R2:+.4f}, "
        f"volume ratio={informed.relative_volume_vs_Y_only:.3f}"
    )
    print()

print("INTERPRETATION")
print("  If Y+Zprev keeps ~95% coverage while R2 rises and relative volume falls,")
print("  the previous physiological state acts like a useful skip/reference channel.")
print()
print("  rho=0 is the negative control: previous physiology contains no true temporal")
print("  continuity and should add little or nothing.")
print()
print("  Higher rho values test progressively stronger longitudinal continuity.")
print("=" * 112)
