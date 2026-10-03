#!/usr/bin/env python
# =============================================================================
# 85_delta_state_informed_bridge_simulation.py
#
# Goal:
#   Test whether CHANGE in phenotype adds information about CHANGE in physiology
#   beyond knowing the person's previous physiological state.
#
# Target:
#       ΔZ_t = Z_t - Z_{t-1}
#
# Compare:
#       1) ΔY only
#       2) Z_{t-1} only
#       3) Z_{t-1} + ΔY
#
# Simulation is grounded in empirical NHANES 2005-08 Z/Y structure.
# NHANES itself is NOT person-level longitudinal data, so these are controlled
# simulation results, not longitudinal evidence.
#
# We vary:
#   rho   = physiological continuity across time
#   kappa = persistence of person-specific phenotype residual/noise
#
# kappa is useful because within-person differencing can cancel stable phenotype
# nuisance. If ΔY becomes more informative as kappa rises, that supports the
# longitudinal-change formulation.
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
KAPPAS = [0.00, 0.50, 0.90]

N_SIM = 30000
SEED = 85


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


def mse_multi(Y, P):
    return float(np.mean((np.asarray(Y) - np.asarray(P)) ** 2))


def r2_multi(Y, P):
    Y = np.asarray(Y, float)
    P = np.asarray(P, float)
    mu = Y.mean(axis=0)
    sse = np.sum((Y - P) ** 2)
    sst = np.sum((Y - mu) ** 2)
    return float(1 - sse / sst) if sst > 0 else np.nan


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
    p = cov.shape[0]
    avg_sd = float(np.mean(np.sqrt(np.diag(cov))))
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise RuntimeError("Residual covariance is not positive definite.")
    logvol = float(0.5 * logdet + 0.5 * p * np.log(cutoff))
    return avg_sd, logvol


# -----------------------------------------------------------------------------
# 1. Load empirical 2005-08 discovery state.
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
# 2. Empirical forward phenotype model Y|Z and residual bank.
# -----------------------------------------------------------------------------
forward, alpha_forward = tune_ridge(Zs, Ys, seed=851)
Yhat = forward.predict(Zs)
Yresid = Ys - Yhat
Yresid = Yresid - Yresid.mean(axis=0, keepdims=True)

prob = w / w.sum()
rng = np.random.default_rng(SEED)

# Fixed ingredients across all scenarios for paired comparisons.
idx_prev = rng.choice(len(d), size=N_SIM, replace=True, p=prob)
idx_innov = rng.choice(len(d), size=N_SIM, replace=True, p=prob)
idx_eps_prev = rng.choice(len(d), size=N_SIM, replace=True, p=prob)
idx_eps_new = rng.choice(len(d), size=N_SIM, replace=True, p=prob)

Zprev_base = Zs[idx_prev]
Zinnov_base = Zs[idx_innov]
eps_prev_base = Yresid[idx_eps_prev]
eps_new_base = Yresid[idx_eps_new]

perm = rng.permutation(N_SIM)
n_train = int(0.60 * N_SIM)
n_cal = int(0.20 * N_SIM)

itr = perm[:n_train]
ica = perm[n_train:n_train + n_cal]
ite = perm[n_train + n_cal:]

rows = []

# -----------------------------------------------------------------------------
# 3. Simulate trajectories and test ΔZ inference.
# -----------------------------------------------------------------------------
for rho in RHOS:
    Zprev = Zprev_base.copy()

    Zt = (
        rho * Zprev_base
        + np.sqrt(max(0.0, 1.0 - rho**2)) * Zinnov_base
    )
    dZ = Zt - Zprev

    for kappa in KAPPAS:
        eps_prev = eps_prev_base.copy()
        eps_t = (
            kappa * eps_prev_base
            + np.sqrt(max(0.0, 1.0 - kappa**2)) * eps_new_base
        )

        Yprev = forward.predict(Zprev) + eps_prev
        Yt = forward.predict(Zt) + eps_t
        dY = Yt - Yprev

        designs = {
            "deltaY_only": dY,
            "Zprev_only": Zprev,
            "Zprev_plus_deltaY": np.column_stack([Zprev, dY]),
        }

        scenario = {}

        for model_name, X in designs.items():
            model, alpha = tune_ridge(X[itr], dZ[itr], seed=852)

            # Independent conformal calibration.
            dZ_cal_hat = model.predict(X[ica])
            Ecal = dZ[ica] - dZ_cal_hat
            emu, ecov = residual_cov(Ecal)
            score_cal = mahal2(Ecal, emu, ecov)
            cutoff95 = conformal_quantile(score_cal, 0.95)

            # Untouched simulated test set.
            dZ_test_hat = model.predict(X[ite])
            Etest = dZ[ite] - dZ_test_hat
            score_test = mahal2(Etest, emu, ecov)

            cov95 = float(np.mean(score_test <= cutoff95))
            dZ_r2 = r2_multi(dZ[ite], dZ_test_hat)

            # Since Zprev is known, reconstruct Zt and audit current-state R2.
            Zt_hat = Zprev[ite] + dZ_test_hat
            Zt_r2 = r2_multi(Zt[ite], Zt_hat)

            avg_sd, logvol = sharpness(ecov, cutoff95)

            res = {
                "rho": rho,
                "kappa": kappa,
                "model": model_name,
                "deltaZ_R2": dZ_r2,
                "Zt_reconstruction_R2": Zt_r2,
                "coverage95": cov95,
                "avg_residual_SD": avg_sd,
                "log_volume_proxy": logvol,
                "ridge_alpha": alpha,
                "conformal_cutoff95": cutoff95,
                "test_n": len(ite),
            }
            rows.append(res)
            scenario[model_name] = res

        base = scenario["Zprev_only"]
        informed = scenario["Zprev_plus_deltaY"]

        # Add direct incremental-information fields to the informed row.
        rows[-1]["deltaZ_R2_gain_vs_Zprev"] = (
            informed["deltaZ_R2"] - base["deltaZ_R2"]
        )
        rows[-1]["Zt_R2_gain_vs_Zprev"] = (
            informed["Zt_reconstruction_R2"]
            - base["Zt_reconstruction_R2"]
        )
        rows[-1]["relative_volume_vs_Zprev"] = float(
            np.exp(
                informed["log_volume_proxy"]
                - base["log_volume_proxy"]
            )
        )

summary = pd.DataFrame(rows)
summary.to_csv(
    RESULTS / "85_delta_state_informed_bridge_simulation.csv",
    index=False,
)

manifest = {
    "script": "85_delta_state_informed_bridge_simulation.py",
    "status": "controlled simulation; NOT observed longitudinal evidence",
    "empirical_source": "NHANES 2005-08",
    "target": "delta Z",
    "predictors_compared": [
        "delta Y only",
        "Z previous only",
        "Z previous + delta Y",
    ],
    "rho_values": RHOS,
    "kappa_values": KAPPAS,
    "rho_definition": "physiological temporal continuity",
    "kappa_definition": "persistence of phenotype residual/nuisance across time",
    "n_simulated": N_SIM,
    "train_cal_test": "60/20/20",
    "uncertainty": "95% split-conformal Mahalanobis region",
    "causal": False,
}

(RESULTS / "85_delta_state_informed_bridge_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("=" * 116)
print("SCRIPT 85 — DELTA-STATE INFORMED REVERSE BRIDGE SIMULATION")
print("=" * 116)
print(f"Empirical discovery rows = {len(d)}")
print(f"Simulated trajectories = {N_SIM}")
print(f"Forward phenotype ridge alpha = {alpha_forward:g}")
print()
print("CONTROLLED SIMULATION ONLY — NHANES IS NOT PERSON-LEVEL LONGITUDINAL.")
print()

for rho in RHOS:
    print(f"[rho={rho:.2f}] physiological continuity")

    for kappa in KAPPAS:
        q = summary[
            (summary["rho"].eq(rho))
            & (summary["kappa"].eq(kappa))
        ]

        dy = q[q["model"].eq("deltaY_only")].iloc[0]
        zp = q[q["model"].eq("Zprev_only")].iloc[0]
        inf = q[q["model"].eq("Zprev_plus_deltaY")].iloc[0]

        gain = inf["deltaZ_R2"] - zp["deltaZ_R2"]
        volratio = np.exp(
            inf["log_volume_proxy"] - zp["log_volume_proxy"]
        )

        print(
            f"  kappa={kappa:.2f} | "
            f"ΔY R2={dy.deltaZ_R2:.4f} | "
            f"Zprev R2={zp.deltaZ_R2:.4f} | "
            f"Zprev+ΔY R2={inf.deltaZ_R2:.4f} | "
            f"gain={gain:+.4f} | "
            f"cov95={inf.coverage95:.3f} | "
            f"volume ratio={volratio:.3f}"
        )

    print()

print("KEY TEST")
print("  gain > 0 and volume ratio < 1 with ~95% coverage means phenotype CHANGE")
print("  contributes information about physiological CHANGE beyond prior physiology.")
print()
print("  If gains increase with kappa, within-person differencing is successfully")
print("  cancelling stable phenotype nuisance — exactly the longitudinal advantage")
print("  this experiment is designed to test.")
print("=" * 116)
