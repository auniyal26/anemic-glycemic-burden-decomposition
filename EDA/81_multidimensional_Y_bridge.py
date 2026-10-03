#!/usr/bin/env python
# =============================================================================
# 81_multidimensional_Y_bridge.py
#
# Goal:
#   Test whether richer PHQ phenotype information makes Y -> Z meaningfully
#   recoverable.
#
# Compare:
#   Y1 = PHQ-9 total
#   Y2 = [somatic, cognitive-affective]
#   Y9 = all 9 PHQ items
#
# Train ONLY on 2005-08; freeze into 2009-18 and 2021-23.
# Uses the same raw-DPQ restoration logic as Script 58.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import json
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
    raise FileNotFoundError("Run Script 73 first.")

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
    "0910": "F", "1112": "G", "1314": "H",
    "1516": "I", "1718": "J", "2123": "L",
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
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def stdize(X, mu, sd):
    return (np.asarray(X, float) - mu) / sd


def as_2d(X):
    X = np.asarray(X, float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def wmse(Y, P, w):
    Y = as_2d(Y)
    P = as_2d(P)
    return float(
        np.sum(w[:, None] * (Y - P) ** 2)
        / (np.sum(w) * Y.shape[1])
    )


def wr2(Y, P, w):
    Y = as_2d(Y)
    P = as_2d(P)
    mu = np.sum(w[:, None] * Y, axis=0) / np.sum(w)
    sse = np.sum(w[:, None] * (Y - P) ** 2)
    sst = np.sum(w[:, None] * (Y - mu) ** 2)
    return float(1 - sse / sst) if sst > 0 else np.nan


def tune_ridge(A, B, w, seed):
    grid = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scored = []

    for alpha in grid:
        errs = []
        for tr, va in kf.split(A):
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(A[tr], B[tr], sample_weight=w[tr])
            errs.append(wmse(B[va], m.predict(A[va]), w[va]))
        scored.append((float(np.mean(errs)), alpha))

    scored.sort(key=lambda x: x[0])
    alpha = scored[0][1]

    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(A, B, sample_weight=w)
    return m, alpha


# -------------------------------------------------------------------------
# 1. Frozen Z scaffold
# -------------------------------------------------------------------------
d = pd.read_csv(INPUT)

required = ["SEQN", "PERIOD", "SURVEY_WT", "ELIGIBLE_ADULT_NONPREG"] + ZCOLS
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Script-73 input missing: {missing}")

d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce")

# Remove any pre-existing PHQ item columns so raw DPQ is authoritative.
drop_existing = [c for c in PHQ if c in d.columns]
if drop_existing:
    d = d.drop(columns=drop_existing)

# -------------------------------------------------------------------------
# 2. Restore all nine PHQ items exactly from raw DPQ XPTs (Script-58 logic)
# -------------------------------------------------------------------------
frames = []

for cycle in ALL_CYCLES:
    suffix = suffix_for(cycle)
    p = xpt_path(base_for(cycle), f"DPQ_{suffix}")
    q = pd.read_sas(p, format="xport")

    have = [c for c in PHQ if c in q.columns]
    if len(have) != 9:
        raise ValueError(f"{cycle}: only found PHQ columns {have}")

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
    dup = phq.loc[phq["SEQN"].duplicated(), "SEQN"].head().tolist()
    raise RuntimeError(f"Duplicate SEQN found in raw DPQ bank: {dup}")

d = d.merge(phq, on="SEQN", how="left", validate="one_to_one")

d["PHQ9_TOTAL_BRIDGE"] = d[PHQ].sum(axis=1, min_count=9)
d["SOMATIC_BRIDGE"] = d[SOMATIC].sum(axis=1, min_count=3)
d["COGAFF_BRIDGE"] = d["PHQ9_TOTAL_BRIDGE"] - d["SOMATIC_BRIDGE"]

print("=" * 108)
print("SCRIPT 81 — MULTIDIMENSIONAL Y BRIDGE")
print("=" * 108)
print("PHQ RESTORE AUDIT")
for period in PERIODS:
    q = d[d["PERIOD"].eq(period)]
    print(
        f"  {period}: rows={len(q)}, "
        f"all9 PHQ={int(q[PHQ].notna().all(axis=1).sum())}, "
        f"all7 Z={int(q[ZCOLS].notna().all(axis=1).sum())}, "
        f"both={int(q[PHQ + ZCOLS].notna().all(axis=1).sum())}"
    )
print()

# -------------------------------------------------------------------------
# 3. Exact common cohort: all 9 PHQ + all 7 Z
# -------------------------------------------------------------------------
d = d[
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[PHQ + ZCOLS + ["SURVEY_WT"]].notna().all(axis=1)
    & (pd.to_numeric(d["SURVEY_WT"], errors="coerce") > 0)
].copy()

REPRESENTATIONS = {
    "Y1_total": ["PHQ9_TOTAL_BRIDGE"],
    "Y2_somatic_cogaff": ["SOMATIC_BRIDGE", "COGAFF_BRIDGE"],
    "Y9_items": PHQ,
}

disc_n = len(d[d["PERIOD"].eq("2005-2008")])
if disc_n < 100:
    raise RuntimeError(
        f"Still too few discovery rows after verified PHQ restoration: n={disc_n}"
    )

# -------------------------------------------------------------------------
# 4. Discovery-only bridge; frozen temporal transfer
# -------------------------------------------------------------------------
rows = []
model_rows = []

for rep, ycols in REPRESENTATIONS.items():
    disc = d[d["PERIOD"].eq("2005-2008")].copy()

    Z0 = disc[ZCOLS].to_numpy(float)
    Y0 = disc[ycols].to_numpy(float)
    w0 = disc["SURVEY_WT"].to_numpy(float)

    zmu, zsd = weighted_mean_sd(Z0, w0)
    ymu, ysd = weighted_mean_sd(Y0, w0)

    Z0s = stdize(Z0, zmu, zsd)
    Y0s = stdize(Y0, ymu, ysd)

    f, af = tune_ridge(Z0s, Y0s, w0, 42)  # Z -> Y
    g, ag = tune_ridge(Y0s, Z0s, w0, 43)  # Y -> Z

    model_rows.append({
        "representation": rep,
        "Y_dimension": len(ycols),
        "Z_dimension": len(ZCOLS),
        "rank_Z_to_Y": int(np.linalg.matrix_rank(np.atleast_2d(f.coef_))),
        "rank_Y_to_Z": int(np.linalg.matrix_rank(np.atleast_2d(g.coef_))),
        "ridge_alpha_Z_to_Y": af,
        "ridge_alpha_Y_to_Z": ag,
        "discovery_n": len(disc),
    })

    for period in PERIODS:
        q = d[d["PERIOD"].eq(period)].copy()
        if len(q) < 50:
            continue

        Z = q[ZCOLS].to_numpy(float)
        Y = q[ycols].to_numpy(float)
        w = q["SURVEY_WT"].to_numpy(float)

        Zs = stdize(Z, zmu, zsd)
        Ys = stdize(Y, ymu, ysd)

        Yhat = as_2d(f.predict(Zs))
        Zhat = as_2d(g.predict(Ys))

        Zround = as_2d(g.predict(Yhat))
        Yround = as_2d(f.predict(Zhat))

        rows.append({
            "representation": rep,
            "period": period,
            "n": len(q),
            "Z_to_Y_R2": wr2(Ys, Yhat, w),
            "Y_to_Z_R2": wr2(Zs, Zhat, w),
            "Z_to_Y_to_Z_R2": wr2(Zs, Zround, w),
            "Y_to_Z_to_Y_R2": wr2(Ys, Yround, w),
        })

summary = pd.DataFrame(rows)
models = pd.DataFrame(model_rows)

summary.to_csv(RESULTS / "81_multidimensional_Y_bridge_summary.csv", index=False)
models.to_csv(RESULTS / "81_multidimensional_Y_bridge_models.csv", index=False)

manifest = {
    "script": "81_multidimensional_Y_bridge.py",
    "training": "2005-2008 only",
    "future_refit": False,
    "Z": ZCOLS,
    "Y_representations": REPRESENTATIONS,
    "common_complete_PHQ9_Z_cohort": True,
    "raw_PHQ_source": "authoritative NHANES DPQ XPT files; Script-58 restoration logic",
    "causal": False,
}
(RESULTS / "81_multidimensional_Y_bridge_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print(f"COMMON DISCOVERY COHORT n={disc_n}")
print()

for rep in ["Y1_total", "Y2_somatic_cogaff", "Y9_items"]:
    print(f"[{rep}]")
    m = models[models["representation"].eq(rep)].iloc[0]
    print(
        f"  dimensions Y={int(m.Y_dimension)}, Z={int(m.Z_dimension)} | "
        f"rank Y->Z={int(m.rank_Y_to_Z)}"
    )

    q = summary[summary["representation"].eq(rep)]
    for period in PERIODS:
        z = q[q["period"].eq(period)]
        if z.empty:
            continue
        r = z.iloc[0]
        print(
            f"  {period}: "
            f"Z->Y={r.Z_to_Y_R2:.4f} | "
            f"Y->Z={r.Y_to_Z_R2:.4f} | "
            f"Z->Y->Z={r.Z_to_Y_to_Z_R2:.4f} | "
            f"Y->Z->Y={r.Y_to_Z_to_Y_R2:.4f}"
        )
    print()

print("KEY QUESTION")
print("  Does Y9 materially improve Y->Z over Y1/Y2?")
print("  If no, the reverse bridge is not a deterministic inverse and should")
print("  become a conditional/distributional map p(Z|Y), not forced reconstruction.")
print("=" * 108)
