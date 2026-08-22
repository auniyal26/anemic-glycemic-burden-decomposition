from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "Processed"
OUT = ROOT / "Results"

df = pd.read_parquet(DATA / "nhanes_core_working.parquet")

items = [f"DPQ0{i}0" for i in range(1, 10)]

cols = [
    "SEQN", "CYCLE",
    "RIDAGEYR", "RIAGENDR", "RIDRETH1"
] + items

df = df[cols].dropna().copy()

def residual_matrix(d):
    d = d.copy()

    d["AGE_Z"] = (
        d["RIDAGEYR"] - d["RIDAGEYR"].mean()
    ) / d["RIDAGEYR"].std()

    X = pd.DataFrame({
        "AGE_Z": d["AGE_Z"],
        "SEX": d["RIAGENDR"].astype(float)
    }, index=d.index)

    race = pd.get_dummies(
        d["RIDRETH1"].astype(int),
        prefix="RACE",
        drop_first=True,
        dtype=float
    )

    X = pd.concat([X, race], axis=1)
    X = sm.add_constant(X).astype(float)

    R = pd.DataFrame(index=d.index)

    for item in items:
        model = sm.OLS(d[item].astype(float), X).fit()
        R[item] = model.resid

    return StandardScaler().fit_transform(R)

def fit_ica(R, seed):
    model = FastICA(
        n_components=4,
        random_state=seed,
        max_iter=10000,
        tol=1e-5,
        whiten="unit-variance"
    )

    model.fit_transform(R)

    L = model.mixing_

    return L / np.linalg.norm(
        L,
        axis=0,
        keepdims=True
    )

def align(reference, candidate):
    S = np.abs(reference.T @ candidate)

    rows, cols = linear_sum_assignment(-S)

    aligned = np.zeros_like(candidate)
    sims = []

    for i, j in zip(rows, cols):
        sign = np.sign(
            np.dot(reference[:, i], candidate[:, j])
        )

        if sign == 0:
            sign = 1

        aligned[:, i] = candidate[:, j] * sign

        sims.append(
            abs(np.dot(reference[:, i], aligned[:, i]))
        )

    return aligned, sims

results = []

for cycle in ["0506", "0708"]:

    R = residual_matrix(
        df[df["CYCLE"] == cycle]
    )

    reference = fit_ica(R, 42)

    for seed in range(100):

        candidate = fit_ica(R, seed)

        aligned, sims = align(
            reference,
            candidate
        )

        for component, similarity in enumerate(
            sims,
            start=1
        ):

            results.append({
                "cycle": cycle,
                "seed": seed,
                "component": f"IC{component}",
                "similarity": similarity
            })

results = pd.DataFrame(results)

summary = (
    results
    .groupby(["cycle", "component"])
    ["similarity"]
    .agg(["mean", "std", "min", "median"])
    .reset_index()
)

results.to_csv(
    OUT / "09_ica_seed_stability_raw.csv",
    index=False
)

summary.to_csv(
    OUT / "09_ica_seed_stability_summary.csv",
    index=False
)

print("\nICA STABILITY")
print(summary.to_string(index=False))