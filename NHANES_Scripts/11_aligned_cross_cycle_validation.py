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

K = 3
SEED = 42

df = pd.read_parquet(DATA / "nhanes_core_working.parquet")

items = [f"DPQ0{i}0" for i in range(1, 10)]

cols = [
    "SEQN", "CYCLE", "LBXHGB", "LBXGH",
    "RIDAGEYR", "RIAGENDR", "RIDRETH1"
] + items

df = df[cols].dropna().copy()

df["HB_THRESHOLD"] = np.where(
    df["RIAGENDR"] == 1, 13.0, 12.0
)

df["A"] = (
    df["HB_THRESHOLD"] - df["LBXHGB"]
).clip(lower=0)

df["G"] = (
    df["LBXGH"] - 5.7
).clip(lower=0)

df["AG"] = df["A"] * df["G"]

def fit_reference(d):
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

    residuals = pd.DataFrame(index=d.index)

    for item in items:
        model = sm.OLS(
            d[item].astype(float),
            X
        ).fit()

        residuals[item] = model.resid

    R = StandardScaler().fit_transform(residuals)

    ica = FastICA(
        n_components=K,
        random_state=SEED,
        max_iter=10000,
        tol=1e-5,
        whiten="unit-variance"
    )

    Z = ica.fit_transform(R)
    L = ica.mixing_

    return Z, L

def normalize(L):
    return L / np.linalg.norm(
        L,
        axis=0,
        keepdims=True
    )

d05 = df[df["CYCLE"] == "0506"].copy()
d07 = df[df["CYCLE"] == "0708"].copy()

z05, l05 = fit_reference(d05)
z07, l07 = fit_reference(d07)

n05 = normalize(l05)
n07 = normalize(l07)

similarity = np.abs(n05.T @ n07)

rows, cols_match = linear_sum_assignment(
    -similarity
)

mapping = []

for i, j in zip(rows, cols_match):

    signed = np.dot(
        n05[:, i],
        n07[:, j]
    )

    sign = 1 if signed >= 0 else -1

    mapping.append({
        "aligned_component": f"IC{i+1}",
        "component_0506": f"IC{i+1}",
        "component_0708": f"IC{j+1}",
        "similarity": abs(signed),
        "sign_0708": sign
    })

mapping = pd.DataFrame(mapping)

boot = pd.read_csv(
    OUT / "10_bootstrap_betas_raw.csv"
)

aligned_rows = []

for _, m in mapping.iterrows():

    c05 = m["component_0506"]
    c07 = m["component_0708"]
    aligned = m["aligned_component"]
    sign07 = m["sign_0708"]

    b05 = boot[
        (boot["cycle"] == 506)
        & (boot["component"] == c05)
    ].copy()

    if len(b05) == 0:
        b05 = boot[
            (boot["cycle"].astype(str) == "0506")
            & (boot["component"] == c05)
        ].copy()

    b07 = boot[
        (boot["cycle"] == 708)
        & (boot["component"] == c07)
    ].copy()

    if len(b07) == 0:
        b07 = boot[
            (boot["cycle"].astype(str) == "0708")
            & (boot["component"] == c07)
        ].copy()

    b05["aligned_component"] = aligned
    b07["aligned_component"] = aligned

    for v in ["A", "G", "AG"]:
        b07[v] = b07[v] * sign07

    aligned_rows.append(b05)
    aligned_rows.append(b07)

aligned_boot = pd.concat(
    aligned_rows,
    ignore_index=True
)

aligned_boot["cycle"] = (
    aligned_boot["cycle"]
    .astype(str)
    .str.zfill(4)
)

summary_rows = []

for component in [
    "IC1", "IC2", "IC3"
]:

    for variable in [
        "A", "G", "AG"
    ]:

        temp05 = aligned_boot[
            (aligned_boot["aligned_component"] == component)
            & (aligned_boot["cycle"] == "0506")
        ][variable]

        temp07 = aligned_boot[
            (aligned_boot["aligned_component"] == component)
            & (aligned_boot["cycle"] == "0708")
        ][variable]

        summary_rows.append({
            "component": component,
            "variable": variable,

            "mean_0506": temp05.mean(),
            "ci05_low": temp05.quantile(0.025),
            "ci05_high": temp05.quantile(0.975),

            "mean_0708": temp07.mean(),
            "ci07_low": temp07.quantile(0.025),
            "ci07_high": temp07.quantile(0.975),

            "same_direction": (
                np.sign(temp05.mean())
                == np.sign(temp07.mean())
            ),

            "0506_nonzero": (
                temp05.quantile(0.025) > 0
                or temp05.quantile(0.975) < 0
            ),

            "0708_nonzero": (
                temp07.quantile(0.025) > 0
                or temp07.quantile(0.975) < 0
            )
        })

summary = pd.DataFrame(
    summary_rows
)

mapping.to_csv(
    OUT / "11_cross_cycle_component_mapping.csv",
    index=False
)

aligned_boot.to_csv(
    OUT / "11_aligned_bootstrap_betas.csv",
    index=False
)

summary.to_csv(
    OUT / "11_aligned_physiological_validation.csv",
    index=False
)

print("\n3-COMPONENT CROSS-CYCLE ALIGNMENT")
print(mapping.to_string(index=False))

print("\nALIGNED PHYSIOLOGICAL VALIDATION")
print(summary.to_string(index=False))