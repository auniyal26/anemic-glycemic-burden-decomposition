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
    "SEQN", "CYCLE", "LBXHGB", "LBXGH",
    "RIDAGEYR", "RIAGENDR", "RIDRETH1"
] + items

df = df[cols].dropna().copy()

df["HB_THRESHOLD"] = np.where(df["RIAGENDR"] == 1, 13.0, 12.0)
df["A"] = (df["HB_THRESHOLD"] - df["LBXHGB"]).clip(lower=0)
df["G"] = (df["LBXGH"] - 5.7).clip(lower=0)
df["AG"] = df["A"] * df["G"]

def fit_cycle(d, cycle_name):
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
        model = sm.OLS(d[item].astype(float), X).fit()
        residuals[item] = model.resid

    scaler = StandardScaler()
    R = scaler.fit_transform(residuals)

    ica = FastICA(
        n_components=4,
        random_state=42,
        max_iter=10000,
        tol=1e-5,
        whiten="unit-variance"
    )

    Z = ica.fit_transform(R)

    loadings = pd.DataFrame(
        ica.mixing_,
        index=items,
        columns=[f"IC{i+1}" for i in range(4)]
    )

    components = pd.DataFrame(
        Z,
        index=d.index,
        columns=[f"IC{i+1}" for i in range(4)]
    )

    associations = []

    for ic in components.columns:
        temp = pd.DataFrame({
            "Y": components[ic],
            "A": d["A"],
            "G": d["G"],
            "AG": d["AG"]
        }).dropna()

        Xm = sm.add_constant(temp[["A", "G", "AG"]])

        model = sm.OLS(temp["Y"], Xm).fit()

        associations.append({
            "cycle": cycle_name,
            "component": ic,
            "n": len(temp),
            "A_beta": model.params["A"],
            "A_p": model.pvalues["A"],
            "G_beta": model.params["G"],
            "G_p": model.pvalues["G"],
            "AG_beta": model.params["AG"],
            "AG_p": model.pvalues["AG"],
            "r2": model.rsquared
        })

    return loadings, components, pd.DataFrame(associations)

l05, z05, a05 = fit_cycle(
    df[df["CYCLE"] == "0506"],
    "0506"
)

l07, z07, a07 = fit_cycle(
    df[df["CYCLE"] == "0708"],
    "0708"
)

def normalize_columns(x):
    arr = x.values.astype(float)
    return arr / np.linalg.norm(arr, axis=0, keepdims=True)

n05 = normalize_columns(l05)
n07 = normalize_columns(l07)

similarity = np.abs(n05.T @ n07)

rows, cols_match = linear_sum_assignment(-similarity)

matches = []
aligned07 = pd.DataFrame(index=items)

for i, j in zip(rows, cols_match):
    v05 = n05[:, i]
    v07 = n07[:, j]

    signed_similarity = np.dot(v05, v07)
    sign = 1 if signed_similarity >= 0 else -1

    aligned07[f"IC{i+1}"] = l07.iloc[:, j] * sign

    matches.append({
        "0506_component": f"IC{i+1}",
        "0708_component": f"IC{j+1}",
        "absolute_similarity": similarity[i, j],
        "sign_flip": sign == -1,
        "signed_similarity_after_alignment": abs(signed_similarity)
    })

matches = pd.DataFrame(matches)

aligned05 = l05.copy()
aligned05.columns = [f"IC{i+1}" for i in range(4)]

comparison = []

for ic in aligned05.columns:
    for item in items:
        comparison.append({
            "component": ic,
            "item": item,
            "loading_0506": aligned05.loc[item, ic],
            "loading_0708": aligned07.loc[item, ic]
        })

comparison = pd.DataFrame(comparison)

associations = pd.concat(
    [a05, a07],
    ignore_index=True
)

l05.to_csv(OUT / "08_loadings_0506.csv")
l07.to_csv(OUT / "08_loadings_0708_raw.csv")
aligned07.to_csv(OUT / "08_loadings_0708_aligned.csv")

matches.to_csv(
    OUT / "08_component_matching.csv",
    index=False
)

comparison.to_csv(
    OUT / "08_loading_replication.csv",
    index=False
)

associations.to_csv(
    OUT / "08_cycle_latent_associations.csv",
    index=False
)

print("\n0506 LOADINGS")
print(aligned05.to_string())

print("\n0708 ALIGNED LOADINGS")
print(aligned07.to_string())

print("\nCOMPONENT MATCHING")
print(matches.to_string(index=False))

print("\nINDEPENDENT CYCLE ASSOCIATIONS")
print(associations.to_string(index=False))