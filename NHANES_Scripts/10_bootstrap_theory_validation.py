from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "Processed"
OUT = ROOT / "Results"

N_BOOT = 200
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

def residualize(d):
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
        model = sm.OLS(
            d[item].astype(float),
            X
        ).fit()

        R[item] = model.resid

    return StandardScaler().fit_transform(R)

def run_ica(R, seed):
    model = FastICA(
        n_components=K,
        random_state=seed,
        max_iter=10000,
        tol=1e-5,
        whiten="unit-variance"
    )

    Z = model.fit_transform(R)
    L = model.mixing_

    return Z, L

def normalized(L):
    return L / np.linalg.norm(
        L,
        axis=0,
        keepdims=True
    )

def align(reference_L, candidate_L, candidate_Z):
    ref = normalized(reference_L)
    cand = normalized(candidate_L)

    S = np.abs(ref.T @ cand)

    rows, cols = linear_sum_assignment(-S)

    aligned_Z = np.zeros_like(candidate_Z)
    similarities = np.zeros(K)

    for i, j in zip(rows, cols):
        sign = np.sign(
            np.dot(ref[:, i], cand[:, j])
        )

        if sign == 0:
            sign = 1

        aligned_Z[:, i] = (
            candidate_Z[:, j] * sign
        )

        similarities[i] = abs(
            np.dot(ref[:, i], cand[:, j])
        )

    return aligned_Z, similarities

def physiological_betas(Z, d):
    rows = []

    for i in range(K):
        X = sm.add_constant(
            d[["A", "G", "AG"]].astype(float)
        )

        model = sm.OLS(
            Z[:, i],
            X
        ).fit()

        rows.append({
            "component": f"IC{i+1}",
            "A": model.params["A"],
            "G": model.params["G"],
            "AG": model.params["AG"]
        })

    return pd.DataFrame(rows)

structural_rows = []
beta_rows = []
reference_rows = []

rng = np.random.default_rng(SEED)

for cycle in ["0506", "0708"]:

    original = (
        df[df["CYCLE"] == cycle]
        .copy()
        .reset_index(drop=True)
    )

    R_ref = residualize(original)

    Z_ref, L_ref = run_ica(
        R_ref,
        SEED
    )

    reference_betas = physiological_betas(
        Z_ref,
        original
    )

    reference_betas["cycle"] = cycle

    reference_rows.append(
        reference_betas
    )

    for b in tqdm(
        range(N_BOOT),
        desc=f"Bootstrap {cycle}"
    ):

        idx = rng.choice(
            len(original),
            size=len(original),
            replace=True
        )

        boot = (
            original
            .iloc[idx]
            .copy()
            .reset_index(drop=True)
        )

        R_boot = residualize(boot)

        try:
            Z_boot, L_boot = run_ica(
                R_boot,
                b
            )
        except Exception:
            continue

        Z_aligned, similarities = align(
            L_ref,
            L_boot,
            Z_boot
        )

        for i, similarity in enumerate(
            similarities
        ):
            structural_rows.append({
                "cycle": cycle,
                "bootstrap": b,
                "component": f"IC{i+1}",
                "similarity": similarity
            })

        betas = physiological_betas(
            Z_aligned,
            boot
        )

        for _, row in betas.iterrows():
            beta_rows.append({
                "cycle": cycle,
                "bootstrap": b,
                "component": row["component"],
                "A": row["A"],
                "G": row["G"],
                "AG": row["AG"]
            })

structural = pd.DataFrame(
    structural_rows
)

betas = pd.DataFrame(
    beta_rows
)

reference = pd.concat(
    reference_rows,
    ignore_index=True
)

structural_summary = (
    structural
    .groupby(["cycle", "component"])
    ["similarity"]
    .agg(
        mean="mean",
        std="std",
        minimum="min",
        p05=lambda x: x.quantile(0.05),
        median="median"
    )
    .reset_index()
)

association_summary = []

for _, ref in reference.iterrows():

    cycle = ref["cycle"]
    component = ref["component"]

    subset = betas[
        (betas["cycle"] == cycle)
        & (betas["component"] == component)
    ]

    for variable in ["A", "G", "AG"]:

        vals = subset[variable]

        reference_beta = ref[variable]

        association_summary.append({
            "cycle": cycle,
            "component": component,
            "variable": variable,
            "reference_beta": reference_beta,
            "bootstrap_mean": vals.mean(),
            "ci_025": vals.quantile(0.025),
            "ci_975": vals.quantile(0.975),
            "same_sign_fraction": (
                np.sign(vals)
                == np.sign(reference_beta)
            ).mean()
        })

association_summary = pd.DataFrame(
    association_summary
)

structural.to_csv(
    OUT / "10_bootstrap_structural_raw.csv",
    index=False
)

structural_summary.to_csv(
    OUT / "10_bootstrap_structural_summary.csv",
    index=False
)

betas.to_csv(
    OUT / "10_bootstrap_betas_raw.csv",
    index=False
)

association_summary.to_csv(
    OUT / "10_bootstrap_association_summary.csv",
    index=False
)

print("\nSTRUCTURAL BOOTSTRAP STABILITY")
print(
    structural_summary.to_string(index=False)
)

print("\nPHYSIOLOGICAL ASSOCIATION STABILITY")
print(
    association_summary.to_string(index=False)
)