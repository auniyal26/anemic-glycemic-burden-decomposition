from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

df["HB_THRESHOLD"] = np.where(
    df["RIAGENDR"] == 1, 13.0, 12.0
)

df["A"] = (
    df["HB_THRESHOLD"] - df["LBXHGB"]
).clip(lower=0)

df["G"] = (
    df["LBXGH"] - 5.7
).clip(lower=0)

df["AGE_Z"] = (
    df["RIDAGEYR"] - df["RIDAGEYR"].mean()
) / df["RIDAGEYR"].std()

original = df[["SEQN", "CYCLE"] + items].copy()

original.to_parquet(
    OUT / "06_original_phq_reference.parquet",
    compression="zstd"
)

base = pd.DataFrame({
    "AGE_Z": df["AGE_Z"],
    "SEX": df["RIAGENDR"].astype(float),
    "A": df["A"],
    "G": df["G"]
})

race = pd.get_dummies(
    df["RIDRETH1"].astype(int),
    prefix="RACE",
    drop_first=True,
    dtype=float
)

X = pd.concat([base, race], axis=1)
X = sm.add_constant(X).astype(float)

residuals = pd.DataFrame(index=df.index)
predictions = pd.DataFrame(index=df.index)

for item in items:
    model = sm.OLS(
        df[item].astype(float),
        X
    ).fit()

    predictions[item] = model.predict(X)
    residuals[item] = (
        df[item] - predictions[item]
    )

residuals.to_parquet(
    OUT / "06_phq_residual_matrix.parquet",
    compression="zstd"
)

scaler = StandardScaler()
R = scaler.fit_transform(residuals)

results = []

for k in range(1, 10):

    pca = PCA(n_components=k)
    Zp = pca.fit_transform(R)
    Rp = pca.inverse_transform(Zp)

    pca_rmse = np.sqrt(
        mean_squared_error(R, Rp)
    )

    ica = FastICA(
        n_components=k,
        random_state=42,
        max_iter=3000,
        whiten="unit-variance"
    )

    Zi = ica.fit_transform(R)
    Ri = ica.inverse_transform(Zi)

    ica_rmse = np.sqrt(
        mean_squared_error(R, Ri)
    )

    results.append({
        "components": k,
        "pca_rmse": pca_rmse,
        "pca_variance": pca.explained_variance_ratio_.sum(),
        "ica_rmse": ica_rmse
    })

results = pd.DataFrame(results)

results.to_csv(
    OUT / "06_decomposition_loss.csv",
    index=False
)

ica = FastICA(
    n_components=9,
    random_state=42,
    max_iter=3000,
    whiten="unit-variance"
)

components = ica.fit_transform(R)

component_df = pd.DataFrame(
    components,
    columns=[f"IC{i+1}" for i in range(9)]
)

component_df["SEQN"] = df["SEQN"].values
component_df["CYCLE"] = df["CYCLE"].values
component_df["A"] = df["A"].values
component_df["G"] = df["G"].values

component_df.to_parquet(
    OUT / "06_ica_components.parquet",
    compression="zstd"
)

associations = []

for ic in [f"IC{i+1}" for i in range(9)]:

    Y = component_df[ic]

    Xa = sm.add_constant(
        component_df[["A", "G"]]
    )

    model = sm.OLS(Y, Xa).fit()

    associations.append({
        "component": ic,
        "A_beta": model.params["A"],
        "A_p": model.pvalues["A"],
        "G_beta": model.params["G"],
        "G_p": model.pvalues["G"],
        "r2": model.rsquared
    })

associations = pd.DataFrame(associations)

associations.to_csv(
    OUT / "06_component_associations.csv",
    index=False
)

print("\nDECOMPOSITION LOSS")
print(results.to_string(index=False))

print("\nICA COMPONENT ASSOCIATIONS")
print(associations.to_string(index=False))