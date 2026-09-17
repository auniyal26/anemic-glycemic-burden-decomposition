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

df["HB_THRESHOLD"] = np.where(df["RIAGENDR"] == 1, 13.0, 12.0)
df["A"] = (df["HB_THRESHOLD"] - df["LBXHGB"]).clip(lower=0)
df["G"] = (df["LBXGH"] - 5.7).clip(lower=0)
df["AG"] = df["A"] * df["G"]

df["AGE_Z"] = (
    df["RIDAGEYR"] - df["RIDAGEYR"].mean()
) / df["RIDAGEYR"].std()

X = pd.DataFrame({
    "AGE_Z": df["AGE_Z"],
    "SEX": df["RIAGENDR"].astype(float)
})

race = pd.get_dummies(
    df["RIDRETH1"].astype(int),
    prefix="RACE",
    drop_first=True,
    dtype=float
)

cycle = pd.get_dummies(
    df["CYCLE"],
    prefix="CYCLE",
    drop_first=True,
    dtype=float
)

X = pd.concat([X, race, cycle], axis=1)
X = sm.add_constant(X).astype(float)

residuals = pd.DataFrame(index=df.index)

for item in items:
    model = sm.OLS(df[item].astype(float), X).fit()
    residuals[item] = model.resid

residuals.to_parquet(
    OUT / "07_confounded_removed_phq.parquet",
    compression="zstd"
)

scaler = StandardScaler()
R = scaler.fit_transform(residuals)

loss = []

for k in range(1, 10):
    pca = PCA(n_components=k)
    Rp = pca.inverse_transform(pca.fit_transform(R))

    ica = FastICA(
        n_components=k,
        random_state=42,
        max_iter=5000,
        whiten="unit-variance"
    )

    Ri = ica.inverse_transform(ica.fit_transform(R))

    loss.append({
        "components": k,
        "pca_rmse": np.sqrt(mean_squared_error(R, Rp)),
        "pca_variance": pca.explained_variance_ratio_.sum(),
        "ica_rmse": np.sqrt(mean_squared_error(R, Ri))
    })

loss = pd.DataFrame(loss)
loss.to_csv(OUT / "07_latent_loss.csv", index=False)

ica = FastICA(
    n_components=4,
    random_state=42,
    max_iter=5000,
    whiten="unit-variance"
)

Z = ica.fit_transform(R)

components = pd.DataFrame(
    Z,
    columns=["IC1", "IC2", "IC3", "IC4"],
    index=df.index
)

loadings = pd.DataFrame(
    ica.mixing_,
    index=items,
    columns=["IC1", "IC2", "IC3", "IC4"]
)

loadings.to_csv(
    OUT / "07_ica_loadings.csv"
)

assoc = []

for cycle_name in ["ALL", "0506", "0708"]:
    if cycle_name == "ALL":
        mask = np.ones(len(df), dtype=bool)
    else:
        mask = df["CYCLE"].eq(cycle_name).values

    for ic in components.columns:
        temp = pd.DataFrame({
            "Y": components.loc[mask, ic].values,
            "A": df.loc[mask, "A"].values,
            "G": df.loc[mask, "G"].values,
            "AG": df.loc[mask, "AG"].values
        })

        Xm = sm.add_constant(temp[["A", "G", "AG"]])

        model = sm.OLS(temp["Y"], Xm).fit()

        assoc.append({
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

assoc = pd.DataFrame(assoc)

assoc.to_csv(
    OUT / "07_latent_associations.csv",
    index=False
)

output = pd.concat(
    [
        df[["SEQN", "CYCLE", "A", "G", "AG"]].reset_index(drop=True),
        components.reset_index(drop=True)
    ],
    axis=1
)

output.to_parquet(
    OUT / "07_latent_components.parquet",
    compression="zstd"
)

print("\nICA LOADINGS")
print(loadings.to_string())

print("\nLATENT ASSOCIATIONS")
print(assoc.to_string(index=False))

print("\nCOMPRESSION LOSS")
print(loss.to_string(index=False))