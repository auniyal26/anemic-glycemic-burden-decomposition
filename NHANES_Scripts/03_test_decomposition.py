from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "Processed"
OUT = ROOT / "Results"
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(DATA / "nhanes_core_working.parquet")

cols = [
    "SEQN", "CYCLE", "LBXHGB", "LBXGH", "PHQ9_TOTAL",
    "RIDAGEYR", "RIAGENDR", "RIDRETH1",
    "WTMEC2YR", "SDMVPSU", "SDMVSTRA"
]

df = df[cols].dropna().copy()

df["A"] = -(df["LBXHGB"] - df["LBXHGB"].mean()) / df["LBXHGB"].std()
df["G"] = (df["LBXGH"] - df["LBXGH"].mean()) / df["LBXGH"].std()
df["AG"] = df["A"] * df["G"]
df["AGE_Z"] = (df["RIDAGEYR"] - df["RIDAGEYR"].mean()) / df["RIDAGEYR"].std()

df["WEIGHT"] = df["WTMEC2YR"] / 2
df["PSU_ID"] = (
    df["CYCLE"].astype(str) + "_" +
    df["SDMVSTRA"].astype(int).astype(str) + "_" +
    df["SDMVPSU"].astype(int).astype(str)
)

reference = df[
    ["SEQN", "CYCLE", "LBXHGB", "LBXGH", "PHQ9_TOTAL"]
].copy()

reference.columns = [
    "SEQN", "CYCLE", "X_HGB", "X_HBA1C", "Y_PHQ9"
]

reference.to_parquet(
    OUT / "03_original_xy_reference.parquet",
    compression="zstd"
)

base = pd.DataFrame({
    "AGE_Z": df["AGE_Z"],
    "SEX": df["RIAGENDR"]
})

race = pd.get_dummies(
    df["RIDRETH1"].astype(int),
    prefix="RACE",
    drop_first=True,
    dtype=float
)

base = pd.concat([base, race], axis=1)
base["SEX"] = base["SEX"].astype(float)

models = {
    "M0_X": [],
    "M1_A": ["A"],
    "M2_G": ["G"],
    "M3_AG": ["A", "G"],
    "M4_INTERACTION": ["A", "G", "AG"]
}

results = []
fitted = {}

for name, physiological in models.items():
    X = base.copy()

    for variable in physiological:
        X[variable] = df[variable]

    X = sm.add_constant(X).astype(float)

    model = sm.WLS(
        df["PHQ9_TOTAL"],
        X,
        weights=df["WEIGHT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["PSU_ID"]}
    )

    fitted[name] = model

    results.append({
        "model": name,
        "n": int(model.nobs),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "A_beta": model.params.get("A", np.nan),
        "A_p": model.pvalues.get("A", np.nan),
        "G_beta": model.params.get("G", np.nan),
        "G_p": model.pvalues.get("G", np.nan),
        "AG_beta": model.params.get("AG", np.nan),
        "AG_p": model.pvalues.get("AG", np.nan)
    })

summary = pd.DataFrame(results)
summary["delta_r2"] = summary["r2"].diff()

summary.to_csv(
    OUT / "03_model_comparison.csv",
    index=False
)

model = fitted["M4_INTERACTION"]

X = base.copy()
X["A"] = df["A"]
X["G"] = df["G"]
X["AG"] = df["AG"]
X = sm.add_constant(X).astype(float)

df["P_HAT"] = model.predict(X)
df["MODEL_RESIDUAL"] = df["PHQ9_TOTAL"] - df["P_HAT"]

df["C_A"] = model.params["A"] * df["A"]
df["C_G"] = model.params["G"] * df["G"]
df["C_AG"] = model.params["AG"] * df["AG"]

covariate_cols = [
    c for c in X.columns
    if c not in ["A", "G", "AG"]
]

df["C_BASE"] = X[covariate_cols].dot(
    model.params[covariate_cols]
)

df["P_RECONSTRUCTED"] = (
    df["C_BASE"] +
    df["C_A"] +
    df["C_G"] +
    df["C_AG"]
)

df["DECOMPOSITION_ERROR"] = (
    df["P_HAT"] - df["P_RECONSTRUCTED"]
)

df[
    [
        "SEQN", "CYCLE",
        "LBXHGB", "LBXGH", "PHQ9_TOTAL",
        "A", "G", "AG",
        "C_A", "C_G", "C_AG",
        "C_BASE",
        "P_HAT",
        "P_RECONSTRUCTED",
        "MODEL_RESIDUAL",
        "DECOMPOSITION_ERROR"
    ]
].to_parquet(
    OUT / "03_decomposition_output.parquet",
    compression="zstd"
)

print("\nMODEL COMPARISON")
print(summary.to_string(index=False))

print("\nFINAL MODEL")
print(model.summary())

print("\nLOSS / RECONSTRUCTION")
print(
    "Model RMSE:",
    np.sqrt(np.mean(df["MODEL_RESIDUAL"] ** 2))
)
print(
    "Mean absolute residual:",
    np.mean(np.abs(df["MODEL_RESIDUAL"]))
)
print(
    "Max decomposition reconstruction error:",
    np.max(np.abs(df["DECOMPOSITION_ERROR"]))
)