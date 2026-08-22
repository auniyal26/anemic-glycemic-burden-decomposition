from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "Processed"
OUT = ROOT / "Results"

df = pd.read_parquet(DATA / "nhanes_core_working.parquet")

cols = [
    "SEQN", "CYCLE", "LBXHGB", "LBXGH", "PHQ9_TOTAL",
    "DEPRESSION_10", "RIDAGEYR", "RIAGENDR", "RIDRETH1",
    "WTMEC2YR", "SDMVPSU", "SDMVSTRA"
]

df = df[cols].dropna().copy()

df["HB_THRESHOLD"] = np.where(df["RIAGENDR"] == 1, 13.0, 12.0)

df["A_DEFICIT"] = (
    df["HB_THRESHOLD"] - df["LBXHGB"]
).clip(lower=0)

df["G_EXCESS"] = (
    df["LBXGH"] - 5.7
).clip(lower=0)

df["AG_CLINICAL"] = df["A_DEFICIT"] * df["G_EXCESS"]

df["ANEMIA"] = (df["A_DEFICIT"] > 0).astype(int)
df["DYSGLYCEMIA"] = (df["LBXGH"] >= 5.7).astype(int)
df["BOTH"] = df["ANEMIA"] * df["DYSGLYCEMIA"]

df["AGE_Z"] = (
    df["RIDAGEYR"] - df["RIDAGEYR"].mean()
) / df["RIDAGEYR"].std()

df["WEIGHT"] = df["WTMEC2YR"] / 2

df["PSU_ID"] = (
    df["CYCLE"].astype(str)
    + "_"
    + df["SDMVSTRA"].astype(int).astype(str)
    + "_"
    + df["SDMVPSU"].astype(int).astype(str)
)

base = pd.DataFrame({
    "AGE_Z": df["AGE_Z"],
    "SEX": df["RIAGENDR"].astype(float)
})

race = pd.get_dummies(
    df["RIDRETH1"].astype(int),
    prefix="RACE",
    drop_first=True,
    dtype=float
)

base = pd.concat([base, race], axis=1)

models = {
    "M0_X": [],
    "M1_A_CLINICAL": ["A_DEFICIT"],
    "M2_G_CLINICAL": ["G_EXCESS"],
    "M3_A_G": ["A_DEFICIT", "G_EXCESS"],
    "M4_INTERACTION": [
        "A_DEFICIT",
        "G_EXCESS",
        "AG_CLINICAL"
    ],
    "M5_BINARY": [
        "ANEMIA",
        "DYSGLYCEMIA",
        "BOTH"
    ]
}

results = []

for name, variables in models.items():
    X = base.copy()

    for v in variables:
        X[v] = df[v]

    X = sm.add_constant(X).astype(float)

    model = sm.WLS(
        df["PHQ9_TOTAL"],
        X,
        weights=df["WEIGHT"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["PSU_ID"]}
    )

    row = {
        "model": name,
        "n": int(model.nobs),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic
    }

    for v in variables:
        row[f"{v}_beta"] = model.params.get(v, np.nan)
        row[f"{v}_p"] = model.pvalues.get(v, np.nan)

    results.append(row)

summary = pd.DataFrame(results)

summary.to_csv(
    OUT / "04_domain_model_comparison.csv",
    index=False
)

df["GROUP"] = (
    df["ANEMIA"].astype(str)
    + "_"
    + df["DYSGLYCEMIA"].astype(str)
)

groups = []

for group, g in df.groupby("GROUP"):
    groups.append({
        "group": group,
        "n": len(g),
        "mean_hb": g["LBXHGB"].mean(),
        "mean_hba1c": g["LBXGH"].mean(),
        "mean_phq9": np.average(
            g["PHQ9_TOTAL"],
            weights=g["WEIGHT"]
        ),
        "depression_prevalence": np.average(
            g["DEPRESSION_10"].astype(float),
            weights=g["WEIGHT"]
        )
    })

groups = pd.DataFrame(groups)

groups.to_csv(
    OUT / "04_clinical_groups.csv",
    index=False
)

print("\nCLINICAL GROUPS")
print(groups.to_string(index=False))

print("\nDOMAIN-INFORMED MODELS")
print(summary.to_string(index=False))