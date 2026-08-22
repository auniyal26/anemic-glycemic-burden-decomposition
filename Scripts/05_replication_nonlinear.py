from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

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

df["HB_THRESHOLD"] = np.where(df["RIAGENDR"] == 1, 13.0, 12.0)
df["A"] = (df["HB_THRESHOLD"] - df["LBXHGB"]).clip(lower=0)
df["G"] = (df["LBXGH"] - 5.7).clip(lower=0)
df["AG"] = df["A"] * df["G"]

df["AGE_Z"] = (
    df["RIDAGEYR"] - df["RIDAGEYR"].mean()
) / df["RIDAGEYR"].std()

df["PSU_ID"] = (
    df["CYCLE"].astype(str)
    + "_"
    + df["SDMVSTRA"].astype(int).astype(str)
    + "_"
    + df["SDMVPSU"].astype(int).astype(str)
)

def fit_linear_cycle(d):
    base = pd.DataFrame({
        "AGE_Z": d["AGE_Z"],
        "SEX": d["RIAGENDR"].astype(float)
    }, index=d.index)

    race = pd.get_dummies(
        d["RIDRETH1"].astype(int),
        prefix="RACE",
        drop_first=True,
        dtype=float
    )

    base = pd.concat([base, race], axis=1)

    rows = []

    for name, vars_ in {
        "ADD": ["A", "G"],
        "INTERACTION": ["A", "G", "AG"]
    }.items():

        X = base.copy()

        for v in vars_:
            X[v] = d[v]

        X = sm.add_constant(X).astype(float)

        model = sm.WLS(
            d["PHQ9_TOTAL"],
            X,
            weights=d["WTMEC2YR"]
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": d["PSU_ID"]}
        )

        rows.append({
            "model": name,
            "n": int(model.nobs),
            "r2": model.rsquared,
            "aic": model.aic,
            "A_beta": model.params.get("A", np.nan),
            "A_p": model.pvalues.get("A", np.nan),
            "G_beta": model.params.get("G", np.nan),
            "G_p": model.pvalues.get("G", np.nan),
            "AG_beta": model.params.get("AG", np.nan),
            "AG_p": model.pvalues.get("AG", np.nan)
        })

    return pd.DataFrame(rows)

cycle_results = []

for cycle in ["0506", "0708"]:
    temp = fit_linear_cycle(df[df["CYCLE"] == cycle].copy())
    temp.insert(0, "cycle", cycle)
    cycle_results.append(temp)

cycle_results = pd.concat(cycle_results, ignore_index=True)

cycle_results.to_csv(
    OUT / "05_cycle_replication.csv",
    index=False
)

a_max = max(5.0, df["A"].max())
g_max = max(5.0, df["G"].max())

add_formula = f"""
PHQ9_TOTAL ~ AGE_Z + C(RIAGENDR) + C(RIDRETH1)
+ bs(A, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound={a_max})
+ bs(G, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound={g_max})
"""

interaction_formula = f"""
PHQ9_TOTAL ~ AGE_Z + C(RIAGENDR) + C(RIDRETH1)
+ bs(A, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound={a_max})
* bs(G, df=4, degree=3, include_intercept=False, lower_bound=0, upper_bound={g_max})
"""

y_add, X_add = patsy.dmatrices(
    add_formula,
    df,
    return_type="dataframe"
)

y_int, X_int = patsy.dmatrices(
    interaction_formula,
    df,
    return_type="dataframe"
)

y = df["PHQ9_TOTAL"].values

def weighted_metrics(y_true, y_pred, w):
    mse = np.average((y_true - y_pred) ** 2, weights=w)
    mean = np.average(y_true, weights=w)
    sse = np.sum(w * (y_true - y_pred) ** 2)
    sst = np.sum(w * (y_true - mean) ** 2)

    return np.sqrt(mse), 1 - sse / sst

cross_results = []

for train_cycle, test_cycle in [
    ("0506", "0708"),
    ("0708", "0506")
]:

    train_mask = df["CYCLE"].eq(train_cycle).values
    test_mask = df["CYCLE"].eq(test_cycle).values

    for name, X in [
        ("NONLINEAR_ADD", X_add),
        ("NONLINEAR_INTERACTION", X_int)
    ]:

        model = sm.WLS(
            y[train_mask],
            X.loc[train_mask],
            weights=df.loc[train_mask, "WTMEC2YR"]
        ).fit()

        pred = model.predict(X.loc[test_mask])

        rmse, r2 = weighted_metrics(
            y[test_mask],
            pred,
            df.loc[test_mask, "WTMEC2YR"].values
        )

        cross_results.append({
            "train": train_cycle,
            "test": test_cycle,
            "model": name,
            "train_n": train_mask.sum(),
            "test_n": test_mask.sum(),
            "test_rmse": rmse,
            "test_r2": r2
        })

cross_results = pd.DataFrame(cross_results)

cross_results.to_csv(
    OUT / "05_cross_cycle_nonlinear.csv",
    index=False
)

pooled_results = []

for name, X in [
    ("NONLINEAR_ADD", X_add),
    ("NONLINEAR_INTERACTION", X_int)
]:

    model = sm.WLS(
        y,
        X,
        weights=df["WTMEC2YR"] / 2
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["PSU_ID"]}
    )

    pooled_results.append({
        "model": name,
        "n": int(model.nobs),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "parameters": len(model.params)
    })

pooled_results = pd.DataFrame(pooled_results)

pooled_results.to_csv(
    OUT / "05_pooled_nonlinear.csv",
    index=False
)

print("\nCYCLE REPLICATION")
print(cycle_results.to_string(index=False))

print("\nPOOLED NONLINEAR")
print(pooled_results.to_string(index=False))

print("\nCROSS-CYCLE GENERALIZATION")
print(cross_results.to_string(index=False))