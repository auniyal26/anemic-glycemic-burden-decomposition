from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
DATA = ROOT / "Data" / "NHANES"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
AUDIT = ROOT / "Audit"

FIG.mkdir(parents=True, exist_ok=True)

GLU_FILES = {
    "0506": "GLU_D.XPT",
    "0708": "GLU_E.XPT"
}

SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]
ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

cohort = pd.read_parquet(PROCESSED / "nhanes_adjustment_cohort_20plus.parquet")
working = pd.read_parquet(PROCESSED / "nhanes_core_working.parquet")

phq = working[["SEQN", "CYCLE"] + ITEMS].copy()
cohort = cohort.merge(phq, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

for c in ITEMS:
    cohort.loc[cohort[c].abs() < 1e-10, c] = 0
    cohort.loc[~cohort[c].isin([0, 1, 2, 3]), c] = np.nan

cohort["SOMATIC_SCORE"] = cohort[SOMATIC].sum(axis=1, min_count=3)

fasting_frames = []

for cycle, filename in GLU_FILES.items():
    glu = pd.read_sas(DATA / cycle / filename, format="xport")
    keep = ["SEQN", "WTSAF2YR", "LBXGLU"]
    missing = [c for c in keep if c not in glu.columns]

    if missing:
        print("FAIL  Missing fasting variables:", ", ".join(missing))
        raise SystemExit(1)

    g = glu[keep].copy()
    g["CYCLE"] = cycle
    fasting_frames.append(g)

fasting = pd.concat(fasting_frames, ignore_index=True)

df = cohort.merge(
    fasting,
    on=["SEQN", "CYCLE"],
    how="inner",
    validate="one_to_one"
)

df["WTSAF4YR"] = df["WTSAF2YR"] / 2.0
df["G_FAST10"] = (df["LBXGLU"] - 100.0).clip(lower=0) / 10.0
df["AG_FAST10"] = df["A"] * df["G_FAST10"]
df["FAST_DYSGLYCEMIA"] = (df["LBXGLU"] >= 100).astype(float)
df["FAST_DIABETES"] = (df["LBXGLU"] >= 126).astype(float)

with open(AUDIT / "15_adjustment_sets.json", "r", encoding="utf-8") as f:
    sets = json.load(f)

X3 = sets["X3_kidney_direct_effect"]

required = sorted(set(
    ["SOMATIC_SCORE", "A", "G_HBA1C", "AG_HBA1C",
     "G_FAST10", "AG_FAST10", "FAST_DYSGLYCEMIA", "FAST_DIABETES",
     "WTSAF4YR", "SDMVPSU", "SDMVSTRA", "CYCLE"] + X3
))

analysis = df.dropna(subset=required).copy()
analysis["STRATUM"] = (
    analysis["CYCLE"].astype(str)
    + "_"
    + analysis["SDMVSTRA"].astype(int).astype(str)
)
analysis["PSU"] = (
    analysis["STRATUM"]
    + "_"
    + analysis["SDMVPSU"].astype(int).astype(str)
)

def design_matrix(data, covariates, g_col, ag_col):
    X = pd.DataFrame(index=data.index)
    X["const"] = 1.0
    X["A"] = data["A"].astype(float)
    X["G"] = data[g_col].astype(float)
    X["AG"] = data[ag_col].astype(float)

    for c in covariates:
        if c in ["RIAGENDR", "RIDRETH1", "EDUC3", "SMOKING3"]:
            d = pd.get_dummies(
                data[c].astype(int),
                prefix=c,
                drop_first=True,
                dtype=float
            )
            X = pd.concat([X, d], axis=1)
        else:
            X[c] = data[c].astype(float)

    X["CYCLE_0708"] = (data["CYCLE"].astype(str) == "0708").astype(float)
    return X.astype(float)

def survey_linear(y, X, weights, strata, psu):
    y = np.asarray(y, dtype=float)
    Xv = np.asarray(X, dtype=float)
    w = np.asarray(weights, dtype=float)

    xtwx = Xv.T @ (w[:, None] * Xv)
    xtwy = Xv.T @ (w * y)

    try:
        beta = np.linalg.solve(xtwx, xtwy)
        bread = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtwx) @ xtwy
        bread = np.linalg.pinv(xtwx)

    resid = y - Xv @ beta
    scores = (w * resid)[:, None] * Xv
    meat = np.zeros((Xv.shape[1], Xv.shape[1]))

    strata = np.asarray(strata)
    psu = np.asarray(psu)

    n_strata = 0
    n_psu = 0

    for h in np.unique(strata):
        idx_h = np.where(strata == h)[0]
        psus_h = np.unique(psu[idx_h])
        m = len(psus_h)

        if m < 2:
            continue

        n_strata += 1
        n_psu += m

        totals = []
        for p in psus_h:
            idx_hp = idx_h[psu[idx_h] == p]
            totals.append(scores[idx_hp].sum(axis=0))

        totals = np.vstack(totals)
        centered = totals - totals.mean(axis=0, keepdims=True)
        meat += (m / (m - 1)) * centered.T @ centered

    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    dof = max(n_psu - n_strata, 1)
    crit = student_t.ppf(0.975, dof)
    tval = beta / se
    p = 2 * student_t.sf(np.abs(tval), dof)

    return pd.DataFrame({
        "term": X.columns,
        "beta": beta,
        "se": se,
        "ci_low": beta - crit * se,
        "ci_high": beta + crit * se,
        "p": p
    }), dof

representations = {
    "HbA1c_excess": ("G_HBA1C", "AG_HBA1C"),
    "Fasting_glucose_excess": ("G_FAST10", "AG_FAST10")
}

rows = []

for rep, (g_col, ag_col) in representations.items():
    for model_name, covariates in sets.items():
        X = design_matrix(analysis, covariates, g_col, ag_col)
        res, dof = survey_linear(
            analysis["SOMATIC_SCORE"],
            X,
            analysis["WTSAF4YR"],
            analysis["STRATUM"],
            analysis["PSU"]
        )

        for term in ["A", "G", "AG"]:
            r = res[res["term"] == term].iloc[0]
            rows.append({
                "representation": rep,
                "model": model_name,
                "term": term,
                "beta": r["beta"],
                "se": r["se"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "p": r["p"],
                "survey_dof": dof,
                "n": len(analysis)
            })

results = pd.DataFrame(rows)
results.to_csv(RESULTS / "18_fasting_glucose_sensitivity.csv", index=False)

binary_summary = pd.DataFrame({
    "measure": [
        "Fasting dysglycemia >=100 mg/dL",
        "Fasting diabetes >=126 mg/dL"
    ],
    "unweighted_n": [
        int(analysis["FAST_DYSGLYCEMIA"].sum()),
        int(analysis["FAST_DIABETES"].sum())
    ],
    "unweighted_percent": [
        100 * analysis["FAST_DYSGLYCEMIA"].mean(),
        100 * analysis["FAST_DIABETES"].mean()
    ]
})
binary_summary.to_csv(RESULTS / "18_fasting_glucose_groups.csv", index=False)

order = [
    "X0_demographic",
    "X1_primary",
    "X2_bmi_sensitivity",
    "X3_kidney_direct_effect"
]
labels = ["X0", "X1", "X2 + BMI", "X3 + Kidney"]

for term, title in [
    ("A", "A across glycemia definitions"),
    ("G", "G across glycemia definitions"),
    ("AG", "A × G across glycemia definitions")
]:
    plt.figure(figsize=(8, 5))

    for rep, display in [
        ("HbA1c_excess", "HbA1c-based G"),
        ("Fasting_glucose_excess", "Fasting-glucose G")
    ]:
        d = (
            results[
                (results["representation"] == rep)
                & (results["term"] == term)
            ]
            .set_index("model")
            .loc[order]
            .reset_index()
        )

        x = np.arange(len(order))
        plt.errorbar(
            x,
            d["beta"],
            yerr=[
                d["beta"] - d["ci_low"],
                d["ci_high"] - d["beta"]
            ],
            marker="o",
            capsize=4,
            label=display
        )

    plt.axhline(0, linewidth=0.8)
    plt.xticks(np.arange(len(order)), labels)
    plt.ylabel("Survey-weighted coefficient")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"18_{term.lower()}_fasting_sensitivity.png", dpi=300)
    plt.close()

def verdict(rep, term):
    d = results[
        (results["representation"] == rep)
        & (results["term"] == term)
    ].set_index("model").loc[order]

    same = np.all(np.sign(d["beta"]) == np.sign(d["beta"].iloc[0]))
    final = d.iloc[-1]
    nonzero = (final["ci_low"] > 0) or (final["ci_high"] < 0)

    if same and nonzero:
        return "SURVIVES"
    if same:
        return "SAME DIRECTION, UNCERTAIN"
    return "UNSTABLE"

print()
print("FASTING-GLUCOSE SENSITIVITY")
print("===========================")
print(f"PASS  Matched fasting-subset analysis n={len(analysis):,}.")
print("PASS  Correct fasting-subsample weights used.")
print()
print("HbA1c-BASED G ON FASTING SUBSET")
print("A     ", verdict("HbA1c_excess", "A"))
print("G     ", verdict("HbA1c_excess", "G"))
print("A×G   ", verdict("HbA1c_excess", "AG"))
print()
print("FASTING-GLUCOSE G")
print("A     ", verdict("Fasting_glucose_excess", "A"))
print("G     ", verdict("Fasting_glucose_excess", "G"))
print("A×G   ", verdict("Fasting_glucose_excess", "AG"))
print()
print("Full results and figures saved.")
print("NEXT  Compare definitions, then add diagnosed-diabetes/medication status if needed.")
