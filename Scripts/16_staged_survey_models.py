from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
AUDIT = ROOT / "Audit"

FIG.mkdir(parents=True, exist_ok=True)

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]

df = pd.read_parquet(PROCESSED / "nhanes_adjustment_cohort_20plus.parquet")
working = pd.read_parquet(PROCESSED / "nhanes_core_working.parquet")

phq = working[["SEQN", "CYCLE"] + ITEMS].copy()
df = df.merge(phq, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

for c in ITEMS:
    df.loc[df[c].abs() < 1e-10, c] = 0
    df.loc[~df[c].isin([0, 1, 2, 3]), c] = np.nan

df["SOMATIC_SCORE"] = df[SOMATIC].sum(axis=1, min_count=3)

with open(AUDIT / "15_adjustment_sets.json", "r", encoding="utf-8") as f:
    sets = json.load(f)

required = sorted(set(
    ["PHQ9_TOTAL", "SOMATIC_SCORE", "A", "G_HBA1C", "AG_HBA1C",
     "WTMEC4YR", "SDMVPSU", "SDMVSTRA", "CYCLE"]
    + sets["X3_kidney_direct_effect"]
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

def design_matrix(data, covariates):
    X = pd.DataFrame(index=data.index)
    X["const"] = 1.0
    X["A"] = data["A"].astype(float)
    X["G"] = data["G_HBA1C"].astype(float)
    X["AG"] = data["AG_HBA1C"].astype(float)

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
    beta = np.linalg.solve(xtwx, xtwy)
    resid = y - Xv @ beta

    scores = (w * resid)[:, None] * Xv
    meat = np.zeros((Xv.shape[1], Xv.shape[1]), dtype=float)

    design = pd.DataFrame({
        "stratum": np.asarray(strata),
        "psu": np.asarray(psu)
    })

    unique_strata = design["stratum"].unique()
    n_strata = 0
    n_psu = 0

    for h in unique_strata:
        idx_h = np.where(design["stratum"].values == h)[0]
        psus_h = np.unique(design["psu"].values[idx_h])
        m = len(psus_h)

        if m < 2:
            continue

        n_strata += 1
        n_psu += m

        totals = []
        for p in psus_h:
            idx_hp = idx_h[design["psu"].values[idx_h] == p]
            totals.append(scores[idx_hp].sum(axis=0))

        totals = np.vstack(totals)
        centered = totals - totals.mean(axis=0, keepdims=True)
        meat += (m / (m - 1)) * (centered.T @ centered)

    bread = np.linalg.inv(xtwx)
    cov = bread @ meat @ bread
    se = np.sqrt(np.diag(cov))

    dof = max(n_psu - n_strata, 1)
    crit = student_t.ppf(0.975, dof)

    tval = beta / se
    pval = 2 * student_t.sf(np.abs(tval), dof)
    low = beta - crit * se
    high = beta + crit * se

    out = pd.DataFrame({
        "term": X.columns,
        "beta": beta,
        "se": se,
        "ci_low": low,
        "ci_high": high,
        "p": pval
    })

    return out, dof

rows = []

for outcome in ["PHQ9_TOTAL", "SOMATIC_SCORE"]:
    for model_name, covariates in sets.items():
        X = design_matrix(analysis, covariates)

        result, dof = survey_linear(
            analysis[outcome],
            X,
            analysis["WTMEC4YR"],
            analysis["STRATUM"],
            analysis["PSU"]
        )

        for term in ["A", "G", "AG"]:
            r = result[result["term"] == term].iloc[0]
            rows.append({
                "outcome": outcome,
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
results.to_csv(RESULTS / "16_staged_survey_results.csv", index=False)

order = [
    "X0_demographic",
    "X1_primary",
    "X2_bmi_sensitivity",
    "X3_kidney_direct_effect"
]

labels = {
    "X0_demographic": "X0\nDemographic",
    "X1_primary": "X1\nPrimary",
    "X2_bmi_sensitivity": "X2\n+ BMI",
    "X3_kidney_direct_effect": "X3\n+ Kidney"
}

for term, title in [
    ("A", "Hematological burden across adjustment stages"),
    ("G", "Glycemic burden across adjustment stages"),
    ("AG", "A × G interaction across adjustment stages")
]:
    plot = results[results["term"] == term].copy()

    plt.figure(figsize=(8, 5))

    for outcome, display in [
        ("PHQ9_TOTAL", "Total PHQ-9"),
        ("SOMATIC_SCORE", "Somatic symptoms")
    ]:
        d = (
            plot[plot["outcome"] == outcome]
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
    plt.xticks(np.arange(len(order)), [labels[m] for m in order])
    plt.ylabel("Survey-weighted coefficient")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"16_{term.lower()}_adjustment_trajectory.png", dpi=300)
    plt.close()

def status(outcome, term):
    d = (
        results[
            (results["outcome"] == outcome)
            & (results["term"] == term)
        ]
        .set_index("model")
        .loc[order]
    )

    nonzero = ((d["ci_low"] > 0) | (d["ci_high"] < 0))
    same_direction = np.all(np.sign(d["beta"]) == np.sign(d["beta"].iloc[0]))

    if nonzero.all() and same_direction:
        return "SURVIVES"
    if nonzero.iloc[0] and not nonzero.iloc[-1]:
        return "ATTENUATES"
    if not nonzero.any():
        return "NO CLEAR SUPPORT"
    return "MIXED"

print()
print("STAGED SURVEY RESULTS")
print("=====================")
print(f"PASS  Matched survey-analysis sample fixed at n={len(analysis):,}.")
print("PASS  Four-year MEC weights, cycle-specific strata and PSU used.")
print()
print("TOTAL PHQ-9")
print("A     ", status("PHQ9_TOTAL", "A"))
print("G     ", status("PHQ9_TOTAL", "G"))
print("A×G   ", status("PHQ9_TOTAL", "AG"))
print()
print("SOMATIC PHQ DIMENSION")
print("A     ", status("SOMATIC_SCORE", "A"))
print("G     ", status("SOMATIC_SCORE", "G"))
print("A×G   ", status("SOMATIC_SCORE", "AG"))
print()
print("Figures and full coefficients saved in Results/.")
print("NEXT  Interpret what survives, then run fasting-glucose and ordinal-factor validation.")
