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
df["WTMEC2YR_RESTORED"] = df["WTMEC4YR"] * 2.0

with open(AUDIT / "15_adjustment_sets.json", "r", encoding="utf-8") as f:
    sets = json.load(f)

X3 = sets["X3_kidney_direct_effect"]

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
    meat = np.zeros((Xv.shape[1], Xv.shape[1]), dtype=float)

    strata_arr = np.asarray(strata)
    psu_arr = np.asarray(psu)

    n_strata = 0
    n_psu = 0

    for h in np.unique(strata_arr):
        idx_h = np.where(strata_arr == h)[0]
        psus_h = np.unique(psu_arr[idx_h])
        m = len(psus_h)

        if m < 2:
            continue

        n_strata += 1
        n_psu += m

        totals = []
        for p in psus_h:
            idx_hp = idx_h[psu_arr[idx_h] == p]
            totals.append(scores[idx_hp].sum(axis=0))

        totals = np.vstack(totals)
        centered = totals - totals.mean(axis=0, keepdims=True)
        meat += (m / (m - 1)) * (centered.T @ centered)

    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))

    dof = max(n_psu - n_strata, 1)
    crit = student_t.ppf(0.975, dof)

    with np.errstate(divide="ignore", invalid="ignore"):
        tval = beta / se

    pval = 2 * student_t.sf(np.abs(tval), dof)
    low = beta - crit * se
    high = beta + crit * se

    return pd.DataFrame({
        "term": X.columns,
        "beta": beta,
        "se": se,
        "ci_low": low,
        "ci_high": high,
        "p": pval
    }), dof

rows = []

for cycle in ["0506", "0708"]:
    d = df[df["CYCLE"].astype(str) == cycle].copy()

    required = sorted(set(
        ["PHQ9_TOTAL", "SOMATIC_SCORE", "A", "G_HBA1C", "AG_HBA1C",
         "WTMEC2YR_RESTORED", "SDMVPSU", "SDMVSTRA"] + X3
    ))

    d = d.dropna(subset=required).copy()
    d["PSU"] = d["SDMVPSU"].astype(int).astype(str)
    d["STRATUM"] = d["SDMVSTRA"].astype(int).astype(str)

    for outcome in ["PHQ9_TOTAL", "SOMATIC_SCORE"]:
        for model_name, covariates in sets.items():
            X = design_matrix(d, covariates)

            result, dof = survey_linear(
                d[outcome],
                X,
                d["WTMEC2YR_RESTORED"],
                d["STRATUM"],
                d["PSU"]
            )

            for term in ["A", "G", "AG"]:
                r = result[result["term"] == term].iloc[0]
                rows.append({
                    "cycle": cycle,
                    "outcome": outcome,
                    "model": model_name,
                    "term": term,
                    "beta": r["beta"],
                    "se": r["se"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "p": r["p"],
                    "survey_dof": dof,
                    "n": len(d)
                })

results = pd.DataFrame(rows)
results.to_csv(RESULTS / "17_cross_cycle_survey_replication.csv", index=False)

order = [
    "X0_demographic",
    "X1_primary",
    "X2_bmi_sensitivity",
    "X3_kidney_direct_effect"
]

xlabels = ["X0", "X1", "X2 + BMI", "X3 + Kidney"]

for term, title in [
    ("A", "Hematological burden: independent-cycle replication"),
    ("G", "Glycemic burden: independent-cycle replication"),
    ("AG", "A × G interaction: independent-cycle replication")
]:
    plot = results[
        (results["term"] == term)
        & (results["outcome"] == "SOMATIC_SCORE")
    ].copy()

    plt.figure(figsize=(8, 5))

    for cycle, display in [("0506", "2005–06"), ("0708", "2007–08")]:
        d = (
            plot[plot["cycle"] == cycle]
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
    plt.xticks(np.arange(len(order)), xlabels)
    plt.ylabel("Survey-weighted coefficient")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"17_{term.lower()}_somatic_cross_cycle.png", dpi=300)
    plt.close()

final = results[
    results["model"] == "X3_kidney_direct_effect"
].copy()

def replicated(outcome, term):
    d = final[
        (final["outcome"] == outcome)
        & (final["term"] == term)
    ].sort_values("cycle")

    if len(d) != 2:
        return "INCOMPLETE"

    signs = np.sign(d["beta"].values)
    same_direction = signs[0] == signs[1]
    nonzero = ((d["ci_low"] > 0) | (d["ci_high"] < 0)).values

    if same_direction and nonzero.all():
        return "REPLICATES"
    if same_direction and nonzero.any():
        return "PARTIAL"
    if same_direction:
        return "SAME DIRECTION, UNCERTAIN"
    return "DOES NOT REPLICATE"

print()
print("INDEPENDENT-CYCLE REPLICATION")
print("=============================")

for cycle in ["0506", "0708"]:
    n = int(final[final["cycle"] == cycle]["n"].iloc[0])
    print(f"PASS  {cycle} fixed X3-complete sample n={n:,}.")

print()
print("TOTAL PHQ-9")
print("A     ", replicated("PHQ9_TOTAL", "A"))
print("G     ", replicated("PHQ9_TOTAL", "G"))
print("A×G   ", replicated("PHQ9_TOTAL", "AG"))

print()
print("SOMATIC PHQ DIMENSION")
print("A     ", replicated("SOMATIC_SCORE", "A"))
print("G     ", replicated("SOMATIC_SCORE", "G"))
print("A×G   ", replicated("SOMATIC_SCORE", "AG"))

print()
print("Figures and full cycle-specific results saved.")
print("NEXT  If somatic A×G replicates, test glycemia with fasting glucose and diabetes definitions.")
