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

DIQ_FILES = {
    "0506": "DIQ_D.XPT",
    "0708": "DIQ_E.XPT"
}

GLU_FILES = {
    "0506": "GLU_D.XPT",
    "0708": "GLU_E.XPT"
}

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]

for cycle in ["0506", "0708"]:
    for filename in [DIQ_FILES[cycle], GLU_FILES[cycle]]:
        path = DATA / cycle / filename
        if not path.exists():
            print()
            print("CLINICAL DIABETES SENSITIVITY")
            print("=============================")
            print(f"FAIL  Missing {path}")
            raise SystemExit(1)

cohort = pd.read_parquet(PROCESSED / "nhanes_adjustment_cohort_20plus.parquet")
working = pd.read_parquet(PROCESSED / "nhanes_core_working.parquet")

phq = working[["SEQN", "CYCLE"] + ITEMS].copy()
cohort = cohort.merge(phq, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

for c in ITEMS:
    cohort.loc[cohort[c].abs() < 1e-10, c] = 0
    cohort.loc[~cohort[c].isin([0, 1, 2, 3]), c] = np.nan

cohort["SOMATIC_SCORE"] = cohort[SOMATIC].sum(axis=1, min_count=3)

frames = []

for cycle in ["0506", "0708"]:
    diq = pd.read_sas(DATA / cycle / DIQ_FILES[cycle], format="xport")
    glu = pd.read_sas(DATA / cycle / GLU_FILES[cycle], format="xport")

    wanted_diq = ["SEQN", "DIQ010", "DIQ050", "DID070"]
    missing = [c for c in wanted_diq if c not in diq.columns]
    if missing:
        print()
        print("CLINICAL DIABETES SENSITIVITY")
        print("=============================")
        print("FAIL  Missing DIQ variables:", ", ".join(missing))
        raise SystemExit(1)

    wanted_glu = ["SEQN", "LBXGLU", "WTSAF2YR"]
    missing = [c for c in wanted_glu if c not in glu.columns]
    if missing:
        print()
        print("CLINICAL DIABETES SENSITIVITY")
        print("=============================")
        print("FAIL  Missing GLU variables:", ", ".join(missing))
        raise SystemExit(1)

    d = diq[wanted_diq].merge(
        glu[wanted_glu],
        on="SEQN",
        how="inner",
        validate="one_to_one"
    )
    d["CYCLE"] = cycle
    frames.append(d)

clinical = pd.concat(frames, ignore_index=True)

df = cohort.merge(
    clinical,
    on=["SEQN", "CYCLE"],
    how="inner",
    validate="one_to_one"
)

df["WTSAF4YR"] = df["WTSAF2YR"] / 2.0

df["DX_DIABETES"] = np.where(
    df["DIQ010"] == 1,
    1.0,
    np.where(df["DIQ010"].isin([2, 3]), 0.0, np.nan)
)

df["ON_INSULIN"] = np.where(
    df["DIQ050"] == 1,
    1.0,
    np.where(df["DIQ050"] == 2, 0.0, np.nan)
)

df["ON_PILLS"] = np.where(
    df["DID070"] == 1,
    1.0,
    np.where(df["DID070"] == 2, 0.0, np.nan)
)

df["TREATED_DIABETES"] = np.where(
    df["DX_DIABETES"] == 1,
    ((df["ON_INSULIN"] == 1) | (df["ON_PILLS"] == 1)).astype(float),
    0.0
)

df["CLINICAL_G_STATE"] = np.nan

normal = (
    (df["DX_DIABETES"] == 0)
    & (df["LBXGLU"] < 100)
)

prediabetes = (
    (df["DX_DIABETES"] == 0)
    & (df["LBXGLU"] >= 100)
    & (df["LBXGLU"] < 126)
)

undiagnosed_diabetes = (
    (df["DX_DIABETES"] == 0)
    & (df["LBXGLU"] >= 126)
)

diagnosed_untreated = (
    (df["DX_DIABETES"] == 1)
    & (df["TREATED_DIABETES"] == 0)
)

diagnosed_treated = (
    (df["DX_DIABETES"] == 1)
    & (df["TREATED_DIABETES"] == 1)
)

df.loc[normal, "CLINICAL_G_STATE"] = 0
df.loc[prediabetes, "CLINICAL_G_STATE"] = 1
df.loc[undiagnosed_diabetes, "CLINICAL_G_STATE"] = 2
df.loc[diagnosed_untreated, "CLINICAL_G_STATE"] = 3
df.loc[diagnosed_treated, "CLINICAL_G_STATE"] = 4

state_labels = {
    0: "Normal",
    1: "Prediabetes",
    2: "Undiagnosed diabetes",
    3: "Diagnosed untreated",
    4: "Diagnosed treated"
}

with open(AUDIT / "15_adjustment_sets.json", "r", encoding="utf-8") as f:
    sets = json.load(f)

X3 = sets["X3_kidney_direct_effect"]

required = sorted(set(
    ["SOMATIC_SCORE", "A", "CLINICAL_G_STATE",
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

def design_matrix(data, covariates):
    X = pd.DataFrame(index=data.index)
    X["const"] = 1.0
    X["A"] = data["A"].astype(float)

    g = pd.get_dummies(
        data["CLINICAL_G_STATE"].astype(int),
        prefix="GSTATE",
        drop_first=True,
        dtype=float
    )
    X = pd.concat([X, g], axis=1)

    for c in g.columns:
        X[f"A_x_{c}"] = X["A"] * X[c]

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

    with np.errstate(divide="ignore", invalid="ignore"):
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

rows = []

for model_name, covariates in sets.items():
    X = design_matrix(analysis, covariates)

    res, dof = survey_linear(
        analysis["SOMATIC_SCORE"],
        X,
        analysis["WTSAF4YR"],
        analysis["STRATUM"],
        analysis["PSU"]
    )

    keep_terms = [
        "A",
        "GSTATE_1",
        "GSTATE_2",
        "GSTATE_3",
        "GSTATE_4",
        "A_x_GSTATE_1",
        "A_x_GSTATE_2",
        "A_x_GSTATE_3",
        "A_x_GSTATE_4"
    ]

    for term in keep_terms:
        if term not in res["term"].values:
            continue

        r = res[res["term"] == term].iloc[0]
        rows.append({
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
results.to_csv(
    RESULTS / "19_clinical_diabetes_sensitivity.csv",
    index=False
)

counts = (
    analysis["CLINICAL_G_STATE"]
    .value_counts()
    .sort_index()
    .rename_axis("state")
    .reset_index(name="n")
)
counts["label"] = counts["state"].map(state_labels)
counts["percent"] = counts["n"] / len(analysis) * 100

counts.to_csv(
    RESULTS / "19_clinical_diabetes_groups.csv",
    index=False
)

full = results[
    results["model"] == "X3_kidney_direct_effect"
].copy()

state_rows = []

for state in [0, 1, 2, 3, 4]:
    if state == 0:
        beta = 0.0
        low = 0.0
        high = 0.0
    else:
        term = f"GSTATE_{state}"
        r = full[full["term"] == term].iloc[0]
        beta = r["beta"]
        low = r["ci_low"]
        high = r["ci_high"]

    state_rows.append({
        "state": state,
        "label": state_labels[state],
        "beta": beta,
        "ci_low": low,
        "ci_high": high
    })

state_effects = pd.DataFrame(state_rows)

plt.figure(figsize=(9, 5))
x = np.arange(len(state_effects))
plt.errorbar(
    x,
    state_effects["beta"],
    yerr=[
        state_effects["beta"] - state_effects["ci_low"],
        state_effects["ci_high"] - state_effects["beta"]
    ],
    fmt="o",
    capsize=4
)
plt.axhline(0, linewidth=0.8)
plt.xticks(x, state_effects["label"], rotation=25, ha="right")
plt.ylabel("Adjusted somatic-score difference")
plt.title("Somatic symptoms across clinical glycemic states")
plt.tight_layout()
plt.savefig(
    FIG / "19_clinical_glycemia_states.png",
    dpi=300
)
plt.close()

interaction_rows = []

for state in [1, 2, 3, 4]:
    term = f"A_x_GSTATE_{state}"
    r = full[full["term"] == term].iloc[0]
    interaction_rows.append({
        "state": state,
        "label": state_labels[state],
        "beta": r["beta"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"]
    })

interaction = pd.DataFrame(interaction_rows)

plt.figure(figsize=(9, 5))
x = np.arange(len(interaction))
plt.errorbar(
    x,
    interaction["beta"],
    yerr=[
        interaction["beta"] - interaction["ci_low"],
        interaction["ci_high"] - interaction["beta"]
    ],
    fmt="o",
    capsize=4
)
plt.axhline(0, linewidth=0.8)
plt.xticks(x, interaction["label"], rotation=25, ha="right")
plt.ylabel("A × glycemic-state interaction")
plt.title("Does anemia behave differently across diabetes states?")
plt.tight_layout()
plt.savefig(
    FIG / "19_anemia_by_diabetes_state.png",
    dpi=300
)
plt.close()

def nonzero(row):
    return (row["ci_low"] > 0) or (row["ci_high"] < 0)

a = full[full["term"] == "A"].iloc[0]
state_supported = [
    state_labels[int(r.term.split("_")[1])]
    for r in full.itertuples()
    if r.term.startswith("GSTATE_") and nonzero(r._asdict())
]
interaction_supported = [
    state_labels[int(r.term.split("_")[-1])]
    for r in full.itertuples()
    if r.term.startswith("A_x_GSTATE_") and nonzero(r._asdict())
]

print()
print("CLINICAL DIABETES SENSITIVITY")
print("=============================")
print(f"PASS  Matched fasting + diabetes-status sample n={len(analysis):,}.")

if nonzero(a):
    print("PASS  Anemia association survives clinical diabetes-state adjustment.")
else:
    print("UNCERTAIN  Anemia association crosses zero after clinical-state adjustment.")

if state_supported:
    print("SIGNAL  Glycemic states associated with somatic symptoms:", ", ".join(state_supported))
else:
    print("NO CLEAR SIGNAL  Clinical glycemic-state contrasts cross zero.")

if interaction_supported:
    print("SIGNAL  A interaction detected in:", ", ".join(interaction_supported))
else:
    print("NO CLEAR SUPPORT  No anemia × clinical-diabetes-state interaction survives.")

print("PASS  Treatment status separated from diagnosis.")
print("Figures and full coefficients saved.")
print()
print("NEXT  Use these results to decide whether G is physiology-, disease-, or treatment-stage driven.")
