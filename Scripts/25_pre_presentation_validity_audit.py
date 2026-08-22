from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
DATA = ROOT / "Data" / "NHANES"
RLIB = ROOT / "R_library"

RESULTS.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RLIB.mkdir(parents=True, exist_ok=True)

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]

print()
print("PRE-PRESENTATION VALIDITY AUDIT")
print("===============================")

# 1) Rebuild exact script-16 cohort
print("[1/4] Rebuilding exact survey-analysis cohort...")

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
analysis["CYCLE"] = analysis["CYCLE"].astype(str)
analysis["STRATUM"] = analysis["CYCLE"] + "_" + analysis["SDMVSTRA"].astype(int).astype(str)
analysis["PSU"] = analysis["STRATUM"] + "_" + analysis["SDMVPSU"].astype(int).astype(str)

print(f"PASS  Exact complete-case survey sample rebuilt: n={len(analysis):,}")

# 2) Missingness / selection audit
print("[2/4] Auditing complete-case selection...")

eligible_n = len(df)
included_n = len(analysis)
selection_rate = included_n / eligible_n if eligible_n else np.nan

selection_rows = [{
    "metric": "overall_selection_rate",
    "eligible_n": eligible_n,
    "included_n": included_n,
    "value": selection_rate
}]

for cycle in sorted(df["CYCLE"].astype(str).unique()):
    e = df[df["CYCLE"].astype(str) == cycle]
    a = analysis[analysis["CYCLE"] == cycle]
    selection_rows.append({
        "metric": f"selection_rate_cycle_{cycle}",
        "eligible_n": len(e),
        "included_n": len(a),
        "value": len(a) / len(e) if len(e) else np.nan
    })

missing_cols = [
    "A", "G_HBA1C", "PHQ9_TOTAL", "RIDAGEYR", "RIAGENDR", "RIDRETH1",
    "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"
]

for c in missing_cols:
    if c in df.columns:
        selection_rows.append({
            "metric": f"missing_{c}",
            "eligible_n": eligible_n,
            "included_n": int(df[c].notna().sum()),
            "value": float(df[c].isna().mean())
        })

selection = pd.DataFrame(selection_rows)
selection.to_csv(RESULTS / "25_missingness_selection_audit.csv", index=False)

df["_INCLUDED"] = df["SEQN"].isin(set(analysis["SEQN"])).astype(int)
compare_rows = []

for c in ["RIDAGEYR", "INDFMPIR", "BMXBMI", "EGFR_2021"]:
    if c not in df.columns:
        continue
    for inc in [0, 1]:
        x = pd.to_numeric(df.loc[df["_INCLUDED"] == inc, c], errors="coerce").dropna()
        if len(x):
            compare_rows.append({
                "variable": c,
                "group": "included" if inc else "excluded",
                "n": len(x),
                "mean": x.mean(),
                "sd": x.std(ddof=1)
            })

pd.DataFrame(compare_rows).to_csv(
    RESULTS / "25_included_vs_excluded_descriptives.csv",
    index=False
)

print(f"INFO  Complete-case selection rate = {selection_rate:.1%}")
if selection_rate < 0.70:
    print("WARNING  Less than 70% of eligible cohort remains; selection bias needs explicit discussion.")
else:
    print("PASS  More than 70% of eligible cohort remains, but missingness still requires reporting.")

# 3) R survey parity
print("[3/4] Cross-checking custom survey estimator against R survey::svyglm...")

survey_input = RESULTS / "25_survey_parity_input.csv"
export_cols = sorted(set(
    ["SEQN", "CYCLE", "STRATUM", "PSU", "WTMEC4YR",
     "PHQ9_TOTAL", "SOMATIC_SCORE", "A", "G_HBA1C", "AG_HBA1C"]
    + sets["X3_kidney_direct_effect"]
))
analysis[export_cols].to_csv(survey_input, index=False)

rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])

if rscript is None:
    print("FAIL  Rscript not found.")
    raise SystemExit(1)

x3 = sets["X3_kidney_direct_effect"]
cat_vars = {"RIAGENDR", "RIDRETH1", "EDUC3", "SMOKING3"}

rhs = ["A", "G_HBA1C", "AG_HBA1C"]
for c in x3:
    rhs.append(f"factor({c})" if c in cat_vars else c)
rhs.append("factor(CYCLE)")
rhs_str = " + ".join(rhs)

r_code = f'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("survey", quietly=TRUE)) {{
    install.packages("survey", repos="https://cloud.r-project.org", lib=user_lib)
}}
library(survey)

d <- read.csv(input_file, stringsAsFactors=FALSE)
options(survey.lonely.psu="adjust")

des <- svydesign(
    ids=~PSU,
    strata=~STRATUM,
    weights=~WTMEC4YR,
    nest=TRUE,
    data=d
)

fit_total <- svyglm(
    PHQ9_TOTAL ~ {rhs_str},
    design=des,
    family=gaussian()
)

fit_somatic <- svyglm(
    SOMATIC_SCORE ~ {rhs_str},
    design=des,
    family=gaussian()
)

extract_core <- function(fit, outcome) {{
    sm <- summary(fit)$coefficients
    ci <- confint(fit)
    terms <- c("A","G_HBA1C","AG_HBA1C")
    out <- data.frame()
    for (term in terms) {{
        out <- rbind(out, data.frame(
            outcome=outcome,
            term=term,
            beta=sm[term,"Estimate"],
            se=sm[term,"Std. Error"],
            p=sm[term,"Pr(>|t|)"],
            ci_low=ci[term,1],
            ci_high=ci[term,2]
        ))
    }}
    out
}}

res <- rbind(
    extract_core(fit_total, "PHQ9_TOTAL"),
    extract_core(fit_somatic, "SOMATIC_SCORE")
)

write.csv(
    res,
    file.path(results_dir, "25_r_survey_parity_results.csv"),
    row.names=FALSE
)
'''

r_path = AUDIT / "25_r_survey_parity.R"
r_path.write_text(r_code, encoding="utf-8")

proc = subprocess.run(
    [str(rscript), str(r_path), str(survey_input), str(RESULTS), str(RLIB)],
    capture_output=True,
    text=True
)

if proc.returncode != 0:
    print("FAIL  R survey parity check failed.")
    print(proc.stdout[-2000:])
    print(proc.stderr[-2000:])
    raise SystemExit(proc.returncode)

r_results = pd.read_csv(RESULTS / "25_r_survey_parity_results.csv")
py_results = pd.read_csv(RESULTS / "16_staged_survey_results.csv")
py_results = py_results[py_results["model"] == "X3_kidney_direct_effect"].copy()

term_map = {"A": "A", "G_HBA1C": "G", "AG_HBA1C": "AG"}
parity_rows = []

for _, rr in r_results.iterrows():
    py_term = term_map[rr["term"]]
    pr = py_results[
        (py_results["outcome"] == rr["outcome"]) &
        (py_results["term"] == py_term)
    ].iloc[0]

    parity_rows.append({
        "outcome": rr["outcome"],
        "term": rr["term"],
        "python_beta": pr["beta"],
        "r_beta": rr["beta"],
        "abs_beta_diff": abs(pr["beta"] - rr["beta"]),
        "python_se": pr["se"],
        "r_se": rr["se"],
        "se_ratio_r_over_python": rr["se"] / pr["se"],
        "python_p": pr["p"],
        "r_p": rr["p"]
    })

parity = pd.DataFrame(parity_rows)
parity.to_csv(RESULTS / "25_survey_estimator_parity.csv", index=False)

max_beta_diff = parity["abs_beta_diff"].max()
se_ratio_min = parity["se_ratio_r_over_python"].min()
se_ratio_max = parity["se_ratio_r_over_python"].max()

if max_beta_diff < 1e-6:
    print("PASS  Python and R survey point estimates match.")
else:
    print(f"WARNING  Survey point-estimate mismatch; max |Δβ|={max_beta_diff:.6g}")

if se_ratio_min >= 0.90 and se_ratio_max <= 1.10:
    print(f"PASS  R/Python SE ratios are within 10% ({se_ratio_min:.3f}–{se_ratio_max:.3f}).")
else:
    print(f"WARNING  R/Python SE ratios differ materially ({se_ratio_min:.3f}–{se_ratio_max:.3f}).")

# 4) Reduced-basis A PCA sensitivity
print("[4/4] Repeating A decomposition with a reduced CBC basis...")

A_REDUCED = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
cycles = {"0506": "CBC_D.XPT", "0708": "CBC_E.XPT"}
models = {}

for cycle, filename in cycles.items():
    cbc = pd.read_sas(DATA / cycle / filename, format="xport")
    cbc = cbc[["SEQN"] + A_REDUCED].dropna().copy()

    scaler = StandardScaler()
    z = scaler.fit_transform(cbc[A_REDUCED])
    pca = PCA()
    scores = pca.fit_transform(z)

    models[cycle] = {
        "data": cbc,
        "pca": pca,
        "loadings": pca.components_.T,
        "scores": scores
    }

def align(ref, target):
    sim = np.abs(ref.T @ target)
    row, col = linear_sum_assignment(-sim)
    order = col[np.argsort(row)]
    aligned = target[:, order].copy()
    signs = np.sign(np.sum(ref * aligned, axis=0))
    signs[signs == 0] = 1
    aligned *= signs
    similarity = np.abs(np.sum(ref * aligned, axis=0))
    return order, signs, aligned, similarity

ref = models["0506"]["loadings"]
order, signs, aligned, similarity = align(ref, models["0708"]["loadings"])

reduced_rows = []
for j in range(len(A_REDUCED)):
    reduced_rows.append({
        "component": j + 1,
        "variance_0506": models["0506"]["pca"].explained_variance_ratio_[j],
        "variance_0708": models["0708"]["pca"].explained_variance_ratio_[order[j]],
        "cross_cycle_similarity": similarity[j]
    })

reduced = pd.DataFrame(reduced_rows)
reduced.to_csv(RESULTS / "25_reduced_A_pca_sensitivity.csv", index=False)

for _, row in reduced.iterrows():
    print(
        f"A-reduced PC{int(row.component)}: "
        f"variance {row.variance_0506:.3f}/{row.variance_0708:.3f}, "
        f"similarity {row.cross_cycle_similarity:.3f}"
    )

print()
print("AUDIT VERDICT")
print("=============")

survey_ok = (
    max_beta_diff < 1e-6
    and se_ratio_min >= 0.90
    and se_ratio_max <= 1.10
)

stable_reduced = (
    reduced.loc[reduced["component"] <= 3, "cross_cycle_similarity"].min() >= 0.90
)

if survey_ok:
    print("PASS  Main survey inference reproduces in R survey.")
else:
    print("WARNING  Main survey inference needs resolution before presentation claims are frozen.")

if stable_reduced:
    print("PASS  Reduced-basis hematology still contains reproducible multivariate structure.")
else:
    print("WARNING  Original A-PCA dimensionality may be substantially driven by derived CBC indices.")

if selection_rate >= 0.70:
    print("PASS/WARN  Retained sample is reasonably large; report missingness explicitly.")
else:
    print("WARNING  Complete-case selection is substantial and must be treated as a major limitation.")

print()
print("If the R survey parity check passes, the core A/G separable-association result is cleared")
print("for a PRELIMINARY presentation. It remains an association result, not causal proof.")
print()
print("Saved:")
print("  Results/25_missingness_selection_audit.csv")
print("  Results/25_included_vs_excluded_descriptives.csv")
print("  Results/25_r_survey_parity_results.csv")
print("  Results/25_survey_estimator_parity.csv")
print("  Results/25_reduced_A_pca_sensitivity.csv")
