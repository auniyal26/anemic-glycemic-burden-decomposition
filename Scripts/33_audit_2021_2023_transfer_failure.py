from pathlib import Path
import shutil
import subprocess
import pandas as pd
import numpy as np

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

INPUT = PROCESSED / "31_transfer_harmonized_FINAL_PREMODEL.parquet"
CSV_INPUT = RESULTS / "33_2123_diagnostic_input.csv"

RESULTS.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RLIB.mkdir(parents=True, exist_ok=True)

print()
print("2021-2023 TRANSFER DIAGNOSTIC")
print("=============================")
print("Diagnostic only: no frozen definition is changed.")

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_parquet(INPUT)

# ------------------------------------------------------------------
# 1) Descriptive diagnostics before any new inferential interpretation
# ------------------------------------------------------------------
xsets = {
    "X0": ["RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN"],
    "X1": ["RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN", "INDFMPIR", "EDUC3", "SMOKING3"],
    "X2": ["RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI"],
    "X3": ["RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"],
}

core = ["SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C", "AG_HBA1C"]

miss_rows = []
for period, mask in {
    "2009-2018": df["CYCLE"].isin(["0910", "1112", "1314", "1516", "1718"]),
    "2021-2023": df["CYCLE"] == "2123",
}.items():
    d = df.loc[mask].copy()
    for c in sorted(set(core + xsets["X3"])):
        miss_rows.append({
            "period": period,
            "variable": c,
            "n": len(d),
            "missing_n": int(d[c].isna().sum()),
            "missing_pct": float(d[c].isna().mean() * 100),
        })

pd.DataFrame(miss_rows).to_csv(
    RESULTS / "33_missingness_by_period.csv", index=False
)

# Distribution checks: these do not change the model.
desc_rows = []
for period, mask in {
    "2009-2018": df["CYCLE"].isin(["0910", "1112", "1314", "1516", "1718"]),
    "2021-2023": df["CYCLE"] == "2123",
}.items():
    d = df.loc[mask].copy()
    for c in ["A", "G_HBA1C", "SOMATIC_SCORE", "PHQ9_TOTAL", "LBXHGB", "LBXGH"]:
        x = pd.to_numeric(d[c], errors="coerce").dropna()
        if len(x):
            desc_rows.append({
                "period": period,
                "variable": c,
                "n": len(x),
                "mean": x.mean(),
                "sd": x.std(ddof=1),
                "median": x.median(),
                "p25": x.quantile(.25),
                "p75": x.quantile(.75),
                "zero_pct": float((x == 0).mean() * 100),
            })

pd.DataFrame(desc_rows).to_csv(
    RESULTS / "33_distribution_comparison.csv", index=False
)

# Category counts to expose sparse cells.
cat_rows = []
for c in ["RIAGENDR", "RIDRETH_FROZEN", "EDUC3", "SMOKING3"]:
    q = (
        df[df["CYCLE"] == "2123"]
        .groupby(c, dropna=False)
        .size()
        .reset_index(name="n")
    )
    q["variable"] = c
    q = q.rename(columns={c: "level"})
    cat_rows.append(q[["variable", "level", "n"]])

pd.concat(cat_rows, ignore_index=True).to_csv(
    RESULTS / "33_2123_category_counts.csv", index=False
)

df.to_csv(CSV_INPUT, index=False)

# ------------------------------------------------------------------
# 2) Frozen prespecified X0 -> X3 ladder in R survey
# ------------------------------------------------------------------
rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])
if rscript is None:
    raise RuntimeError("Rscript not found.")

r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("survey", quietly=TRUE)) {
    install.packages("survey", repos="https://cloud.r-project.org", lib=user_lib)
}
library(survey)
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)
d$CYCLE <- as.character(d$CYCLE)
d <- d[d$CYCLE=="2123", ]

xsets <- list(
    X0=c("RIDAGEYR","factor(RIAGENDR)","factor(RIDRETH_FROZEN)"),
    X1=c("RIDAGEYR","factor(RIAGENDR)","factor(RIDRETH_FROZEN)",
         "INDFMPIR","factor(EDUC3)","factor(SMOKING3)"),
    X2=c("RIDAGEYR","factor(RIAGENDR)","factor(RIDRETH_FROZEN)",
         "INDFMPIR","factor(EDUC3)","factor(SMOKING3)","BMXBMI"),
    X3=c("RIDAGEYR","factor(RIAGENDR)","factor(RIDRETH_FROZEN)",
         "INDFMPIR","factor(EDUC3)","factor(SMOKING3)","BMXBMI","EGFR_2021")
)

raw_xsets <- list(
    X0=c("RIDAGEYR","RIAGENDR","RIDRETH_FROZEN"),
    X1=c("RIDAGEYR","RIAGENDR","RIDRETH_FROZEN","INDFMPIR","EDUC3","SMOKING3"),
    X2=c("RIDAGEYR","RIAGENDR","RIDRETH_FROZEN","INDFMPIR","EDUC3","SMOKING3","BMXBMI"),
    X3=c("RIDAGEYR","RIAGENDR","RIDRETH_FROZEN","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
)

rows <- list()
design_rows <- list()

for (nm in names(xsets)) {

    need <- unique(c(
        "SOMATIC_SCORE","A","G_HBA1C","AG_HBA1C",
        "WTPH2YR","SDMVPSU","SDMVSTRA",
        raw_xsets[[nm]]
    ))

    q <- d[complete.cases(d[,need]), ]

    des <- svydesign(
        ids=~SDMVPSU,
        strata=~SDMVSTRA,
        weights=~WTPH2YR,
        nest=TRUE,
        data=q
    )

    rhs <- paste(xsets[[nm]], collapse=" + ")

    form_add <- as.formula(
        paste("SOMATIC_SCORE ~ A + G_HBA1C +", rhs)
    )

    fit <- svyglm(form_add, design=des, family=gaussian())

    mm <- model.matrix(fit)
    rank <- qr(mm)$rank

    n_strata <- length(unique(q$SDMVSTRA))
    n_psu <- nrow(unique(q[,c("SDMVSTRA","SDMVPSU")]))

    design_rows[[length(design_rows)+1]] <- data.frame(
        model=nm,
        n=nrow(q),
        n_strata=n_strata,
        n_psu=n_psu,
        survey_degf=degf(des),
        model_matrix_columns=ncol(mm),
        model_rank=rank,
        residual_design_df=degf(des) - (rank - 1),
        stringsAsFactors=FALSE
    )

    sm <- summary(fit)$coefficients

    for (term in c("A","G_HBA1C")) {

        est <- sm[term,"Estimate"]
        se <- sm[term,"Std. Error"]

        # Compute normal-based interval only as a descriptive diagnostic
        # when survey residual df are exhausted; do NOT treat as primary inference.
        lo_norm <- est - 1.96*se
        hi_norm <- est + 1.96*se

        pval <- NA_real_
        ci_lo <- NA_real_
        ci_hi <- NA_real_

        suppressWarnings({
            cc <- try(confint(fit), silent=TRUE)
        })

        if (!inherits(cc, "try-error") && term %in% rownames(cc)) {
            ci_lo <- cc[term,1]
            ci_hi <- cc[term,2]
        }

        if ("Pr(>|t|)" %in% colnames(sm)) {
            pval <- sm[term,"Pr(>|t|)"]
        }

        rows[[length(rows)+1]] <- data.frame(
            model=nm,
            term=term,
            beta=est,
            se=se,
            ci_low=ci_lo,
            ci_high=ci_hi,
            p=pval,
            normal_diag_ci_low=lo_norm,
            normal_diag_ci_high=hi_norm,
            stringsAsFactors=FALSE
        )
    }
}

coef <- do.call(rbind, rows)
design <- do.call(rbind, design_rows)

write.csv(
    coef,
    file.path(results_dir, "33_2123_x_ladder_coefficients.csv"),
    row.names=FALSE
)
write.csv(
    design,
    file.path(results_dir, "33_2123_design_diagnostics.csv"),
    row.names=FALSE
)

cat("\n2021-2023 DESIGN DIAGNOSTICS\n")
cat("============================\n")
print(design)

cat("\n2021-2023 FROZEN X-LADDER\n")
cat("=========================\n")
print(coef)

cat("\nINTERPRETATION RULE\n")
cat("===================\n")
cat("This ladder was prespecified before validation.\n")
cat("If A changes sign before the model loses inferential degrees of freedom, that is substantive evidence of non-transfer.\n")
cat("If sign instability appears only after design df collapse, treat X3 inference as technically unresolved rather than rescued.\n")
cat("Normal-based diagnostic intervals are descriptive only and are NOT a replacement for survey-design inference.\n")
'''

r_path = AUDIT / "33_2123_transfer_diagnostic.R"
r_path.write_text(r_code, encoding="utf-8")

proc = subprocess.run(
    [str(rscript), str(r_path), str(CSV_INPUT), str(RESULTS), str(RLIB)],
    capture_output=True,
    text=True
)

print(proc.stdout)

if proc.returncode != 0:
    print(proc.stderr[-4000:])
    raise SystemExit(proc.returncode)

print("SAVED")
print("-----")
print("Results/33_missingness_by_period.csv")
print("Results/33_distribution_comparison.csv")
print("Results/33_2123_category_counts.csv")
print("Results/33_2123_design_diagnostics.csv")
print("Results/33_2123_x_ladder_coefficients.csv")
