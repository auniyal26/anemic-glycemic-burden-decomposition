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
CSV_INPUT = RESULTS / "32_frozen_transfer_input.csv"

RESULTS.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RLIB.mkdir(parents=True, exist_ok=True)

print()
print("FROZEN TEMPORAL TRANSFER TEST")
print("============================")

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_parquet(INPUT)

X3 = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN",
    "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"
]

required = [
    "SEQN", "CYCLE", "SOMATIC_SCORE", "PHQ9_TOTAL",
    "A", "G_HBA1C", "AG_HBA1C",
    "SDMVPSU", "SDMVSTRA",
    "WEIGHT_0918_POOLED", "WEIGHT_2123",
    "STRATUM_TRANSFER", "PSU_TRANSFER"
] + X3

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

# Keep rows needed for either validation period.
df.to_csv(CSV_INPUT, index=False)

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

X3 <- c(
    "RIDAGEYR",
    "factor(RIAGENDR)",
    "factor(RIDRETH_FROZEN)",
    "INDFMPIR",
    "factor(EDUC3)",
    "factor(SMOKING3)",
    "BMXBMI",
    "EGFR_2021"
)

weighted_r2 <- function(fit, dat, outcome, weight_col) {
    pred <- as.numeric(predict(fit, newdata=dat, type="response"))
    y <- dat[[outcome]]
    w <- dat[[weight_col]]
    ybar <- sum(w*y)/sum(w)
    sse <- sum(w*(y-pred)^2)
    sst <- sum(w*(y-ybar)^2)
    1 - sse/sst
}

extract_core <- function(fit, population, outcome, model_name) {
    sm <- summary(fit)$coefficients
    ci <- confint(fit)
    terms <- intersect(c("A", "G_HBA1C", "AG_HBA1C"), rownames(sm))
    out <- data.frame()

    for (term in terms) {
        out <- rbind(
            out,
            data.frame(
                population=population,
                outcome=outcome,
                model=model_name,
                term=term,
                beta=sm[term, "Estimate"],
                se=sm[term, "Std. Error"],
                ci_low=ci[term,1],
                ci_high=ci[term,2],
                p=sm[term,ncol(sm)],
                stringsAsFactors=FALSE
            )
        )
    }
    out
}

run_period <- function(dat, population, weight_col, pooled_cycles=FALSE) {

    # Complete case after period selection, exactly using frozen X3.
    needed <- c(
        "SOMATIC_SCORE", "PHQ9_TOTAL",
        "A", "G_HBA1C", "AG_HBA1C",
        "RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN",
        "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021",
        weight_col, "SDMVPSU", "SDMVSTRA",
        "STRATUM_TRANSFER", "PSU_TRANSFER"
    )
    dat <- dat[complete.cases(dat[,needed]), ]

    if (pooled_cycles) {
        des <- svydesign(
            ids=~PSU_TRANSFER,
            strata=~STRATUM_TRANSFER,
            weights=as.formula(paste0("~", weight_col)),
            nest=TRUE,
            data=dat
        )
        x_rhs <- paste(c(X3, "factor(CYCLE)"), collapse=" + ")
    } else {
        des <- svydesign(
            ids=~SDMVPSU,
            strata=~SDMVSTRA,
            weights=as.formula(paste0("~", weight_col)),
            nest=TRUE,
            data=dat
        )
        x_rhs <- paste(X3, collapse=" + ")
    }

    coef_rows <- list()
    incr_rows <- list()
    ladder_rows <- list()

    for (outcome in c("SOMATIC_SCORE", "PHQ9_TOTAL")) {

        fX   <- as.formula(paste(outcome, "~", x_rhs))
        fA   <- as.formula(paste(outcome, "~ A +", x_rhs))
        fG   <- as.formula(paste(outcome, "~ G_HBA1C +", x_rhs))
        fAG  <- as.formula(paste(outcome, "~ A + G_HBA1C +", x_rhs))
        fAGI <- as.formula(paste(outcome, "~ A + G_HBA1C + AG_HBA1C +", x_rhs))

        mX   <- svyglm(fX, design=des, family=gaussian())
        mA   <- svyglm(fA, design=des, family=gaussian())
        mG   <- svyglm(fG, design=des, family=gaussian())
        mAG  <- svyglm(fAG, design=des, family=gaussian())
        mAGI <- svyglm(fAGI, design=des, family=gaussian())

        coef_rows[[length(coef_rows)+1]] <- extract_core(
            mAG, population, outcome, "Additive_A+G+X"
        )
        coef_rows[[length(coef_rows)+1]] <- extract_core(
            mAGI, population, outcome, "Interaction_A+G+AxG+X"
        )

        rX   <- weighted_r2(mX, dat, outcome, weight_col)
        rA   <- weighted_r2(mA, dat, outcome, weight_col)
        rG   <- weighted_r2(mG, dat, outcome, weight_col)
        rAG  <- weighted_r2(mAG, dat, outcome, weight_col)
        rAGI <- weighted_r2(mAGI, dat, outcome, weight_col)

        tA <- regTermTest(mAG, ~A, method="Wald")
        tG <- regTermTest(mAG, ~G_HBA1C, method="Wald")
        tI <- regTermTest(mAGI, ~AG_HBA1C, method="Wald")

        incr_rows[[length(incr_rows)+1]] <- data.frame(
            population=population,
            outcome=outcome,
            test=c(
                "Add_A_to_G_plus_X",
                "Add_G_to_A_plus_X",
                "Add_AxG_to_A_plus_G_plus_X"
            ),
            F=c(
                as.numeric(tA$Ftest),
                as.numeric(tG$Ftest),
                as.numeric(tI$Ftest)
            ),
            df_num=c(
                as.numeric(tA$df),
                as.numeric(tG$df),
                as.numeric(tI$df)
            ),
            df_den=c(
                as.numeric(tA$ddf),
                as.numeric(tG$ddf),
                as.numeric(tI$ddf)
            ),
            p=c(tA$p, tG$p, tI$p),
            weighted_delta_R2=c(
                rAG-rG,
                rAG-rA,
                rAGI-rAG
            ),
            stringsAsFactors=FALSE
        )

        ladder_rows[[length(ladder_rows)+1]] <- data.frame(
            population=population,
            outcome=outcome,
            model=c("X", "X+A", "X+G", "X+A+G", "X+A+G+AxG"),
            weighted_R2=c(rX, rA, rG, rAG, rAGI),
            n=nrow(dat),
            design_df=degf(des),
            stringsAsFactors=FALSE
        )
    }

    list(
        coef=do.call(rbind, coef_rows),
        incr=do.call(rbind, incr_rows),
        ladder=do.call(rbind, ladder_rows),
        n=nrow(dat),
        design_df=degf(des)
    )
}

# Frozen temporal replication: 2009-2018 pooled.
d0918 <- d[d$CYCLE %in% c("0910","1112","1314","1516","1718"), ]
r0918 <- run_period(
    d0918,
    "2009-2018",
    "WEIGHT_0918_POOLED",
    pooled_cycles=TRUE
)

# Untouched modern holdout.
d2123 <- d[d$CYCLE=="2123", ]
r2123 <- run_period(
    d2123,
    "2021-2023",
    "WEIGHT_2123",
    pooled_cycles=FALSE
)

coefs <- rbind(r0918$coef, r2123$coef)
incr  <- rbind(r0918$incr, r2123$incr)
ladder <- rbind(r0918$ladder, r2123$ladder)

write.csv(
    coefs,
    file.path(results_dir, "32_transfer_design_based_coefficients.csv"),
    row.names=FALSE
)
write.csv(
    incr,
    file.path(results_dir, "32_transfer_incremental_tests.csv"),
    row.names=FALSE
)
write.csv(
    ladder,
    file.path(results_dir, "32_transfer_model_ladder.csv"),
    row.names=FALSE
)

cat("\nFROZEN PRIMARY OUTCOME: SOMATIC SCORE\n")
cat("====================================\n")

for (pop in c("2009-2018","2021-2023")) {

    s <- coefs[
        coefs$population==pop &
        coefs$outcome=="SOMATIC_SCORE" &
        coefs$model=="Additive_A+G+X", ]

    ia <- incr[
        incr$population==pop &
        incr$outcome=="SOMATIC_SCORE" &
        incr$test=="Add_A_to_G_plus_X", ]

    ig <- incr[
        incr$population==pop &
        incr$outcome=="SOMATIC_SCORE" &
        incr$test=="Add_G_to_A_plus_X", ]

    ii <- incr[
        incr$population==pop &
        incr$outcome=="SOMATIC_SCORE" &
        incr$test=="Add_AxG_to_A_plus_G_plus_X", ]

    a <- s[s$term=="A", ]
    g <- s[s$term=="G_HBA1C", ]

    cat("\n", pop, "\n", sep="")
    cat(sprintf(
        "A: beta=%.4f, 95%% CI [%.4f, %.4f], p=%.5f, incremental p=%.5f, dR2=%.6f\n",
        a$beta, a$ci_low, a$ci_high, a$p, ia$p, ia$weighted_delta_R2
    ))
    cat(sprintf(
        "G: beta=%.4f, 95%% CI [%.4f, %.4f], p=%.5f, incremental p=%.5f, dR2=%.6f\n",
        g$beta, g$ci_low, g$ci_high, g$p, ig$p, ig$weighted_delta_R2
    ))
    cat(sprintf(
        "AxG incremental: p=%.5f, dR2=%.6f\n",
        ii$p, ii$weighted_delta_R2
    ))
}

# Frozen replication criteria.
get_add <- function(pop, term) {
    coefs[
        coefs$population==pop &
        coefs$outcome=="SOMATIC_SCORE" &
        coefs$model=="Additive_A+G+X" &
        coefs$term==term, ]
}

get_inc <- function(pop, test) {
    incr[
        incr$population==pop &
        incr$outcome=="SOMATIC_SCORE" &
        incr$test==test, ]
}

a0918 <- get_add("2009-2018","A")
a2123 <- get_add("2021-2023","A")
g0918 <- get_add("2009-2018","G_HBA1C")
g2123 <- get_add("2021-2023","G_HBA1C")

gi0918 <- get_inc("2009-2018","Add_G_to_A_plus_X")
gi2123 <- get_inc("2021-2023","Add_G_to_A_plus_X")

A_minimum <- (a0918$beta > 0) && (a2123$beta > 0)
G_minimum <- (g0918$beta > 0) && (g2123$beta > 0) &&
             (gi0918$weighted_delta_R2 > 0) &&
             (gi2123$weighted_delta_R2 > 0)

A_strong_0918 <- a0918$ci_low > 0
A_strong_2123 <- a2123$ci_low > 0
G_strong_0918 <- g0918$ci_low > 0
G_strong_2123 <- g2123$ci_low > 0

cat("\nTRANSFER VERDICT\n")
cat("================\n")

if (A_minimum) {
    cat("PASS  A meets the frozen minimum temporal-transfer criterion.\n")
} else {
    cat("FAIL  A does not meet the frozen minimum temporal-transfer criterion.\n")
}

if (G_minimum) {
    cat("PASS  G meets the frozen minimum temporal-transfer criterion.\n")
} else {
    cat("FAIL  G does not meet the frozen minimum temporal-transfer criterion.\n")
}

cat(sprintf(
    "A strong-CI support: 2009-2018=%s; 2021-2023=%s\n",
    A_strong_0918, A_strong_2123
))
cat(sprintf(
    "G strong-CI support: 2009-2018=%s; 2021-2023=%s\n",
    G_strong_0918, G_strong_2123
))

if (A_minimum && G_minimum) {
    cat("OVERALL  The frozen A/G association pattern transfers forward in time at the prespecified minimum level.\n")
} else {
    cat("OVERALL  The full frozen A/G transfer pattern does not meet the prespecified minimum criterion.\n")
}

cat("NOTE  This is temporal population replication, not causal or clinical proof.\n")
'''

r_path = AUDIT / "32_frozen_temporal_transfer_test.R"
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
print("Results/32_transfer_design_based_coefficients.csv")
print("Results/32_transfer_incremental_tests.csv")
print("Results/32_transfer_model_ladder.csv")
