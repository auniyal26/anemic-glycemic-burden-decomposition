from pathlib import Path
import shutil
import subprocess
import pandas as pd

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

RESULTS.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RLIB.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "25_survey_parity_input.csv"

print()
print("FORMAL A/G SEPARABILITY TEST")
print("============================")

if not INPUT.exists():
    raise FileNotFoundError(f"Missing {INPUT}")

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

weighted_r2 <- function(fit, dat, outcome) {
    pred <- as.numeric(predict(fit, newdata=dat, type="response"))
    y <- dat[[outcome]]
    w <- dat$WTMEC4YR
    ybar <- sum(w * y) / sum(w)
    sse <- sum(w * (y - pred)^2)
    sst <- sum(w * (y - ybar)^2)
    return(1 - sse / sst)
}

weighted_cor <- function(x, y, w) {
    mx <- sum(w*x)/sum(w)
    my <- sum(w*y)/sum(w)
    covxy <- sum(w*(x-mx)*(y-my))/sum(w)
    vx <- sum(w*(x-mx)^2)/sum(w)
    vy <- sum(w*(y-my)^2)/sum(w)
    covxy / sqrt(vx*vy)
}

extract_term <- function(fit, outcome, population, term, model_name) {
    sm <- summary(fit)$coefficients
    ci <- confint(fit)

    if (!(term %in% rownames(sm))) {
        return(NULL)
    }

    p_col <- ncol(sm)

    data.frame(
        population=population,
        outcome=outcome,
        model=model_name,
        term=term,
        beta=sm[term, "Estimate"],
        se=sm[term, "Std. Error"],
        ci_low=ci[term, 1],
        ci_high=ci[term, 2],
        p=sm[term, p_col],
        stringsAsFactors=FALSE
    )
}

run_population <- function(dat, population) {

    if (population == "Pooled") {
        X <- paste(
            "RIDAGEYR",
            "factor(RIAGENDR)",
            "factor(RIDRETH1)",
            "INDFMPIR",
            "factor(EDUC3)",
            "factor(SMOKING3)",
            "BMXBMI",
            "EGFR_2021",
            "factor(CYCLE)",
            sep=" + "
        )
    } else {
        X <- paste(
            "RIDAGEYR",
            "factor(RIAGENDR)",
            "factor(RIDRETH1)",
            "INDFMPIR",
            "factor(EDUC3)",
            "factor(SMOKING3)",
            "BMXBMI",
            "EGFR_2021",
            sep=" + "
        )
    }

    des <- svydesign(
        ids=~PSU,
        strata=~STRATUM,
        weights=~WTMEC4YR,
        nest=TRUE,
        data=dat
    )

    ladder_rows <- list()
    contrib_rows <- list()
    coef_rows <- list()

    for (outcome in c("PHQ9_TOTAL", "SOMATIC_SCORE")) {

        f0   <- as.formula(paste(outcome, "~", X))
        fA   <- as.formula(paste(outcome, "~ A +", X))
        fG   <- as.formula(paste(outcome, "~ G_HBA1C +", X))
        fAG  <- as.formula(paste(outcome, "~ A + G_HBA1C +", X))
        fAGI <- as.formula(paste(outcome, "~ A + G_HBA1C + AG_HBA1C +", X))

        m0   <- svyglm(f0, design=des, family=gaussian())
        mA   <- svyglm(fA, design=des, family=gaussian())
        mG   <- svyglm(fG, design=des, family=gaussian())
        mAG  <- svyglm(fAG, design=des, family=gaussian())
        mAGI <- svyglm(fAGI, design=des, family=gaussian())

        r0   <- weighted_r2(m0, dat, outcome)
        rA   <- weighted_r2(mA, dat, outcome)
        rG   <- weighted_r2(mG, dat, outcome)
        rAG  <- weighted_r2(mAG, dat, outcome)
        rAGI <- weighted_r2(mAGI, dat, outcome)

        ladder_rows[[length(ladder_rows)+1]] <- data.frame(
            population=population,
            outcome=outcome,
            model=c("X", "X+A", "X+G", "X+A+G", "X+A+G+AxG"),
            weighted_R2=c(r0, rA, rG, rAG, rAGI),
            n=nrow(dat),
            design_df=degf(des),
            stringsAsFactors=FALSE
        )

        unique_A <- rAG - rG
        unique_G <- rAG - rA
        total_AG <- rAG - r0
        shared_AG <- total_AG - unique_A - unique_G
        interaction_extra <- rAGI - rAG

        corr_AG <- weighted_cor(dat$A, dat$G_HBA1C, dat$WTMEC4YR)

        contrib_rows[[length(contrib_rows)+1]] <- data.frame(
            population=population,
            outcome=outcome,
            baseline_X_R2=r0,
            A_increment_over_X=rA-r0,
            G_increment_over_X=rG-r0,
            joint_AG_increment_over_X=total_AG,
            unique_A_beyond_G_X=unique_A,
            unique_G_beyond_A_X=unique_G,
            shared_or_overlap=shared_AG,
            AxG_extra_beyond_additive=interaction_extra,
            weighted_corr_A_G=corr_AG,
            stringsAsFactors=FALSE
        )

        coef_rows[[length(coef_rows)+1]] <- extract_term(
            mAG, outcome, population, "A", "Additive_A+G+X"
        )
        coef_rows[[length(coef_rows)+1]] <- extract_term(
            mAG, outcome, population, "G_HBA1C", "Additive_A+G+X"
        )

        coef_rows[[length(coef_rows)+1]] <- extract_term(
            mAGI, outcome, population, "A", "Interaction_A+G+AxG+X"
        )
        coef_rows[[length(coef_rows)+1]] <- extract_term(
            mAGI, outcome, population, "G_HBA1C", "Interaction_A+G+AxG+X"
        )
        coef_rows[[length(coef_rows)+1]] <- extract_term(
            mAGI, outcome, population, "AG_HBA1C", "Interaction_A+G+AxG+X"
        )
    }

    list(
        ladder=do.call(rbind, ladder_rows),
        contributions=do.call(rbind, contrib_rows),
        coefficients=do.call(rbind, coef_rows)
    )
}

all_ladder <- list()
all_contrib <- list()
all_coef <- list()

populations <- list(
    "Pooled" = d,
    "0506" = d[d$CYCLE == "506", ],
    "0708" = d[d$CYCLE == "708", ]
)

for (nm in names(populations)) {
    cat("Running", nm, "...\n")
    res <- run_population(populations[[nm]], nm)
    all_ladder[[length(all_ladder)+1]] <- res$ladder
    all_contrib[[length(all_contrib)+1]] <- res$contributions
    all_coef[[length(all_coef)+1]] <- res$coefficients
}

ladder <- do.call(rbind, all_ladder)
contrib <- do.call(rbind, all_contrib)
coefs <- do.call(rbind, all_coef)

write.csv(
    ladder,
    file.path(results_dir, "26_model_ladder_weighted_R2.csv"),
    row.names=FALSE
)

write.csv(
    contrib,
    file.path(results_dir, "26_unique_contribution_decomposition.csv"),
    row.names=FALSE
)

write.csv(
    coefs,
    file.path(results_dir, "26_design_based_coefficients.csv"),
    row.names=FALSE
)

ps <- coefs[
    coefs$population=="Pooled" &
    coefs$outcome=="SOMATIC_SCORE" &
    coefs$model=="Additive_A+G+X", ]

pa <- ps[ps$term=="A", ]
pg <- ps[ps$term=="G_HBA1C", ]

pc <- contrib[
    contrib$population=="Pooled" &
    contrib$outcome=="SOMATIC_SCORE", ]

cycle_add <- coefs[
    coefs$population %in% c("0506","0708") &
    coefs$outcome=="SOMATIC_SCORE" &
    coefs$model=="Additive_A+G+X", ]

a_cycle <- cycle_add[cycle_add$term=="A", ]
g_cycle <- cycle_add[cycle_add$term=="G_HBA1C", ]

a_same_direction <- all(a_cycle$beta > 0)
g_same_direction <- all(g_cycle$beta > 0)

cat("\nSEPARABILITY VERDICT\n")
cat("====================\n")

cat(sprintf(
    "Pooled A: beta=%.4f, 95%% CI [%.4f, %.4f], p=%.5f\n",
    pa$beta, pa$ci_low, pa$ci_high, pa$p
))

cat(sprintf(
    "Pooled G: beta=%.4f, 95%% CI [%.4f, %.4f], p=%.5f\n",
    pg$beta, pg$ci_low, pg$ci_high, pg$p
))

cat(sprintf(
    "Unique weighted variance beyond G+X from A: %.6f\n",
    pc$unique_A_beyond_G_X
))

cat(sprintf(
    "Unique weighted variance beyond A+X from G: %.6f\n",
    pc$unique_G_beyond_A_X
))

cat(sprintf(
    "Weighted A-G correlation: %.4f\n",
    pc$weighted_corr_A_G
))

if (pa$p < 0.05 && pg$p < 0.05 &&
    pc$unique_A_beyond_G_X > 0 &&
    pc$unique_G_beyond_A_X > 0) {
    cat("PASS  A and G each add statistically distinct information in the pooled somatic model.\n")
} else {
    cat("FAIL/UNCERTAIN  Pooled separability criterion not fully met.\n")
}

if (a_same_direction && g_same_direction) {
    cat("PASS  A and G directions are consistent across both independent cycles.\n")
} else {
    cat("WARNING  At least one exposure changes direction across cycles.\n")
}

pi <- coefs[
    coefs$population=="Pooled" &
    coefs$outcome=="SOMATIC_SCORE" &
    coefs$model=="Interaction_A+G+AxG+X" &
    coefs$term=="AG_HBA1C", ]

cat(sprintf(
    "A×G pooled: beta=%.4f, 95%% CI [%.4f, %.4f], p=%.5f; extra weighted R2=%.6f\n",
    pi$beta, pi$ci_low, pi$ci_high, pi$p,
    pc$AxG_extra_beyond_additive
))

cat("NOTE  A×G remains secondary unless it also survives representation and cycle checks.\n")
cat("NOTE  Weighted R2 decomposition is descriptive; design-based coefficient tests provide inference.\n")
cat("NOTE  This establishes separable association, not causation.\n")
'''

r_path = AUDIT / "26_formal_separability_test.R"
r_path.write_text(r_code, encoding="utf-8")

proc = subprocess.run(
    [str(rscript), str(r_path), str(INPUT), str(RESULTS), str(RLIB)],
    capture_output=True,
    text=True
)

print(proc.stdout)
if proc.returncode != 0:
    print(proc.stderr[-4000:])
    raise SystemExit(proc.returncode)

coefs = pd.read_csv(RESULTS / "26_design_based_coefficients.csv")
contrib = pd.read_csv(RESULTS / "26_unique_contribution_decomposition.csv")

som = coefs[
    (coefs["population"] == "Pooled") &
    (coefs["outcome"] == "SOMATIC_SCORE") &
    (coefs["model"] == "Additive_A+G+X")
]

c = contrib[
    (contrib["population"] == "Pooled") &
    (contrib["outcome"] == "SOMATIC_SCORE")
].iloc[0]

print("PRESENTATION-READY NUMBERS IF AUDIT PASSES")
print("-----------------------------------------")
for _, r in som.iterrows():
    label = "A" if r["term"] == "A" else "G"
    print(
        f"{label}: beta={r['beta']:.4f}, "
        f"95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}], "
        f"p={r['p']:.5f}"
    )

print(f"Unique A weighted ΔR²: {c['unique_A_beyond_G_X']:.6f}")
print(f"Unique G weighted ΔR²: {c['unique_G_beyond_A_X']:.6f}")
print(f"Shared/overlap weighted R²: {c['shared_or_overlap']:.6f}")
print(f"A×G extra weighted ΔR²: {c['AxG_extra_beyond_additive']:.6f}")
print(f"Weighted corr(A,G): {c['weighted_corr_A_G']:.4f}")
print()
print("Saved:")
print("  Results/26_model_ladder_weighted_R2.csv")
print("  Results/26_unique_contribution_decomposition.csv")
print("  Results/26_design_based_coefficients.csv")
