
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
