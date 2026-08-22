
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
    ybar <- sum(w*y)/sum(w)
    sse <- sum(w*(y-pred)^2)
    sst <- sum(w*(y-ybar)^2)
    1 - sse/sst
}

run_test <- function(dat, population, outcome) {

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

    fX   <- as.formula(paste(outcome, "~", X))
    fA   <- as.formula(paste(outcome, "~ A +", X))
    fG   <- as.formula(paste(outcome, "~ G_HBA1C +", X))
    fAG  <- as.formula(paste(outcome, "~ A + G_HBA1C +", X))
    fAGI <- as.formula(paste(outcome, "~ A + G_HBA1C + AG_HBA1C +", X))

    mX   <- svyglm(fX, design=des, family=gaussian())
    mA   <- svyglm(fA, design=des, family=gaussian())
    mG   <- svyglm(fG, design=des, family=gaussian())
    mAG  <- svyglm(fAG, design=des, family=gaussian())
    mAGI <- svyglm(fAGI, design=des, family=gaussian())

    # Design-based Wald tests of incremental terms in the larger model.
    test_A  <- regTermTest(mAG,  ~A, method="Wald")
    test_G  <- regTermTest(mAG,  ~G_HBA1C, method="Wald")
    test_AG <- regTermTest(mAGI, ~AG_HBA1C, method="Wald")

    rX   <- weighted_r2(mX, dat, outcome)
    rA   <- weighted_r2(mA, dat, outcome)
    rG   <- weighted_r2(mG, dat, outcome)
    rAG  <- weighted_r2(mAG, dat, outcome)
    rAGI <- weighted_r2(mAGI, dat, outcome)

    data.frame(
        population=population,
        outcome=outcome,
        test=c(
            "Add_A_to_G_plus_X",
            "Add_G_to_A_plus_X",
            "Add_AxG_to_A_plus_G_plus_X"
        ),
        F=c(
            as.numeric(test_A$Ftest),
            as.numeric(test_G$Ftest),
            as.numeric(test_AG$Ftest)
        ),
        df_num=c(
            as.numeric(test_A$df),
            as.numeric(test_G$df),
            as.numeric(test_AG$df)
        ),
        df_den=c(
            as.numeric(test_A$ddf),
            as.numeric(test_G$ddf),
            as.numeric(test_AG$ddf)
        ),
        p=c(
            test_A$p,
            test_G$p,
            test_AG$p
        ),
        weighted_delta_R2=c(
            rAG-rG,
            rAG-rA,
            rAGI-rAG
        ),
        stringsAsFactors=FALSE
    )
}

all <- list()

populations <- list(
    "Pooled" = d,
    "0506" = d[d$CYCLE=="506", ],
    "0708" = d[d$CYCLE=="708", ]
)

for (population in names(populations)) {
    cat("Running", population, "...\n")
    dat <- populations[[population]]

    for (outcome in c("PHQ9_TOTAL", "SOMATIC_SCORE")) {
        all[[length(all)+1]] <- run_test(dat, population, outcome)
    }
}

res <- do.call(rbind, all)

write.csv(
    res,
    file.path(results_dir, "27_incremental_information_tests.csv"),
    row.names=FALSE
)

cat("\nPOOLED SOMATIC VERDICT\n")
cat("======================\n")

ps <- res[
    res$population=="Pooled" &
    res$outcome=="SOMATIC_SCORE", ]

for (i in seq_len(nrow(ps))) {
    cat(sprintf(
        "%s: F=%.4f, df=%g/%g, p=%.5f, weighted ΔR²=%.6f\n",
        ps$test[i],
        ps$F[i],
        ps$df_num[i],
        ps$df_den[i],
        ps$p[i],
        ps$weighted_delta_R2[i]
    ))
}

pA <- ps$p[ps$test=="Add_A_to_G_plus_X"]
pG <- ps$p[ps$test=="Add_G_to_A_plus_X"]
pI <- ps$p[ps$test=="Add_AxG_to_A_plus_G_plus_X"]

if (pA < 0.05 && pG < 0.05) {
    cat("PASS  Both A and G add statistically significant incremental information beyond each other + X.\n")
} else if (pA < 0.05 && pG < 0.10) {
    cat("PARTIAL  A is clearly incremental; G is borderline incremental in the pooled somatic model.\n")
} else {
    cat("UNCERTAIN  Full two-way separability is not established by the pooled somatic survey model.\n")
}

if (pI < 0.05) {
    cat("SIGNAL  A×G adds pooled incremental information, but replication/sensitivity still governs interpretation.\n")
} else {
    cat("NO CLEAR SUPPORT  A×G does not add significant pooled incremental information.\n")
}

cat("NOTE  These are design-based Wald tests under the NHANES complex survey design.\n")
cat("NOTE  This tests incremental association, not causation.\n")
