
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
    PHQ9_TOTAL ~ A + G_HBA1C + AG_HBA1C + RIDAGEYR + factor(RIAGENDR) + factor(RIDRETH1) + INDFMPIR + factor(EDUC3) + factor(SMOKING3) + BMXBMI + EGFR_2021 + factor(CYCLE),
    design=des,
    family=gaussian()
)

fit_somatic <- svyglm(
    SOMATIC_SCORE ~ A + G_HBA1C + AG_HBA1C + RIDAGEYR + factor(RIAGENDR) + factor(RIDRETH1) + INDFMPIR + factor(EDUC3) + factor(SMOKING3) + BMXBMI + EGFR_2021 + factor(CYCLE),
    design=des,
    family=gaussian()
)

extract_core <- function(fit, outcome) {
    sm <- summary(fit)$coefficients
    ci <- confint(fit)
    terms <- c("A","G_HBA1C","AG_HBA1C")
    out <- data.frame()
    for (term in terms) {
        out <- rbind(out, data.frame(
            outcome=outcome,
            term=term,
            beta=sm[term,"Estimate"],
            se=sm[term,"Std. Error"],
            p=sm[term,"Pr(>|t|)"],
            ci_low=ci[term,1],
            ci_high=ci[term,2]
        ))
    }
    out
}

res <- rbind(
    extract_core(fit_total, "PHQ9_TOTAL"),
    extract_core(fit_somatic, "SOMATIC_SCORE")
)

write.csv(
    res,
    file.path(results_dir, "25_r_survey_parity_results.csv"),
    row.names=FALSE
)
