
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
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])

A <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")
PCS <- c(A,G)
AXG <- c("AXG_A1_G1","AXG_A1_G2","AXG_A1_G3",
         "AXG_A2_G1","AXG_A2_G2","AXG_A2_G3",
         "AXG_A3_G1","AXG_A3_G2","AXG_A3_G3")
AGEINT <- c("A1_X_AGE10","A2_X_AGE10","A3_X_AGE10",
            "G1_X_AGE10","G2_X_AGE10","G3_X_AGE10")
SEXINT <- c("A1_X_FEMALE","A2_X_FEMALE","A3_X_FEMALE",
            "G1_X_FEMALE","G2_X_FEMALE","G3_X_FEMALE")
TIMEINT <- c("A1_X_TIME2","A2_X_TIME2","A3_X_TIME2",
             "G1_X_TIME2","G2_X_TIME2","G3_X_TIME2")
POSTINT <- c("A1_X_POST","A2_X_POST","A3_X_POST",
             "G1_X_POST","G2_X_POST","G3_X_POST")
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")

make_formula <- function(outcome, terms) {
  as.formula(paste(outcome, "~", paste(unique(terms), collapse=" + ")))
}

safe_test <- function(fit, terms, population, family, label) {
  z <- tryCatch(
    regTermTest(fit, as.formula(paste("~", paste(terms, collapse=" + "))), method="Wald"),
    error=function(e) NULL
  )
  if (is.null(z)) {
    return(data.frame(population=population, family=family, test=label,
      F=NA_real_, df_num=NA_real_, df_den=NA_real_, p=NA_real_,
      design_df=degf(fit$survey.design), inference_status="unavailable"))
  }
  ddf <- as.numeric(z$ddf)
  status <- ifelse(is.na(ddf), "unavailable", ifelse(ddf <= 3, "design_limited", "available"))
  data.frame(population=population, family=family, test=label,
    F=as.numeric(z$Ftest), df_num=as.numeric(z$df), df_den=ddf, p=as.numeric(z$p),
    design_df=degf(fit$survey.design), inference_status=status)
}

coef_rows <- data.frame()
extract_terms <- function(fit, terms, population, family, model) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]
  out <- data.frame()
  for (term in terms) {
    if (!(term %in% rownames(sm))) next
    out <- rbind(out, data.frame(population=population, family=family, model=model, term=term,
      beta=sm[term,"Estimate"], se=sm[term,"Std. Error"],
      ci_low=ci[term,1], ci_high=ci[term,2], p=sm[term,pcol], stringsAsFactors=FALSE))
  }
  out
}

run_period <- function(dat, population) {
  des_full <- svydesign(ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT,
                        nest=TRUE, data=dat)
  des <- subset(des_full, DOMAIN_SOMATIC_SCORE_X3==1)
  dom <- dat[dat$DOMAIN_SOMATIC_SCORE_X3==1,]
  covars <- c(X3, "CYCLE")

  base <- svyglm(make_formula("SOMATIC_SCORE", c(PCS,covars)), design=des, family=gaussian())
  fit_axg <- svyglm(make_formula("SOMATIC_SCORE", c(PCS,AXG,covars)), design=des, family=gaussian())
  fit_age <- svyglm(make_formula("SOMATIC_SCORE", c(PCS,"AGE10_C",AGEINT,covars)), design=des, family=gaussian())
  fit_sex <- svyglm(make_formula("SOMATIC_SCORE", c(PCS,"FEMALE",SEXINT,covars)), design=des, family=gaussian())

  tests <- rbind(
    safe_test(fit_axg, "AXG_A1_G1", population, "A_x_G", "PC1_x_PC1"),
    safe_test(fit_axg, AXG, population, "A_x_G", "all_9_AxG_interactions"),
    safe_test(fit_axg, AXG[-1], population, "A_x_G", "8_non_PC1xPC1_interactions"),
    safe_test(fit_age, AGEINT[1:3], population, "effect_modification", "A_components_x_age"),
    safe_test(fit_age, AGEINT[4:6], population, "effect_modification", "G_components_x_age"),
    safe_test(fit_age, AGEINT, population, "effect_modification", "all_components_x_age"),
    safe_test(fit_sex, SEXINT[1:3], population, "effect_modification", "A_components_x_sex"),
    safe_test(fit_sex, SEXINT[4:6], population, "effect_modification", "G_components_x_sex"),
    safe_test(fit_sex, SEXINT, population, "effect_modification", "all_components_x_sex")
  )

  coefs <- rbind(
    extract_terms(fit_axg, AXG, population, "A_x_G", "AxG_full"),
    extract_terms(fit_age, AGEINT, population, "effect_modification", "age_interactions"),
    extract_terms(fit_sex, SEXINT, population, "effect_modification", "sex_interactions")
  )

  audit <- data.frame(population=population,
    full_design_n=nrow(dat), analytic_domain_n=nrow(dom),
    full_design_df=degf(des_full), domain_design_df=degf(des),
    full_design_strata=length(unique(dat$STRATUM)),
    full_design_psu=length(unique(dat$PSU)))
  list(tests=tests, coefs=coefs, audit=audit)
}

r1 <- run_period(d[d$PERIOD=="2005-2008",], "2005-2008")
r2 <- run_period(d[d$PERIOD=="2009-2018",], "2009-2018")

# Pooled 2005-2018 time models with correct 7-cycle fasting weights.
t <- d[d$CYCLE %in% c("0506","0708","0910","1112","1314","1516","1718") &
       !is.na(d$WT_0518) & d$WT_0518 > 0,]
des_full_t <- svydesign(ids=~PSU, strata=~STRATUM, weights=~WT_0518,
                        nest=TRUE, data=t)
des_t <- subset(des_full_t, DOMAIN_SOMATIC_SCORE_X3==1)
dom_t <- t[t$DOMAIN_SOMATIC_SCORE_X3==1,]

# CYCLE absorbs arbitrary mean shifts by cycle; interaction blocks ask whether
# component slopes themselves drift.
base_t <- svyglm(make_formula("SOMATIC_SCORE", c(PCS,X3,"CYCLE")), design=des_t, family=gaussian())
trend_t <- svyglm(make_formula("SOMATIC_SCORE", c(PCS,X3,"CYCLE",TIMEINT)), design=des_t, family=gaussian())
post_t <- svyglm(make_formula("SOMATIC_SCORE", c(PCS,X3,"CYCLE",POSTINT)), design=des_t, family=gaussian())

time_tests <- rbind(
  safe_test(trend_t, TIMEINT[1:3], "2005-2018", "temporal_drift", "A_components_linear_cycle_drift"),
  safe_test(trend_t, TIMEINT[4:6], "2005-2018", "temporal_drift", "G_components_linear_cycle_drift"),
  safe_test(trend_t, TIMEINT, "2005-2018", "temporal_drift", "all_components_linear_cycle_drift"),
  safe_test(post_t, POSTINT[1:3], "2005-2018", "temporal_drift", "A_components_development_vs_replication_shift"),
  safe_test(post_t, POSTINT[4:6], "2005-2018", "temporal_drift", "G_components_development_vs_replication_shift"),
  safe_test(post_t, POSTINT, "2005-2018", "temporal_drift", "all_components_development_vs_replication_shift")
)

time_coefs <- rbind(
  extract_terms(trend_t, TIMEINT, "2005-2018", "temporal_drift", "linear_cycle_interactions"),
  extract_terms(post_t, POSTINT, "2005-2018", "temporal_drift", "development_replication_interactions")
)

time_audit <- data.frame(population="2005-2018",
  full_design_n=nrow(t), analytic_domain_n=nrow(dom_t),
  full_design_df=degf(des_full_t), domain_design_df=degf(des_t),
  full_design_strata=length(unique(t$STRATUM)), full_design_psu=length(unique(t$PSU)))

write.csv(rbind(r1$tests,r2$tests), file.path(results_dir,"49_interaction_block_tests.csv"), row.names=FALSE)
write.csv(rbind(r1$coefs,r2$coefs), file.path(results_dir,"49_interaction_coefficients.csv"), row.names=FALSE)
write.csv(time_tests, file.path(results_dir,"49_temporal_drift_tests.csv"), row.names=FALSE)
write.csv(time_coefs, file.path(results_dir,"49_temporal_drift_coefficients.csv"), row.names=FALSE)
write.csv(rbind(r1$audit,r2$audit,time_audit), file.path(results_dir,"49_interaction_temporal_design_audit.csv"), row.names=FALSE)
