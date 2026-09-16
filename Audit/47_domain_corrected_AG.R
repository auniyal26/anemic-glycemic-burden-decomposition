
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

A_PCS <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G_PCS <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
X0 <- c("RIDAGEYR","RIAGENDR","RACE")

make_formula <- function(outcome, terms) as.formula(paste(outcome, "~", paste(terms, collapse=" + ")))

safe_test <- function(fit, terms, population, adjustment, outcome, label) {
  z <- tryCatch(regTermTest(fit, as.formula(paste("~", paste(terms, collapse=" + "))), method="Wald"), error=function(e) NULL)
  if (is.null(z)) return(data.frame(population=population, adjustment=adjustment, outcome=outcome, test=label,
                                    F=NA_real_, df_num=NA_real_, df_den=NA_real_, p=NA_real_,
                                    design_df=degf(fit$survey.design), inference_status="unavailable"))
  ddf <- as.numeric(z$ddf)
  status <- ifelse(is.na(ddf), "unavailable", ifelse(ddf <= 3, "design_limited", "available"))
  data.frame(population=population, adjustment=adjustment, outcome=outcome, test=label,
             F=as.numeric(z$Ftest), df_num=as.numeric(z$df), df_den=ddf, p=as.numeric(z$p),
             design_df=degf(fit$survey.design), inference_status=status)
}

extract_coef <- function(fit, population, adjustment, outcome, model, terms) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]
  out <- data.frame()
  for (term in terms) {
    if (!(term %in% rownames(sm))) next
    out <- rbind(out, data.frame(population=population, adjustment=adjustment, outcome=outcome, model=model,
      term=term, beta=sm[term,"Estimate"], se=sm[term,"Std. Error"], ci_low=ci[term,1], ci_high=ci[term,2],
      p=sm[term,pcol], stringsAsFactors=FALSE))
  }
  out
}

weighted_r2 <- function(fit, domain_data, outcome) {
  pred <- as.numeric(predict(fit, newdata=domain_data, type="response"))
  y <- domain_data[[outcome]]
  w <- domain_data$SURVEY_WT
  mu <- sum(w*y)/sum(w)
  sse <- sum(w*(y-pred)^2)
  sst <- sum(w*(y-mu)^2)
  if (sst <= 0) return(NA_real_)
  1 - sse/sst
}

run_period <- function(dat, population, adjustment, include_cycle) {
  # CRITICAL: all positive-weight records are in des_full.
  des_full <- svydesign(ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT, nest=TRUE, data=dat)
  covars <- if (adjustment=="X3") X3 else X0
  if (include_cycle) covars <- c(covars, "CYCLE")

  coef_rows <- data.frame(); test_rows <- data.frame(); r2_rows <- data.frame(); audit_rows <- data.frame()
  for (outcome in c("SOMATIC_SCORE","PHQ9_TOTAL","COGAFF_SUM")) {
    if (adjustment=="X3" && outcome=="SOMATIC_SCORE") { des <- subset(des_full, DOMAIN_SOMATIC_SCORE_X3==1); dom <- dat[dat$DOMAIN_SOMATIC_SCORE_X3==1,] }
    if (adjustment=="X3" && outcome=="PHQ9_TOTAL")   { des <- subset(des_full, DOMAIN_PHQ9_TOTAL_X3==1);   dom <- dat[dat$DOMAIN_PHQ9_TOTAL_X3==1,] }
    if (adjustment=="X3" && outcome=="COGAFF_SUM")  { des <- subset(des_full, DOMAIN_COGAFF_SUM_X3==1);  dom <- dat[dat$DOMAIN_COGAFF_SUM_X3==1,] }
    if (adjustment=="X0" && outcome=="SOMATIC_SCORE") { des <- subset(des_full, DOMAIN_SOMATIC_SCORE_X0==1); dom <- dat[dat$DOMAIN_SOMATIC_SCORE_X0==1,] }
    if (adjustment=="X0" && outcome=="PHQ9_TOTAL")   { des <- subset(des_full, DOMAIN_PHQ9_TOTAL_X0==1);   dom <- dat[dat$DOMAIN_PHQ9_TOTAL_X0==1,] }
    if (adjustment=="X0" && outcome=="COGAFF_SUM")  { des <- subset(des_full, DOMAIN_COGAFF_SUM_X0==1);  dom <- dat[dat$DOMAIN_COGAFF_SUM_X0==1,] }

    audit_rows <- rbind(audit_rows, data.frame(population=population, adjustment=adjustment, outcome=outcome,
      full_design_n=nrow(dat), domain_n=nrow(dom), full_design_df=degf(des_full), domain_design_df=degf(des)))

    fit_x <- svyglm(make_formula(outcome, covars), design=des, family=gaussian())
    fit_scalars <- svyglm(make_formula(outcome, c("A","G_HBA1C",covars)), design=des, family=gaussian())
    fit_Apcs_Gscalar <- svyglm(make_formula(outcome, c(A_PCS,"G_HBA1C",covars)), design=des, family=gaussian())
    fit_Ascalar_Gpcs <- svyglm(make_formula(outcome, c("A",G_PCS,covars)), design=des, family=gaussian())
    fit_pcs <- svyglm(make_formula(outcome, c(A_PCS,G_PCS,covars)), design=des, family=gaussian())
    fit_union <- svyglm(make_formula(outcome, c("A","G_HBA1C",A_PCS,G_PCS,covars)), design=des, family=gaussian())

    coef_rows <- rbind(coef_rows,
      extract_coef(fit_pcs,population,adjustment,outcome,"all_PCs",c(A_PCS,G_PCS)),
      extract_coef(fit_union,population,adjustment,outcome,"union",c("A","G_HBA1C",A_PCS,G_PCS)))

    test_rows <- rbind(test_rows,
      safe_test(fit_pcs,A_PCS,population,adjustment,outcome,"A_PCs_joint_given_G_PCs_X"),
      safe_test(fit_pcs,G_PCS,population,adjustment,outcome,"G_PCs_joint_given_A_PCs_X"),
      safe_test(fit_pcs,A_PCS[-1],population,adjustment,outcome,"A_PC2_3_extra_beyond_A_PC1_G_PCs_X"),
      safe_test(fit_pcs,G_PCS[-1],population,adjustment,outcome,"G_PC2_3_extra_beyond_G_PC1_A_PCs_X"),
      safe_test(fit_union,A_PCS,population,adjustment,outcome,"A_PCs_extra_beyond_scalar_A_scalar_G_G_PCs_X"),
      safe_test(fit_union,G_PCS,population,adjustment,outcome,"G_PCs_extra_beyond_scalar_G_scalar_A_A_PCs_X"),
      safe_test(fit_union,c(A_PCS,G_PCS),population,adjustment,outcome,"all_PCs_extra_beyond_scalar_A_scalar_G_X"),
      safe_test(fit_union,c("A","G_HBA1C"),population,adjustment,outcome,"both_scalars_extra_beyond_all_PCs_X"),
      safe_test(fit_union,"A",population,adjustment,outcome,"scalar_A_extra_beyond_all_PCs_scalar_G_X"),
      safe_test(fit_union,"G_HBA1C",population,adjustment,outcome,"scalar_G_extra_beyond_all_PCs_scalar_A_X"))

    fits <- list(X=fit_x, X_scalarA_scalarG=fit_scalars, X_Apcs_scalarG=fit_Apcs_Gscalar,
                 X_scalarA_Gpcs=fit_Ascalar_Gpcs, X_Apcs_Gpcs=fit_pcs, X_scalars_Apcs_Gpcs=fit_union)
    for (nm in names(fits)) {
      r2_rows <- rbind(r2_rows, data.frame(population=population,adjustment=adjustment,outcome=outcome,
        model=nm,weighted_R2=weighted_r2(fits[[nm]],dom,outcome),n=nrow(dom),
        design_df_full=degf(des_full),design_df_domain=degf(des),stringsAsFactors=FALSE))
    }
  }
  list(coef=coef_rows, tests=test_rows, r2=r2_rows, audit=audit_rows)
}

r1 <- run_period(d[d$PERIOD=="2005-2008",], "2005-2008", "X3", TRUE)
r2 <- run_period(d[d$PERIOD=="2009-2018",], "2009-2018", "X3", TRUE)
r3 <- run_period(d[d$PERIOD=="2021-2023",], "2021-2023", "X3", FALSE)
r4 <- run_period(d[d$PERIOD=="2021-2023",], "2021-2023", "X0", FALSE)

write.csv(rbind(r1$coef,r2$coef,r3$coef,r4$coef), file.path(results_dir,"47_domain_corrected_coefficients.csv"), row.names=FALSE)
write.csv(rbind(r1$tests,r2$tests,r3$tests,r4$tests), file.path(results_dir,"47_domain_corrected_block_tests.csv"), row.names=FALSE)
write.csv(rbind(r1$r2,r2$r2,r3$r2,r4$r2), file.path(results_dir,"47_domain_corrected_model_R2.csv"), row.names=FALSE)
write.csv(rbind(r1$audit,r2$audit,r3$audit,r4$audit), file.path(results_dir,"47_domain_corrected_design_audit.csv"), row.names=FALSE)
