
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]; results_dir <- args[2]; user_lib <- args[3]
dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib,.libPaths()))
if (!requireNamespace("survey", quietly=TRUE)) install.packages("survey",repos="https://cloud.r-project.org",lib=user_lib)
library(survey); library(splines)
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])
A <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ"); G3 <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ"); RAW7 <- c("LBXHGB_DISC_Z","LBXRBCSI_DISC_Z","LBXMCVSI_DISC_Z","LBXRDW_DISC_Z","LBXGH_DISC_Z","LBXGLU_DISC_Z","LOG_IN_DISC_Z"); G2A <- c("G2A_PC1_FZ","G2A_PC2_FZ"); G2B <- c("G2B_PC1_FZ","G2B_PC2_FZ")
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
make_formula <- function(y,x) as.formula(paste(y,"~",paste(x,collapse=" + ")))

safe_test <- function(fit, terms, population, analysis, label) {
  z <- tryCatch(regTermTest(fit, as.formula(paste("~",paste(terms,collapse=" + "))), method="Wald"), error=function(e) NULL)
  if (is.null(z)) return(data.frame(population=population,analysis=analysis,test=label,F=NA,df_num=NA,df_den=NA,p=NA,design_df=degf(fit$survey.design)))
  data.frame(population=population,analysis=analysis,test=label,F=as.numeric(z$Ftest),df_num=as.numeric(z$df),df_den=as.numeric(z$ddf),p=as.numeric(z$p),design_df=degf(fit$survey.design))
}

wr2 <- function(fit, dat, outcome="SOMATIC_SCORE") {
  pr <- as.numeric(predict(fit,newdata=dat,type="response")); y <- dat[[outcome]]; w <- dat$SURVEY_WT
  mu <- sum(w*y)/sum(w); sse <- sum(w*(y-pr)^2); sst <- sum(w*(y-mu)^2)
  if (sst<=0) return(NA_real_); 1-sse/sst
}

comparison_rows <- data.frame(); test_rows <- data.frame(); sensitivity_rows <- data.frame(); qp_rows <- data.frame()

for (period in c("2005-2008","2009-2018","2021-2023")) {
  dat <- d[d$PERIOD==period,]
  des_full <- svydesign(ids=~PSU,strata=~STRATUM,weights=~SURVEY_WT,nest=TRUE,data=dat)

  # Strict same-sample comparator audit.
  des <- subset(des_full, DOMAIN_COMPARE_X3==1); dom <- dat[dat$DOMAIN_COMPARE_X3==1,]
  cyc <- if (period=="2021-2023") character(0) else "CYCLE"
  X <- c(X3,cyc)

  fits <- list(
    threshold_scalars=svyglm(make_formula("SOMATIC_SCORE",c("A","G_HBA1C",X)),design=des,family=gaussian()),
    continuous_Hb_HbA1c=svyglm(make_formula("SOMATIC_SCORE",c("LBXHGB","LBXGH",X)),design=des,family=gaussian()),
    spline_Hb_HbA1c=svyglm(as.formula(paste("SOMATIC_SCORE ~ ns(LBXHGB,df=3) + ns(LBXGH,df=3) +",paste(X,collapse=" + "))),design=des,family=gaussian()),
    raw7_multivariate=svyglm(make_formula("SOMATIC_SCORE",c(RAW7,X)),design=des,family=gaussian()),
    frozen_components=svyglm(make_formula("SOMATIC_SCORE",c(A,G3,X)),design=des,family=gaussian())
  )
  for (nm in names(fits)) comparison_rows <- rbind(comparison_rows,data.frame(
    population=period,model=nm,weighted_R2=wr2(fits[[nm]],dom),n=nrow(dom),design_df=degf(des)))

  # Raw markers beyond threshold scalars, and vice versa.
  union_raw <- svyglm(make_formula("SOMATIC_SCORE",c("A","G_HBA1C",RAW7,X)),design=des,family=gaussian())
  test_rows <- rbind(test_rows,
    safe_test(union_raw,RAW7,period,"comparator","raw7_extra_beyond_threshold_scalars"),
    safe_test(union_raw,c("A","G_HBA1C"),period,"comparator","threshold_scalars_extra_beyond_raw7"))

  # No-insulin G2: HbA1c + fasting glucose.
  des_a <- subset(des_full, DOMAIN_G2A_X3==1); dom_a <- dat[dat$DOMAIN_G2A_X3==1,]
  fit_a <- svyglm(make_formula("SOMATIC_SCORE",c(A,G2A,X)),design=des_a,family=gaussian())
  union_a <- svyglm(make_formula("SOMATIC_SCORE",c("A","G_HBA1C",A,G2A,X)),design=des_a,family=gaussian())
  sensitivity_rows <- rbind(sensitivity_rows,data.frame(population=period,sensitivity="G2_no_insulin",model="Apcs_G2pcs",weighted_R2=wr2(fit_a,dom_a),n=nrow(dom_a),design_df=degf(des_a)))
  test_rows <- rbind(test_rows,
    safe_test(fit_a,A,period,"G2_no_insulin","A_PCs_joint_given_G2_X"),
    safe_test(fit_a,G2A,period,"G2_no_insulin","G2_PCs_joint_given_A_X"),
    safe_test(union_a,c(A,G2A),period,"G2_no_insulin","all_components_extra_beyond_threshold_scalars"),
    safe_test(union_a,c("A","G_HBA1C"),period,"G2_no_insulin","threshold_scalars_extra_beyond_components"))

  # No-HbA1c G2: fasting glucose + insulin, with FPG-deficit scalar comparator.
  des_b <- subset(des_full, DOMAIN_G2B_X3==1); dom_b <- dat[dat$DOMAIN_G2B_X3==1,]
  fit_b <- svyglm(make_formula("SOMATIC_SCORE",c(A,G2B,X)),design=des_b,family=gaussian())
  union_b <- svyglm(make_formula("SOMATIC_SCORE",c("A","G_FPG",A,G2B,X)),design=des_b,family=gaussian())
  sensitivity_rows <- rbind(sensitivity_rows,data.frame(population=period,sensitivity="G2_no_HbA1c",model="Apcs_G2pcs",weighted_R2=wr2(fit_b,dom_b),n=nrow(dom_b),design_df=degf(des_b)))
  test_rows <- rbind(test_rows,
    safe_test(fit_b,A,period,"G2_no_HbA1c","A_PCs_joint_given_G2_X"),
    safe_test(fit_b,G2B,period,"G2_no_HbA1c","G2_PCs_joint_given_A_X"),
    safe_test(union_b,c(A,G2B),period,"G2_no_HbA1c","all_components_extra_beyond_A_FPG_scalars"),
    safe_test(union_b,c("A","G_FPG"),period,"G2_no_HbA1c","A_FPG_scalars_extra_beyond_components"))

  # Outcome-model robustness: quasi-Poisson for the bounded count-like somatic score.
  # Only report this where denominator df are useful; the code still runs all periods.
  des_q <- subset(des_full, DOMAIN_SOMATIC_SCORE_X3==1)
  fit_q_scalar <- svyglm(make_formula("SOMATIC_SCORE",c("A","G_HBA1C",X)),design=des_q,family=quasipoisson(link="log"))
  fit_q_union <- svyglm(make_formula("SOMATIC_SCORE",c("A","G_HBA1C",A,G3,X)),design=des_q,family=quasipoisson(link="log"))
  qt <- safe_test(fit_q_union,c(A,G3),period,"quasipoisson","all_PCs_extra_beyond_threshold_scalars")
  qp_rows <- rbind(qp_rows,qt)
}

write.csv(comparison_rows,file.path(results_dir,"48_representation_comparator_R2.csv"),row.names=FALSE)
write.csv(test_rows,file.path(results_dir,"48_comparator_sensitivity_tests.csv"),row.names=FALSE)
write.csv(sensitivity_rows,file.path(results_dir,"48_glycemic_sensitivity_R2.csv"),row.names=FALSE)
write.csv(qp_rows,file.path(results_dir,"48_quasipoisson_somatic_tests.csv"),row.names=FALSE)
