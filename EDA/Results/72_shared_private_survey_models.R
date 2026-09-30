args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
out_dir <- args[2]
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])

SHARED <- c("SP_SHARED1","SP_SHARED2")
DISCORD <- c("SP_DISCORD1","SP_DISCORD2")
A_PRIVATE <- c("SP_A_PRIVATE3","SP_A_PRIVATE4")
G_PRIVATE <- c("SP_G_PRIVATE3")
PHYS <- c(SHARED, DISCORD, A_PRIVATE, G_PRIVATE)
A_CANON <- c("CCA_A1","CCA_A2","CCA_A3","CCA_A4")
G_CANON <- c("CCA_G1","CCA_G2","CCA_G3")
INTERACTIONS <- as.vector(outer(A_CANON, G_CANON, function(a,g) paste0(a,":",g)))
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")

make_formula <- function(outcome, terms) as.formula(paste(outcome, "~", paste(terms, collapse=" + ")))

safe_test <- function(fit, terms, period, outcome, label) {
  if (length(terms)==0) return(NULL)
  z <- tryCatch(regTermTest(fit, as.formula(paste("~", paste(terms, collapse=" + "))), method="Wald"), error=function(e) NULL)
  if (is.null(z)) return(data.frame(period=period,outcome=outcome,test=label,F=NA,df_num=NA,df_den=NA,p=NA,inference_status="unavailable"))
  ddf <- as.numeric(z$ddf)
  status <- ifelse(is.na(ddf),"unavailable",ifelse(ddf<=3,"design_limited","available"))
  data.frame(period=period,outcome=outcome,test=label,F=as.numeric(z$Ftest),df_num=as.numeric(z$df),df_den=ddf,p=as.numeric(z$p),inference_status=status)
}

extract_coef <- function(fit, period, outcome, model, terms) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]
  out <- data.frame()
  for (term in terms) {
    if (!(term %in% rownames(sm))) next
    out <- rbind(out, data.frame(period=period,outcome=outcome,model=model,term=term,beta=sm[term,"Estimate"],se=sm[term,"Std. Error"],ci_low=ci[term,1],ci_high=ci[term,2],p=sm[term,pcol]))
  }
  out
}

weighted_r2 <- function(fit, dat, outcome) {
  pred <- as.numeric(predict(fit, newdata=dat, type="response"))
  y <- dat[[outcome]]; w <- dat$SURVEY_WT
  ok <- is.finite(y) & is.finite(w) & is.finite(pred) & w>0
  y <- y[ok]; w <- w[ok]; pred <- pred[ok]
  mu <- sum(w*y)/sum(w); sst <- sum(w*(y-mu)^2); sse <- sum(w*(y-pred)^2)
  ifelse(sst>0, 1-sse/sst, NA)
}

all_tests <- data.frame(); all_coef <- data.frame(); all_r2 <- data.frame(); audit <- data.frame()
periods <- unique(d$PERIOD)
outcomes <- c(SOMATIC="SOMATIC_SCORE", COGAFF="COGAFF_SUM", PHQ9_TOTAL="PHQ9_TOTAL")

for (period in periods) {
  dp <- d[d$PERIOD==period,]
  if (nrow(dp)==0) next
  covars <- X3
  if (length(unique(dp$CYCLE))>1) covars <- c(covars,"CYCLE")

  for (oname in names(outcomes)) {
    outcome <- outcomes[[oname]]
    need <- unique(c(outcome, PHYS, A_CANON, G_CANON, covars, "SURVEY_WT","STRATUM","PSU"))
    ok <- dp$ELIGIBLE_ADULT_NONPREG==1 &
      complete.cases(dp[,need]) &
      is.finite(dp$SURVEY_WT) &
      dp$SURVEY_WT>0
    dp$MODEL_OK <- ok
    des_full <- svydesign(ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT, nest=TRUE, data=dp)
    des <- subset(des_full, MODEL_OK)
    dm <- dp[ok,]
    if (nrow(dm)<100) next

    fit0 <- svyglm(make_formula(outcome,covars), design=des, family=gaussian())
    fitmain <- svyglm(make_formula(outcome,c(covars,PHYS)), design=des, family=gaussian())
    fitint <- svyglm(make_formula(outcome,c(covars,PHYS,INTERACTIONS)), design=des, family=gaussian())

    all_tests <- rbind(all_tests,
      safe_test(fitmain, SHARED, period,oname,"shared_joint_physiology_given_private_discord_covars"),
      safe_test(fitmain, DISCORD, period,oname,"AG_discordance_given_shared_private_covars"),
      safe_test(fitmain, A_PRIVATE, period,oname,"A_private_given_shared_Gprivate_discord_covars"),
      safe_test(fitmain, G_PRIVATE, period,oname,"G_private_given_shared_Aprivate_discord_covars"),
      safe_test(fitmain, PHYS, period,oname,"all_shared_private_physiology_given_covars"),
      safe_test(fitint, INTERACTIONS, period,oname,"multiplicative_AxG_beyond_shared_private_main_effects")
    )

    all_coef <- rbind(all_coef, extract_coef(fitmain,period,oname,"shared_private_main",PHYS))
    all_r2 <- rbind(all_r2,
      data.frame(period=period,outcome=oname,model="covariates_only",weighted_R2=weighted_r2(fit0,dm,outcome)),
      data.frame(period=period,outcome=oname,model="shared_private_main",weighted_R2=weighted_r2(fitmain,dm,outcome)),
      data.frame(period=period,outcome=oname,model="shared_private_plus_multiplicative",weighted_R2=weighted_r2(fitint,dm,outcome))
    )
    audit <- rbind(audit,data.frame(period=period,outcome=oname,n=nrow(dm),full_design_df=degf(des_full),domain_df=degf(des)))
  }
}

write.csv(all_tests,file.path(out_dir,"72_shared_private_block_tests.csv"),row.names=FALSE)
write.csv(all_coef,file.path(out_dir,"72_shared_private_coefficients.csv"),row.names=FALSE)
write.csv(all_r2,file.path(out_dir,"72_shared_private_model_R2.csv"),row.names=FALSE)
write.csv(audit,file.path(out_dir,"72_shared_private_design_audit.csv"),row.names=FALSE)
