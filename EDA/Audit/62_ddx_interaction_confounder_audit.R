
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
A <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
BLOCKS <- list("social_behavioral_extra"=c("PARTNERED"),"inflammation"=c("LBXWBCSI"),"systemic_nutrition_liver"=c("LBXSAL","LOG_ALT","LOG_AST","LBXSBU","LBXSCH","LBXSUA"),"clinical_comorbidity"=c("DIABETES_SELFREPORT"))

for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) {
  if (v %in% names(d)) d[[v]] <- factor(d[[v]])
}

make_f <- function(y, terms) as.formula(paste(y,"~",paste(terms,collapse=" + ")))

safe_regterm <- function(fit, terms) {
  z <- tryCatch(
    regTermTest(fit, as.formula(paste("~",paste(terms,collapse=" + "))), method="Wald"),
    error=function(e) NULL
  )
  if (is.null(z)) return(c(F=NA,df_num=NA,df_den=NA,p=NA))
  c(F=as.numeric(z$Ftest),df_num=as.numeric(z$df),
    df_den=as.numeric(z$ddf),p=as.numeric(z$p))
}

coef_row <- function(fit, term) {
  s <- coef(summary(fit))
  if (!(term %in% rownames(s))) return(c(beta=NA,se=NA,t=NA,p=NA))
  c(beta=s[term,1],se=s[term,2],t=s[term,3],p=s[term,4])
}

# A) 3x3 component interaction map
int_rows <- list(); joint_rows <- list(); idx <- 1; jdx <- 1

for (period in c("2005-2008","2009-2018")) {
  pdat <- d[d$PERIOD==period & is.finite(d$SURVEY_WT) & d$SURVEY_WT>0,]
  if (nrow(pdat)==0) next
  des <- svydesign(ids=~PSU,strata=~STRATUM,weights=~SURVEY_WT,nest=TRUE,data=pdat)

  for (outcome in c("SOMATIC_SCORE","PHQ9_TOTAL")) {
    flag <- if (outcome=="SOMATIC_SCORE") "DOMAIN_SOMATIC_SCORE_X3" else "DOMAIN_PHQ9_TOTAL_X3"
    dom <- subset(des, get(flag)==1)
    main_terms <- c(A,G,X3)
    int_terms <- as.vector(outer(A,G,function(a,g) paste0(a,":",g)))

    fit_full <- tryCatch(svyglm(make_f(outcome,c(main_terms,int_terms)),design=dom),
                         error=function(e) NULL)
    if (!is.null(fit_full)) {
      jt <- safe_regterm(fit_full,int_terms)
      joint_rows[[jdx]] <- data.frame(
        period=period,outcome=outcome,
        test="all_9_Ai_x_Gj_interactions_given_main_effects_X3",
        F=jt["F"],df_num=jt["df_num"],df_den=jt["df_den"],p=jt["p"],
        n=nrow(model.frame(fit_full)),design_df=degf(dom)
      )
      jdx <- jdx+1
    }

    local <- list(); ldx <- 1
    for (a in A) for (g in G) {
      term <- paste0(a,":",g)
      fit <- tryCatch(svyglm(make_f(outcome,c(main_terms,term)),design=dom),
                      error=function(e) NULL)
      if (is.null(fit)) next
      tt <- safe_regterm(fit,term)
      cr <- coef_row(fit,term)
      local[[ldx]] <- data.frame(
        period=period,outcome=outcome,A_component=a,G_component=g,
        interaction=term,beta=cr["beta"],se=cr["se"],t=cr["t"],p=tt["p"],
        df_den=tt["df_den"],n=nrow(model.frame(fit)),design_df=degf(dom)
      )
      ldx <- ldx+1
    }
    if (length(local)>0) {
      z <- do.call(rbind,local)
      z$BH_p <- p.adjust(z$p,method="BH")
      int_rows[[idx]] <- z
      idx <- idx+1
    }
  }
}

if (length(int_rows)>0)
  write.csv(do.call(rbind,int_rows),
            file.path(results_dir,"62_AG_component_interaction_map.csv"),row.names=FALSE)
if (length(joint_rows)>0)
  write.csv(do.call(rbind,joint_rows),
            file.path(results_dir,"62_AG_component_interaction_joint_tests.csv"),row.names=FALSE)

# B) matched-sample attenuation, discovery, somatic PHQ
att_rows <- list(); block_rows <- list(); ai <- 1; bi <- 1
pdat <- d[d$PERIOD=="2005-2008" & is.finite(d$SURVEY_WT) & d$SURVEY_WT>0,]
des0 <- svydesign(ids=~PSU,strata=~STRATUM,weights=~SURVEY_WT,nest=TRUE,data=pdat)
base_terms <- c(A,G,X3)

for (bn in names(BLOCKS)) {
  bvars <- BLOCKS[[bn]]
  if (length(bvars)==0) next
  cc_expr <- paste0("DOMAIN_SOMATIC_SCORE_X3 == 1 & ",
                    paste(paste0("!is.na(",bvars,")"),collapse=" & "))
  dom <- subset(des0,eval(parse(text=cc_expr)))

  fit0 <- tryCatch(svyglm(make_f("SOMATIC_SCORE",base_terms),design=dom),
                   error=function(e) NULL)
  fit1 <- tryCatch(svyglm(make_f("SOMATIC_SCORE",c(base_terms,bvars)),design=dom),
                   error=function(e) NULL)
  if (is.null(fit0) || is.null(fit1)) next

  bt <- safe_regterm(fit1,bvars)
  block_rows[[bi]] <- data.frame(
    block=bn,variables=paste(bvars,collapse=";"),
    n=nrow(model.frame(fit1)),design_df=degf(dom),
    F=bt["F"],df_num=bt["df_num"],df_den=bt["df_den"],p=bt["p"])
  bi <- bi+1

  for (term in c(A,G)) {
    c0 <- coef_row(fit0,term); c1 <- coef_row(fit1,term)
    b0 <- as.numeric(c0["beta"]); b1 <- as.numeric(c1["beta"])
    pct <- ifelse(is.finite(b0) && abs(b0)>1e-12,100*(b1-b0)/abs(b0),NA)
    att <- ifelse(is.finite(b0) && abs(b0)>1e-12,
                  100*(abs(b0)-abs(b1))/abs(b0),NA)
    att_rows[[ai]] <- data.frame(
      block=bn,component=term,base_beta=b0,extended_beta=b1,
      signed_percent_change=pct,
      absolute_magnitude_attenuation_percent=att,
      base_p=as.numeric(c0["p"]),extended_p=as.numeric(c1["p"]),
      n=nrow(model.frame(fit1)),design_df=degf(dom))
    ai <- ai+1
  }
}

if (length(att_rows)>0)
  write.csv(do.call(rbind,att_rows),
            file.path(results_dir,"62_matched_block_attenuation.csv"),row.names=FALSE)
if (length(block_rows)>0)
  write.csv(do.call(rbind,block_rows),
            file.path(results_dir,"62_explanatory_block_tests.csv"),row.names=FALSE)

# C) RBC-linked glycation / measurement-coupling fingerprint
couple_rows <- list(); ci <- 1
need_g <- c("LBXGH","LBXGLU","LOG_IN")

if (all(need_g %in% names(d))) {
  cc_expr <- paste0("DOMAIN_SOMATIC_SCORE_X3 == 1 & ",
                    paste(paste0("!is.na(",need_g,")"),collapse=" & "))
  domg <- subset(des0,eval(parse(text=cc_expr)))

  specs <- list(
    HbA1c=list(y="LBXGH",other=c("LBXGLU","LOG_IN")),
    FastingGlucose=list(y="LBXGLU",other=c("LBXGH","LOG_IN")),
    LogInsulin=list(y="LOG_IN",other=c("LBXGH","LBXGLU"))
  )

  for (nm in names(specs)) {
    sp <- specs[[nm]]
    fit <- tryCatch(svyglm(make_f(sp$y,c(sp$other,A,X3)),design=domg),
                    error=function(e) NULL)
    if (!is.null(fit)) {
      z <- safe_regterm(fit,A)
      couple_rows[[ci]] <- data.frame(
        target=nm,test="A_PC_block_given_other_two_G_markers_X3",
        F=z["F"],df_num=z["df_num"],df_den=z["df_den"],p=z["p"],
        n=nrow(model.frame(fit)),design_df=degf(domg))
      ci <- ci+1
    }
  }

  rawA <- c("LBXHGB","LBXRBCSI","LBXMCVSI","LBXRDW")
  if (all(rawA %in% names(d))) {
    cc2 <- paste0("DOMAIN_SOMATIC_SCORE_X3 == 1 & ",
                  paste(paste0("!is.na(",c(need_g,rawA),")"),collapse=" & "))
    domr <- subset(des0,eval(parse(text=cc2)))
    for (nm in names(specs)) {
      sp <- specs[[nm]]
      fit <- tryCatch(svyglm(make_f(sp$y,c(sp$other,rawA,X3)),design=domr),
                      error=function(e) NULL)
      if (!is.null(fit)) {
        z <- safe_regterm(fit,rawA)
        couple_rows[[ci]] <- data.frame(
          target=nm,test="raw_Hb_RBC_MCV_RDW_block_given_other_two_G_markers_X3",
          F=z["F"],df_num=z["df_num"],df_den=z["df_den"],p=z["p"],
          n=nrow(model.frame(fit)),design_df=degf(domr))
        ci <- ci+1
      }
    }
  }
}

if (length(couple_rows)>0)
  write.csv(do.call(rbind,couple_rows),
            file.path(results_dir,"62_measurement_coupling_fingerprint.csv"),row.names=FALSE)
