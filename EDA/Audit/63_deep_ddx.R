
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(user_lib,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(user_lib,.libPaths()))
if (!requireNamespace("survey",quietly=TRUE))
  install.packages("survey",repos="https://cloud.r-project.org",lib=user_lib)
library(survey)
options(survey.lonely.psu="adjust")

d <- read.csv(input_file,stringsAsFactors=FALSE)
A <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE"))
  if (v %in% names(d)) d[[v]] <- factor(d[[v]])

fmla <- function(y,terms) as.formula(paste(y,"~",paste(terms,collapse=" + ")))

safe_test <- function(fit,terms) {
  z <- tryCatch(regTermTest(fit,as.formula(paste("~",paste(terms,collapse=" + "))),method="Wald"),
                error=function(e) NULL)
  if (is.null(z)) return(c(F=NA,df_num=NA,df_den=NA,p=NA))
  c(F=as.numeric(z$Ftest),df_num=as.numeric(z$df),df_den=as.numeric(z$ddf),p=as.numeric(z$p))
}

coefrow <- function(fit,term) {
  s <- coef(summary(fit))
  if (!(term %in% rownames(s))) return(c(beta=NA,se=NA,t=NA,p=NA))
  c(beta=s[term,1],se=s[term,2],t=s[term,3],p=s[term,4])
}

wr2 <- function(fit,des,y) {
  dat <- des$variables
  pred <- as.numeric(predict(fit,newdata=dat,type="response"))
  obs <- dat[[y]]
  w <- as.numeric(weights(des))
  ok <- is.finite(obs)&is.finite(pred)&is.finite(w)&w>0
  obs<-obs[ok]; pred<-pred[ok]; w<-w[ok]
  mu <- sum(w*obs)/sum(w)
  1 - sum(w*(obs-pred)^2)/sum(w*(obs-mu)^2)
}

pdat <- d[d$PERIOD=="2005-2008" & is.finite(d$SURVEY_WT) & d$SURVEY_WT>0,]
des <- svydesign(ids=~PSU,strata=~STRATUM,weights=~SURVEY_WT,nest=TRUE,data=pdat)

crows <- list(); ci <- 1
specs <- list(
  HbA1c=list(y="LBXGH",other=c("LBXGLU","LOG_IN")),
  FastingGlucose=list(y="LBXGLU",other=c("LBXGH","LOG_IN")),
  LogInsulin=list(y="LOG_IN",other=c("LBXGH","LBXGLU"))
)

for (nm in names(specs)) {
  sp <- specs[[nm]]
  needed <- c(sp$y,sp$other,A,X3)
  dom <- subset(des,DOMAIN_SOMATIC_SCORE_X3==1 & complete.cases(des$variables[,needed]))
  f0 <- svyglm(fmla(sp$y,c(sp$other,X3)),design=dom)
  f1 <- svyglm(fmla(sp$y,c(sp$other,A,X3)),design=dom)
  jt <- safe_test(f1,A)
  r20 <- wr2(f0,dom,sp$y); r21 <- wr2(f1,dom,sp$y)
  for (a in A) {
    cr <- coefrow(f1,a)
    crows[[ci]] <- data.frame(target=nm,component=a,beta=cr["beta"],se=cr["se"],p=cr["p"],
      block_F=jt["F"],block_p=jt["p"],base_weighted_R2=r20,full_weighted_R2=r21,
      delta_weighted_R2=r21-r20,n=nrow(model.frame(f1)),design_df=degf(dom))
    ci <- ci+1
  }
}
write.csv(do.call(rbind,crows),file.path(results_dir,"63_coupling_driver_decomposition.csv"),row.names=FALSE)

arows <- list(); ai <- 1
blocks <- list()
if ("LOG_CRP" %in% names(d)) blocks[["CRP"]] <- c("LOG_CRP")
if ("LOG_COTININE" %in% names(d)) blocks[["cotinine"]] <- c("LOG_COTININE")
if (all(c("LOG_CRP","LOG_COTININE") %in% names(d)))
  blocks[["CRP_plus_cotinine"]] <- c("LOG_CRP","LOG_COTININE")

for (bn in names(blocks)) {
  bv <- blocks[[bn]]
  needed <- c("SOMATIC_SCORE",A,G,X3,bv)
  dom <- subset(des,DOMAIN_SOMATIC_SCORE_X3==1 & complete.cases(des$variables[,needed]))
  f0 <- svyglm(fmla("SOMATIC_SCORE",c(A,G,X3)),design=dom)
  f1 <- svyglm(fmla("SOMATIC_SCORE",c(A,G,X3,bv)),design=dom)
  bt <- safe_test(f1,bv)
  for (term in c(A,G)) {
    b0 <- coefrow(f0,term); b1 <- coefrow(f1,term)
    v0 <- as.numeric(b0["beta"]); v1 <- as.numeric(b1["beta"])
    att <- ifelse(is.finite(v0)&abs(v0)>1e-12,100*(abs(v0)-abs(v1))/abs(v0),NA)
    arows[[ai]] <- data.frame(block=bn,component=term,base_beta=v0,extended_beta=v1,
      attenuation_percent=att,base_p=b0["p"],extended_p=b1["p"],block_p=bt["p"],
      n=nrow(model.frame(f1)),design_df=degf(dom))
    ai <- ai+1
  }
}
if (length(arows)>0)
  write.csv(do.call(rbind,arows),file.path(results_dir,"63_CRP_cotinine_attenuation.csv"),row.names=FALSE)

if (all(c("HRR_Q","TYG_Q","PHQ10") %in% names(d))) {
  ext <- subset(des,DOMAIN_PHQ9_TOTAL_X3==1 & HRR_Q %in% c(1,4) & TYG_Q %in% c(1,4))
  ext$variables$LOW_HRR <- as.numeric(ext$variables$HRR_Q==1)
  ext$variables$HIGH_TYG <- as.numeric(ext$variables$TYG_Q==4)
  lpm <- svyglm(fmla("PHQ10",c("LOW_HRR","HIGH_TYG","LOW_HRR:HIGH_TYG",X3)),design=ext,family=gaussian())
  logit <- svyglm(fmla("PHQ10",c("LOW_HRR","HIGH_TYG","LOW_HRR:HIGH_TYG",X3)),design=ext,family=quasibinomial())
  lr <- coefrow(lpm,"LOW_HRR:HIGH_TYG"); gr <- coefrow(logit,"LOW_HRR:HIGH_TYG")
  out <- data.frame(scale=c("additive_risk_difference","multiplicative_log_odds"),
    interaction_beta=c(lr["beta"],gr["beta"]),interaction_se=c(lr["se"],gr["se"]),
    interaction_p=c(lr["p"],gr["p"]),multiplicative_OR=c(NA,exp(as.numeric(gr["beta"]))),
    n=c(nrow(model.frame(lpm)),nrow(model.frame(logit))),design_df=c(degf(ext),degf(ext)))
  write.csv(out,file.path(results_dir,"63_HRR_TyG_interaction_scales.csv"),row.names=FALSE)
  prev <- svyby(~PHQ10,~LOW_HRR+HIGH_TYG,ext,svymean,na.rm=TRUE,keep.names=FALSE)
  write.csv(prev,file.path(results_dir,"63_HRR_TyG_joint_prevalence.csv"),row.names=FALSE)
}

if (all(c("LOG_FERRITIN","LOG_TFR","LOG_CRP","IRON_WOMEN_20_49_NONANAEMIC") %in% names(d))) {
  Xw <- c("RIDAGEYR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
  needed <- c("SOMATIC_SCORE","LOG_FERRITIN","LOG_TFR","LOG_CRP",A,G,Xw)
  iw <- subset(des,DOMAIN_SOMATIC_SCORE_X3==1 & IRON_WOMEN_20_49_NONANAEMIC==1 &
                  complete.cases(des$variables[,needed]))
  irows <- list(); ii <- 1
  for (target in c("LOG_FERRITIN","LOG_TFR","LOG_CRP")) {
    fit <- svyglm(fmla(target,c(A,Xw)),design=iw)
    for (a in A) {
      cr <- coefrow(fit,a)
      irows[[ii]] <- data.frame(
        analysis=paste0(target,"_on_A_PCs"),
        target=target,
        component=a,
        beta=cr["beta"],
        se=cr["se"],
        p=cr["p"],
        n=nrow(model.frame(fit)),
        design_df=degf(iw),
        base_beta=NA_real_,
        attenuation_percent=NA_real_,
        iron_CRP_block_p=NA_real_
      )
      ii <- ii+1
    }
  }
  f0 <- svyglm(fmla("SOMATIC_SCORE",c(A,G,Xw)),design=iw)
  f1 <- svyglm(fmla("SOMATIC_SCORE",c(A,G,Xw,"LOG_FERRITIN","LOG_TFR","LOG_CRP")),design=iw)
  bt <- safe_test(f1,c("LOG_FERRITIN","LOG_TFR","LOG_CRP"))
  for (a in A) {
    b0 <- coefrow(f0,a); b1 <- coefrow(f1,a)
    v0<-as.numeric(b0["beta"]); v1<-as.numeric(b1["beta"])
    att<-ifelse(is.finite(v0)&abs(v0)>1e-12,100*(abs(v0)-abs(v1))/abs(v0),NA)
    irows[[ii]] <- data.frame(
      analysis="PHQ_A_PC_attenuation_after_iron_CRP",
      target="SOMATIC_SCORE",
      component=a,
      beta=v1,
      se=b1["se"],
      p=b1["p"],
      n=nrow(model.frame(f1)),
      design_df=degf(iw),
      base_beta=v0,
      attenuation_percent=att,
      iron_CRP_block_p=bt["p"]
    )
    ii<-ii+1
  }
  write.csv(do.call(rbind,irows),file.path(results_dir,"63_iron_mechanism_women20_49_nonanaemic.csv"),row.names=FALSE)
}
