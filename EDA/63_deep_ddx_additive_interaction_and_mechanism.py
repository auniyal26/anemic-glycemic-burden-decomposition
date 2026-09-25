#!/usr/bin/env python
from pathlib import Path
import json, re, shutil, subprocess, urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDA_RESULTS = ROOT / "EDA" / "Results"
EDA_AUDIT = ROOT / "EDA" / "Audit"
EXT = ROOT / "EDA" / "External_NHANES"
RLIB = ROOT / "R_library"
INPUT = EDA_RESULTS / "62_ddx_model_input.csv"

for p in [EDA_RESULTS, EDA_AUDIT, EXT, RLIB]:
    p.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError("Run EDA/62_ddx_interaction_confounder_audit.py first.")

d = pd.read_csv(INPUT, dtype={"CYCLE": str, "PSU": str, "STRATUM": str})
d["CYCLE"] = d["CYCLE"].astype(str).str.replace(r"\.0$", "", regex=True).replace({"506":"0506","708":"0708"})

A_PCS = [f"A_OI_PC{i}_FZ" for i in range(1,4)]
G_PCS = [f"G3_OI_PC{i}_FZ" for i in range(1,4)]
X3 = ["RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021"]

for c in A_PCS + G_PCS + X3 + ["SEQN","CYCLE","PERIOD","SURVEY_WT","PSU","STRATUM",
                                "DOMAIN_SOMATIC_SCORE_X3","DOMAIN_PHQ9_TOTAL_X3",
                                "SOMATIC_SCORE","PHQ9_TOTAL","LBXHGB","LBXRBCSI",
                                "LBXMCVSI","LBXRDW","LBXGH","LBXGLU","LOG_IN"]:
    if c not in d.columns:
        raise ValueError(f"Missing required Script-62 column: {c}")

FILES = {
    "0506": {
        "CRP_D": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/CRP_D.XPT",
        "COT_D": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/COT_D.XPT",
        "FERTIN_D": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/FERTIN_D.XPT",
        "TFR_D": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/TFR_D.XPT",
        "TRIGLY_D": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/TRIGLY_D.XPT",
    },
    "0708": {
        "CRP_E": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/CRP_E.XPT",
        "COTNAL_E": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/COTNAL_E.XPT",
        "FERTIN_E": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/FERTIN_E.XPT",
        "TFR_E": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/TFR_E.XPT",
        "TRIGLY_E": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/TRIGLY_E.XPT",
    },
}

audit_rows = []
for cycle, items in FILES.items():
    cdir = EXT / cycle
    cdir.mkdir(parents=True, exist_ok=True)
    for stem, url in items.items():
        out = cdir / f"{stem}.XPT"
        status = "already_present"
        err = ""
        if not out.exists():
            try:
                urllib.request.urlretrieve(url, out)
                status = "downloaded"
            except Exception as e:
                status = "download_failed"
                err = str(e)
        audit_rows.append({"cycle":cycle,"stem":stem,"url":url,"path":str(out.relative_to(ROOT)),
                           "status":status,"error":err})
pd.DataFrame(audit_rows).to_csv(EDA_RESULTS/"63_external_file_audit.csv",index=False)

def read_external(cycle, stem, cols):
    p = EXT/cycle/f"{stem}.XPT"
    if not p.exists():
        return None
    x = pd.read_sas(p, format="xport")
    have = [c for c in cols if c in x.columns]
    if "SEQN" not in x.columns or not have:
        return None
    return x[["SEQN"]+have].copy()

specs = {
    "0506":[("CRP_D",["LBXCRP"]),("COT_D",["LBXCOT"]),("FERTIN_D",["LBXFER"]),
            ("TFR_D",["LBXTFR"]),("TRIGLY_D",["LBXTR"])],
    "0708":[("CRP_E",["LBXCRP"]),("COTNAL_E",["LBXCOT"]),("FERTIN_E",["LBXFER"]),
            ("TFR_E",["LBXTFR"]),("TRIGLY_E",["LBXTR"])],
}
extras=[]
for cycle, items in specs.items():
    q=d[d["CYCLE"].eq(cycle)][["SEQN"]].drop_duplicates().copy()
    for stem, cols in items:
        x=read_external(cycle,stem,cols)
        if x is not None:
            use=[c for c in cols if c in x.columns and c not in q.columns]
            if use:
                q=q.merge(x[["SEQN"]+use],on="SEQN",how="left",validate="one_to_one")
    q["CYCLE"]=cycle
    extras.append(q)
ext=pd.concat(extras,ignore_index=True)
new_cols=[c for c in ext.columns if c not in {"SEQN","CYCLE"}]
d=d.drop(columns=[c for c in new_cols if c in d.columns],errors="ignore")
d=d.merge(ext,on=["SEQN","CYCLE"],how="left",validate="one_to_one")

def num(c):
    return pd.to_numeric(d[c],errors="coerce") if c in d else pd.Series(np.nan,index=d.index)

if "LBXCRP" in d:
    x=num("LBXCRP"); d["LOG_CRP"]=np.log(x.where(x>0))
if "LBXCOT" in d:
    x=num("LBXCOT"); d["LOG_COTININE"]=np.log1p(x.where(x>=0))
if "LBXFER" in d:
    x=num("LBXFER"); d["LOG_FERRITIN"]=np.log(x.where(x>0))
if "LBXTFR" in d:
    x=num("LBXTFR"); d["LOG_TFR"]=np.log(x.where(x>0))

hb,rdw=num("LBXHGB"),num("LBXRDW")
d["HRR"]=hb/rdw.where(rdw>0)
if "LBXTR" in d:
    tg,fg=num("LBXTR"),num("LBXGLU")
    d["TYG_FASTING"]=np.log((tg*fg/2.0).where((tg>0)&(fg>0)))

phq=pd.to_numeric(d["PHQ9_TOTAL"],errors="coerce")
d["PHQ10"]=np.where(phq>=10,1.0,np.where(phq.notna(),0.0,np.nan))

def wcorr(x,y,w):
    x=pd.to_numeric(x,errors="coerce").to_numpy(float)
    y=pd.to_numeric(y,errors="coerce").to_numpy(float)
    w=pd.to_numeric(w,errors="coerce").to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    if ok.sum()<3: return np.nan
    x,y,w=x[ok],y[ok],w[ok]; sw=w.sum()
    mx=(w*x).sum()/sw; my=(w*y).sum()/sw
    vx=(w*(x-mx)**2).sum()/sw; vy=(w*(y-my)**2).sum()/sw
    if vx<=0 or vy<=0: return np.nan
    return float((w*(x-mx)*(y-my)).sum()/sw/np.sqrt(vx*vy))

def wquant(x,w,probs):
    x=pd.to_numeric(x,errors="coerce").to_numpy(float)
    w=pd.to_numeric(w,errors="coerce").to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    x,w=x[ok],w[ok]
    if len(x)==0: return [np.nan]*len(probs)
    ii=np.argsort(x); x=x[ii]; w=w[ii]
    cw=np.cumsum(w)/w.sum()
    return [float(np.interp(p,cw,x)) for p in probs]

disc=d[(d["PERIOD"]=="2005-2008")&(d["DOMAIN_SOMATIC_SCORE_X3"]==1)].copy()

gvars=[("LBXGH","HbA1c"),("LBXGLU","fasting glucose"),("LOG_IN","log insulin"),
       ("BMXBMI","BMI"),("DIABETES_SELFREPORT","self-reported diabetes"),
       ("INSULIN_USE","insulin use"),("LBXTR","fasting triglycerides"),
       ("TYG_FASTING","TyG")]
rows=[]
for pc in G_PCS:
    for var,label in gvars:
        if var in disc.columns:
            rows.append({"component":pc,"variable":var,"label":label,
                         "n":int(disc[[pc,var,"SURVEY_WT"]].dropna().shape[0]),
                         "weighted_correlation":wcorr(disc[pc],disc[var],disc["SURVEY_WT"])})
pd.DataFrame(rows).to_csv(EDA_RESULTS/"63_G_component_decoding.csv",index=False)

if {"HRR","TYG_FASTING"}.issubset(disc.columns):
    qh=wquant(disc["HRR"],disc["SURVEY_WT"],[.25,.5,.75])
    qt=wquant(disc["TYG_FASTING"],disc["SURVEY_WT"],[.25,.5,.75])
    d["HRR_Q"]=np.digitize(num("HRR"),qh,right=True)+1
    d["TYG_Q"]=np.digitize(num("TYG_FASTING"),qt,right=True)+1
    d.loc[num("HRR").isna(),"HRR_Q"]=np.nan
    d.loc[num("TYG_FASTING").isna(),"TYG_Q"]=np.nan
else:
    d["HRR_Q"]=np.nan; d["TYG_Q"]=np.nan

if "A" in d.columns:
    d["IRON_WOMEN_20_49_NONANAEMIC"]=(
        (pd.to_numeric(d["RIAGENDR"],errors="coerce")==2) &
        pd.to_numeric(d["RIDAGEYR"],errors="coerce").between(20,49) &
        (pd.to_numeric(d["A"],errors="coerce")==0)
    ).astype(int)
else:
    d["IRON_WOMEN_20_49_NONANAEMIC"]=0

model_csv=EDA_RESULTS/"63_deep_ddx_input.csv"
d.to_csv(model_csv,index=False)

rscript=shutil.which("Rscript")
if rscript is None:
    cand=sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if cand: rscript=str(cand[-1])
if rscript is None:
    raise RuntimeError("Rscript not found.")

def rvec(xs):
    return "c("+",".join(json.dumps(x) for x in xs)+")"

r_code = r'''
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
A <- __A__
G <- __G__
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
'''

r_code = r_code.replace("__A__", rvec(A_PCS)).replace("__G__", rvec(G_PCS))
rpath=EDA_AUDIT/"63_deep_ddx.R"
rpath.write_text(r_code,encoding="utf-8")

proc=subprocess.run([rscript,str(rpath),str(model_csv),str(EDA_RESULTS),str(RLIB)],
                    cwd=ROOT,capture_output=True,text=True)
(EDA_RESULTS/"63_R_stdout.txt").write_text(proc.stdout or "",encoding="utf-8")
(EDA_RESULTS/"63_R_stderr.txt").write_text(proc.stderr or "",encoding="utf-8")
if proc.returncode!=0:
    print(proc.stdout); print(proc.stderr)
    raise RuntimeError("R stage failed; see EDA/Results/63_R_stderr.txt")

manifest={
    "script":"63_deep_ddx_additive_interaction_and_mechanism.py",
    "questions":[
        "additive vs multiplicative HRR/TyG interaction",
        "which A PC drives residual HbA1c/glucose/insulin coupling and its incremental R2",
        "CRP/cotinine attenuation",
        "ferritin/sTfR/CRP mechanism substudy in non-anaemic women 20-49",
        "G-PC1/2/3 metabolic decoding"
    ],
    "downloads_location":"EDA/External_NHANES only",
    "causal_boundary":"diagnostic/falsification only; repeated cross-sectional NHANES does not identify causal effects",
    "locked_outputs_modified":False
}
(EDA_RESULTS/"63_deep_ddx_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

print("PASS  Script 63 deep DDx complete.")
print("PASS  Missing official NHANES files stored only under EDA/External_NHANES.")
print("PASS  G-component decoding written.")
print("PASS  Coupling-driver coefficients + incremental weighted R2 written.")
print("PASS  HRR/TyG additive-vs-multiplicative interaction written when fasting TG was available.")
print("PASS  CRP/cotinine matched attenuation written when available.")
print("PASS  Women 20-49 iron-mechanism substudy written when ferritin/sTfR/CRP were available.")
print("PASS  Locked Results untouched.")
