#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, math, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, f as f_dist

A_NAMES = ["Hb","RBC","MCV","RDW"]
G_NAMES = ["HbA1c","Glucose","logInsulin"]
X_NAMES = A_NAMES + G_NAMES
Z_TERMS = ["SP_SHARED1","SP_SHARED2","SP_DISCORD1","SP_DISCORD2","SP_A_PRIVATE3","SP_A_PRIVATE4","SP_G_PRIVATE3"]
Z_PRETTY = dict(zip(Z_TERMS,["Shared1","Shared2","Discord1","Discord2","A_private3","A_private4","G_private3"]))
X3 = ["RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021"]
SURVEY = ["SURVEY_WT","STRATUM","PSU"]
OUTCOME_CANDIDATES = {
    "SOMATIC":["SOMATIC_SCORE"],
    "COGAFF":["COGAFF_SUM"],
    "PHQ9_TOTAL":["PHQ9_TOTAL"],
}

def header(path):
    try:
        with path.open("r",encoding="utf-8-sig",errors="ignore",newline="") as f:
            return next(csv.reader(f))
    except Exception:
        return []

def choose_input(root, explicit):
    if explicit:
        p = Path(explicit)
        p = p if p.is_absolute() else root / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    required = set(["PERIOD","CYCLE","ELIGIBLE_ADULT_NONPREG"] + Z_TERMS + X3 + SURVEY)
    found = []
    seen = set()
    for sr in [root/"EDA"/"Results", root/"EDA", root]:
        if not sr.exists():
            continue
        iterator = sr.glob("*.csv") if sr == root else sr.rglob("*.csv")
        for p in iterator:
            try:
                key = p.resolve()
            except Exception:
                key = p
            if key in seen:
                continue
            seen.add(key)
            h = set(header(p))
            if not required.issubset(h):
                continue
            if not all(any(c in h for c in choices) for choices in OUTCOME_CANDIDATES.values()):
                continue
            score = p.stat().st_size
            n = p.name.lower()
            if "72" in n: score += 10**12
            if "input" in n or "model" in n or "shared_private" in n: score += 10**11
            found.append((score,p))
    if not found:
        raise FileNotFoundError("Could not auto-detect Script-72 modelling CSV. Use --input <path>.")
    return max(found,key=lambda x:x[0])[1]

def locate_transform(root):
    for p in [root/"EDA"/"Results"/"72_shared_private_frozen_transform.npz",
              root/"72_shared_private_frozen_transform.npz"]:
        if p.exists():
            return p
    raise FileNotFoundError("72_shared_private_frozen_transform.npz not found")

def build_jacobian(p):
    z = np.load(p)
    Asd = np.asarray(z["A_sd"],float).reshape(-1)
    Gsd = np.asarray(z["G_sd"],float).reshape(-1)
    WA = np.asarray(z["WA"],float)
    WG = np.asarray(z["WG"],float)
    rho = np.asarray(z["rho"],float).reshape(-1)
    if int(np.asarray(z["shared_rank"]).reshape(-1)[0]) != 2:
        raise ValueError("Expected shared_rank=2")
    Ju = WA.T @ np.diag(1/Asd)
    Jv = WG.T @ np.diag(1/Gsd)
    Jcx = np.zeros((7,7))
    Jcx[:4,:4] = Ju
    Jcx[4:,4:] = Jv
    H = np.zeros((7,7))
    for j in range(2):
        s = np.sqrt(2*(1+rho[j]))
        d = np.sqrt(2*(1-rho[j]))
        H[j,j] = 1/s
        H[j,4+j] = 1/s
        H[2+j,j] = 1/d
        H[2+j,4+j] = -1/d
    H[4,2] = 1
    H[5,3] = 1
    H[6,6] = 1
    J = H @ Jcx
    if np.linalg.matrix_rank(J) != 7:
        raise RuntimeError("Frozen X->Z transform is not full rank")
    return J, np.r_[Asd,Gsd]

R_CODE = r'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
out_dir <- args[2]
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)

PHYS <- c("SP_SHARED1","SP_SHARED2","SP_DISCORD1","SP_DISCORD2",
          "SP_A_PRIVATE3","SP_A_PRIVATE4","SP_G_PRIVATE3")

BLOCKS <- list(
  AG_shared=c("SP_SHARED1","SP_SHARED2"),
  AG_discord=c("SP_DISCORD1","SP_DISCORD2"),
  A_private=c("SP_A_PRIVATE3","SP_A_PRIVATE4"),
  G_private=c("SP_G_PRIVATE3")
)

X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")

for(v in intersect(c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE"), names(d))) {
  d[[v]] <- factor(d[[v]])
}

pick <- function(x) {
  z <- x[x %in% names(d)]
  if(length(z)==0) NA_character_ else z[1]
}

OUT <- c(
  SOMATIC="SOMATIC_SCORE",
  COGAFF="COGAFF_SUM",
  PHQ9_TOTAL="PHQ9_TOTAL"
)

if(!all(OUT %in% names(d))) stop("Missing corrected phenotype score column(s)")
if(!("ELIGIBLE_ADULT_NONPREG" %in% names(d))) stop("Missing ELIGIBLE_ADULT_NONPREG")

ff <- function(y, terms) as.formula(paste(y, "~", paste(terms, collapse=" + ")))

wr2 <- function(fit, dat, y) {
  p <- as.numeric(predict(fit,newdata=dat,type="response"))
  yy <- dat[[y]]
  w <- dat$SURVEY_WT
  ok <- is.finite(yy) & is.finite(w) & is.finite(p) & w>0
  yy <- yy[ok]; w <- w[ok]; p <- p[ok]
  mu <- sum(w*yy)/sum(w)
  sst <- sum(w*(yy-mu)^2)
  sse <- sum(w*(yy-p)^2)
  ifelse(sst>0, 1-sse/sst, NA)
}

wald <- function(fit, terms) {
  z <- tryCatch(
    regTermTest(fit, as.formula(paste("~",paste(terms,collapse=" + "))), method="Wald"),
    error=function(e) NULL
  )
  if(is.null(z)) return(c(F=NA,df_num=NA,df_den=NA,p=NA))
  c(F=as.numeric(z$Ftest), df_num=as.numeric(z$df), df_den=as.numeric(z$ddf), p=as.numeric(z$p))
}

coefs <- data.frame()
covs <- data.frame()
abls <- data.frame()
audit <- data.frame()

for(period in unique(d$PERIOD)) {
  dp <- d[d$PERIOD==period,,drop=FALSE]
  if(nrow(dp)==0) next
  covars <- X3
  if("CYCLE" %in% names(dp) && length(unique(dp$CYCLE[!is.na(dp$CYCLE)]))>1) {
    covars <- c(covars,"CYCLE")
  }

  for(oname in names(OUT)) {
    y <- OUT[[oname]]
    need <- unique(c(y,PHYS,covars,"SURVEY_WT","STRATUM","PSU"))
    if(!all(need %in% names(dp))) next

    ok <- dp$ELIGIBLE_ADULT_NONPREG==1 &
      complete.cases(dp[,need,drop=FALSE]) &
      is.finite(dp$SURVEY_WT) &
      dp$SURVEY_WT>0
    dp$MODEL_OK <- ok

    desfull <- svydesign(ids=~PSU,strata=~STRATUM,weights=~SURVEY_WT,nest=TRUE,data=dp)
    des <- subset(desfull,MODEL_OK)
    dm <- dp[ok,,drop=FALSE]
    if(nrow(dm)<100) next

    fit <- svyglm(ff(y,c(covars,PHYS)), design=des, family=gaussian())
    fullr2 <- wr2(fit,dm,y)

    sm <- summary(fit)$coefficients
    ci <- suppressMessages(confint(fit))
    pc <- grep("^Pr",colnames(sm),value=TRUE)[1]
    V <- vcov(fit)[PHYS,PHYS,drop=FALSE]

    for(term in PHYS) {
      coefs <- rbind(coefs,data.frame(
        period=period,outcome=oname,term=term,beta=coef(fit)[term],
        se=sm[term,"Std. Error"],ci_low=ci[term,1],ci_high=ci[term,2],
        p=sm[term,pc],domain_df=degf(des),n=nrow(dm)
      ))
    }

    for(ti in PHYS) for(tj in PHYS) {
      covs <- rbind(covs,data.frame(
        period=period,outcome=oname,term_i=ti,term_j=tj,
        covariance=V[ti,tj],domain_df=degf(des),n=nrow(dm)
      ))
    }

    for(term in PHYS) {
      red <- setdiff(PHYS,term)
      fitr <- svyglm(ff(y,c(covars,red)),design=des,family=gaussian())
      r2red <- wr2(fitr,dm,y)
      w <- wald(fit,term)
      abls <- rbind(abls,data.frame(
        period=period,outcome=oname,ablation_level="component",removed=term,n_removed=1,
        full_weighted_R2=fullr2,reduced_weighted_R2=r2red,
        delta_R2_full_minus_reduced=fullr2-r2red,
        wald_F=w["F"],wald_df_num=w["df_num"],wald_df_den=w["df_den"],wald_p=w["p"],
        n=nrow(dm),domain_df=degf(des)
      ))
    }

    for(bn in names(BLOCKS)) {
      rem <- BLOCKS[[bn]]
      red <- setdiff(PHYS,rem)
      fitr <- svyglm(ff(y,c(covars,red)),design=des,family=gaussian())
      r2red <- wr2(fitr,dm,y)
      w <- wald(fit,rem)
      abls <- rbind(abls,data.frame(
        period=period,outcome=oname,ablation_level="block",removed=bn,n_removed=length(rem),
        full_weighted_R2=fullr2,reduced_weighted_R2=r2red,
        delta_R2_full_minus_reduced=fullr2-r2red,
        wald_F=w["F"],wald_df_num=w["df_num"],wald_df_den=w["df_den"],wald_p=w["p"],
        n=nrow(dm),domain_df=degf(des)
      ))
    }

    audit <- rbind(audit,data.frame(
      period=period,outcome=oname,n=nrow(dm),domain_df=degf(des),full_weighted_R2=fullr2
    ))
  }
}

write.csv(coefs,file.path(out_dir,"77_component_coefficients_refit.csv"),row.names=FALSE)
write.csv(covs,file.path(out_dir,"77_component_covariance_long.csv"),row.names=FALSE)
write.csv(abls,file.path(out_dir,"77_component_and_block_ablation.csv"),row.names=FALSE)
write.csv(audit,file.path(out_dir,"77_refit_design_audit.csv"),row.names=FALSE)
'''

def run_r(inp,outdir):
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td)/"script77_refit.R"
        rp.write_text(R_CODE,encoding="utf-8")
        pr = subprocess.run(["Rscript",str(rp),str(inp),str(outdir)],capture_output=True,text=True)
        if pr.stdout.strip():
            print(pr.stdout)
        if pr.returncode != 0:
            if pr.stderr.strip():
                print(pr.stderr,file=sys.stderr)
            raise RuntimeError("R survey refit failed. Check Rscript + survey package.")

def covmat(df,period,outcome):
    q = df[(df.period.astype(str)==str(period)) & (df.outcome.astype(str)==str(outcome))]
    V = np.zeros((7,7))
    for i,ti in enumerate(Z_TERMS):
        for j,tj in enumerate(Z_TERMS):
            z = q[(q.term_i==ti) & (q.term_j==tj)]
            if len(z)!=1:
                raise RuntimeError(f"Missing covariance {period}/{outcome}/{ti}/{tj}")
            V[i,j] = float(z.iloc[0].covariance)
    return V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",default=None,help="Optional Script-72 modelling CSV")
    args = ap.parse_args()

    root = Path.cwd()
    out = root/"EDA"/"Results"
    out.mkdir(parents=True,exist_ok=True)

    inp = choose_input(root,args.input)
    tf = locate_transform(root)
    J,xsd = build_jacobian(tf)

    print("="*108)
    print("SCRIPT 77 — DERIVATIVE UNCERTAINTY + COMPONENT ABLATION [CORRECTED OUTCOME MAP]")
    print("="*108)
    print("Input:",inp)
    print("Transform:",tf)

    run_r(inp,out)

    c = pd.read_csv(out/"77_component_coefficients_refit.csv")
    v = pd.read_csv(out/"77_component_covariance_long.csv")
    ab = pd.read_csv(out/"77_component_and_block_ablation.csv")

    # -------------------------------------------------------------------------
    # CRITICAL GUARDRAIL:
    # The refit must reproduce the already accepted Script-72 coefficients.
    # If it does not, stop rather than interpret a mismatched phenotype model.
    # -------------------------------------------------------------------------
    ref_file = out/"72_shared_private_coefficients.csv"
    if not ref_file.exists():
        raise FileNotFoundError("72_shared_private_coefficients.csv not found for refit validation")

    ref = pd.read_csv(ref_file)
    ref = ref[ref["term"].isin(Z_TERMS)].copy()

    chk = c.merge(
        ref[["period","outcome","term","beta"]].rename(columns={"beta":"beta_script72"}),
        on=["period","outcome","term"],
        how="left"
    )
    chk["abs_beta_difference"] = (chk["beta"] - chk["beta_script72"]).abs()

    if chk["beta_script72"].isna().any():
        bad = chk.loc[chk["beta_script72"].isna(), ["period","outcome","term"]]
        raise RuntimeError(
            "Could not match all refitted coefficients to Script 72:\n"
            + bad.to_string(index=False)
        )

    max_beta_diff = float(chk["abs_beta_difference"].max())
    print(f"\nSCRIPT-72 REFIT CHECK: max |beta_refit - beta_72| = {max_beta_diff:.3e}")

    if max_beta_diff > 1e-8:
        print(chk[["period","outcome","term","beta","beta_script72","abs_beta_difference"]].to_string(index=False))
        raise RuntimeError(
            "STOP: Script-77 refit does not reproduce Script-72 coefficients. "
            "Do not interpret ablations/uncertainty until the modelling sample is identical."
        )

    print("PASS — Script 77 exactly reproduces the accepted Script-72 phenotype coefficients.")

    raw = []
    blocks = []

    for period,outcome in c[["period","outcome"]].drop_duplicates().itertuples(index=False,name=None):
        q = c[(c.period.astype(str)==str(period)) & (c.outcome.astype(str)==str(outcome))]
        bm = dict(zip(q.term,q.beta))
        if not all(t in bm for t in Z_TERMS):
            continue

        beta = np.array([bm[t] for t in Z_TERMS],float)
        Vb = covmat(v,period,outcome)
        df = float(q.domain_df.iloc[0])
        df_use = df if np.isfinite(df) and df>0 else 1e9
        tc = float(student_t.ppf(.975,df_use))

        g = J.T @ beta
        Vg = J.T @ Vb @ J

        D = np.diag(xsd)
        gs = D @ g
        Vgs = D @ Vg @ D

        for j,nm in enumerate(X_NAMES):
            se = math.sqrt(max(float(Vg[j,j]),0))
            ses = math.sqrt(max(float(Vgs[j,j]),0))
            stat = g[j]/se if se>0 else np.nan
            pval = float(2*student_t.sf(abs(stat),df_use)) if np.isfinite(stat) else np.nan
            raw.append(dict(
                period=period,outcome=outcome,block="A" if j<4 else "G",
                raw_variable=nm,dP_dx=float(g[j]),se_dP_dx=se,
                ci_low=float(g[j]-tc*se),ci_high=float(g[j]+tc*se),p=pval,
                std_dP_dx_per_1SD=float(gs[j]),std_se=ses,
                std_ci_low=float(gs[j]-tc*ses),std_ci_high=float(gs[j]+tc*ses),
                domain_df=df
            ))

        for bn,idx in [("A",np.arange(4)),("G",np.arange(4,7))]:
            gb = gs[idx]
            VV = Vgs[np.ix_(idx,idx)]
            rank = int(np.linalg.matrix_rank(VV))
            if rank:
                W = float(gb.T @ np.linalg.pinv(VV) @ gb)
                F = W/rank
                pv = float(f_dist.sf(F,rank,df_use))
            else:
                F = np.nan
                pv = np.nan
            blocks.append(dict(
                period=period,outcome=outcome,block=bn,
                standardized_gradient_L2=float(np.linalg.norm(gb)),
                wald_F=F,wald_df_num=rank,wald_df_den=df,wald_p=pv
            ))

    rawdf = pd.DataFrame(raw)
    blockdf = pd.DataFrame(blocks)
    rawdf.to_csv(out/"77_raw_derivative_uncertainty.csv",index=False)
    blockdf.to_csv(out/"77_raw_block_gradient_wald.csv",index=False)

    ct = c.copy()
    ct["component"] = ct.term.map(Z_PRETTY)
    ct["partial_derivative_dP_dz"] = ct.beta
    ct["ci_contains_zero"] = (ct.ci_low<=0) & (ct.ci_high>=0)
    ct = ct[["period","outcome","component","partial_derivative_dP_dz","se","ci_low","ci_high","p","ci_contains_zero","domain_df","n"]]
    ct.to_csv(out/"77_component_derivative_table.csv",index=False)

    ca = ab[ab.ablation_level.eq("component")].copy()
    ca["component"] = ca.removed.map(Z_PRETTY)
    cs = ca.merge(ct,on=["period","outcome","component"],how="left")
    cs.to_csv(out/"77_component_derivative_plus_ablation.csv",index=False)

    print("\nCOMPONENT DERIVATIVES + DROP-ONE ABLATION")
    cols = ["period","outcome","component","partial_derivative_dP_dz","ci_low","ci_high","p","delta_R2_full_minus_reduced","wald_p"]
    print(cs[cols].sort_values(["period","outcome","component"]).to_string(index=False))

    print("\nRAW DERIVATIVES WITH PROPAGATED UNCERTAINTY")
    cols2 = ["period","outcome","block","raw_variable","std_dP_dx_per_1SD","std_ci_low","std_ci_high","p"]
    print(rawdf[cols2].sort_values(["period","outcome","block","raw_variable"]).to_string(index=False))

    print("\nRAW BLOCK GRADIENT WALD")
    print(blockdf.sort_values(["period","outcome","block"]).to_string(index=False))

    print("\nIMPORTANT: this tests component ablation and derivative uncertainty.")
    print("Raw-variable indispensability still needs remove-variable -> rebuild X->Z -> revalidate.")
    print("\nOutputs written to:",out)

if __name__ == "__main__":
    main()
