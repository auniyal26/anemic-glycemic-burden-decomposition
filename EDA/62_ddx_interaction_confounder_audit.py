#!/usr/bin/env python
# =============================================================================
# 62_ddx_interaction_confounder_audit.py
#
# "Differential diagnosis" of the current A/G physiological representation.
#
# Questions:
#   1) Which measured covariates align with which frozen A/G components?
#   2) Which candidate explanatory blocks attenuate component -> PHQ relations?
#   3) Is the old scalar A×G null hiding specific Ai×Gj interactions?
#   4) Is there evidence of RBC-linked HbA1c coupling beyond glucose/insulin?
#
# No PCA refit. No locked output modification.
# =============================================================================

from pathlib import Path
import json
import re
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
EDA_RESULTS = ROOT / "EDA" / "Results"
EDA_FIGURES = ROOT / "EDA" / "Figures"
AUDIT = ROOT / "EDA" / "Audit"
RLIB = ROOT / "R_library"
DATA_DISC = ROOT / "Data" / "NHANES"

for p in [EDA_RESULTS, EDA_FIGURES, AUDIT, RLIB]:
    p.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "47_domain_corrected_AG_input.csv"
if not INPUT.exists():
    raise FileNotFoundError(f"Missing {INPUT}. Run locked Script 47 first.")

def normalize_cycle(s):
    s = re.sub(r"\.0$", "", str(s))
    return {"506": "0506", "708": "0708"}.get(s, s)

def pc_sort(cols):
    return sorted(cols, key=lambda c: int(re.search(r"PC(\d+)", c).group(1)))

def xpt_path(base: Path, stem: str):
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    return matches[0] if len(matches) == 1 else None

def load_cols(base: Path, stem: str, cols):
    p = xpt_path(base, stem)
    if p is None:
        return None, []
    x = pd.read_sas(p, format="xport")
    have = [c for c in cols if c in x.columns]
    if "SEQN" not in x.columns or not have:
        return None, []
    return x[["SEQN"] + have].copy(), have

def merge_optional(left, base, stem, cols, source_rows):
    p = xpt_path(base, stem)
    x, have = load_cols(base, stem, cols)
    source_rows.append({
        "stem": stem,
        "found_file": p is not None,
        "requested": ";".join(cols),
        "found_columns": ";".join(have),
    })
    if x is None:
        return left
    use = [c for c in have if c not in left.columns]
    if not use:
        return left
    return left.merge(x[["SEQN"] + use], on="SEQN", how="left", validate="one_to_one")

def clean_binary_yes_no(x):
    x = pd.to_numeric(x, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out[x == 1] = 1.0
    out[x == 2] = 0.0
    return out

def weighted_var(x, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.nan
    x, w = x[ok], w[ok]
    mu = np.sum(w*x)/np.sum(w)
    return float(np.sum(w*(x-mu)**2)/np.sum(w))

def weighted_corr(x, y, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    sw = w.sum()
    mx, my = np.sum(w*x)/sw, np.sum(w*y)/sw
    vx = np.sum(w*(x-mx)**2)/sw
    vy = np.sum(w*(y-my)**2)/sw
    if vx <= 0 or vy <= 0:
        return np.nan
    cov = np.sum(w*(x-mx)*(y-my))/sw
    return float(cov/np.sqrt(vx*vy))

# Locked scaffold
d = pd.read_csv(INPUT, dtype={"CYCLE": str, "PSU": str, "STRATUM": str})
d["CYCLE"] = d["CYCLE"].map(normalize_cycle)

# Prefer the explicitly frozen Script-47 scores. The input also contains
# unsuffixed copies, so generic regex matching would return six columns/block.
A_PCS = [
    f"A_OI_PC{i}_FZ" if f"A_OI_PC{i}_FZ" in d.columns else f"A_OI_PC{i}"
    for i in range(1, 4)
]
G_PCS = [
    f"G3_OI_PC{i}_FZ" if f"G3_OI_PC{i}_FZ" in d.columns else f"G3_OI_PC{i}"
    for i in range(1, 4)
]

missing_pcs = [c for c in A_PCS + G_PCS if c not in d.columns]
if missing_pcs:
    raise ValueError(f"Could not detect required frozen PCs: {missing_pcs}")

X3 = ["RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021"]
required = ["SEQN","CYCLE","PERIOD","SURVEY_WT","PSU","STRATUM",
            "DOMAIN_SOMATIC_SCORE_X3","DOMAIN_PHQ9_TOTAL_X3",
            "SOMATIC_SCORE","PHQ9_TOTAL"] + X3 + A_PCS + G_PCS
missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Missing locked columns: {missing}")

# Restore extra discovery variables.
suffix = {"0506":"D", "0708":"E"}
extras = []

for cycle, s in suffix.items():
    q = d[d["CYCLE"].eq(cycle)][["SEQN"]].drop_duplicates().copy()
    base = DATA_DISC / cycle
    audit_rows = []

    specs = [
        (f"DEMO_{s}", ["DMDMARTL"]),
        (f"CBC_{s}", ["LBXHGB","LBXRBCSI","LBXMCVSI","LBXRDW","LBXWBCSI","LBXPLTSI"]),
        (f"CRP_{s}", ["LBXCRP"]),
        (f"BIOPRO_{s}", ["LBXSAL","LBXSATSI","LBXSASSI","LBXSBU","LBXSCH","LBXSUA","LBXSTR"]),
        (f"ALQ_{s}", ["ALQ130"]),
        (f"SLQ_{s}", ["SLD010H"]),
        (f"BPQ_{s}", ["BPQ020"]),
        (f"DIQ_{s}", ["DIQ010","DIQ050"]),
        (f"MCQ_{s}", ["MCQ160B","MCQ160C","MCQ160D","MCQ160E","MCQ220"]),
        (f"GHB_{s}", ["LBXGH"]),
        (f"GLU_{s}", ["LBXGLU","LBXIN"]),
        (f"FERTIN_{s}", ["LBXFER"]),
        (f"TFR_{s}", ["LBXTFR"]),
        (f"FETIB_{s}", ["LBXIRN","LBXTIB","LBDPCT"]),
    ]
    for stem, cols in specs:
        q = merge_optional(q, base, stem, cols, audit_rows)

    if "LBXIN" not in q.columns:
        q = merge_optional(q, base, f"INS_{s}", ["LBXIN"], audit_rows)

    q["CYCLE"] = cycle
    for r in audit_rows:
        r["cycle"] = cycle
    pd.DataFrame(audit_rows).to_csv(EDA_RESULTS / f"62_raw_source_audit_{cycle}.csv", index=False)
    extras.append(q)

extra = pd.concat(extras, ignore_index=True)
d = d.merge(extra, on=["SEQN","CYCLE"], how="left", validate="one_to_one")

# Derived candidates.
d["FEMALE"] = (pd.to_numeric(d["RIAGENDR"], errors="coerce") == 2).astype(float)
sm = pd.to_numeric(d["SMOKING3"], errors="coerce")
d["CURRENT_SMOKER"] = np.where(sm == 2, 1.0, np.where(sm.notna(), 0.0, np.nan))
d["FORMER_SMOKER"] = np.where(sm == 1, 1.0, np.where(sm.notna(), 0.0, np.nan))

if "DMDMARTL" in d:
    x = pd.to_numeric(d["DMDMARTL"], errors="coerce")
    d["PARTNERED"] = np.where(x.isin([1,6]), 1.0,
                      np.where(x.isin([2,3,4,5]), 0.0, np.nan))
if "ALQ130" in d:
    x = pd.to_numeric(d["ALQ130"], errors="coerce")
    d["ALCOHOL_DRINKS_DAY"] = x.where((x >= 0) & (x < 777))
if "SLD010H" in d:
    x = pd.to_numeric(d["SLD010H"], errors="coerce")
    d["SLEEP_HOURS"] = x.where((x >= 1) & (x <= 24))
if "BPQ020" in d:
    d["HYPERTENSION_SELFREPORT"] = clean_binary_yes_no(d["BPQ020"])
if "DIQ010" in d:
    d["DIABETES_SELFREPORT"] = clean_binary_yes_no(d["DIQ010"])
if "DIQ050" in d:
    d["INSULIN_USE"] = clean_binary_yes_no(d["DIQ050"])
if "MCQ220" in d:
    d["CANCER_HISTORY"] = clean_binary_yes_no(d["MCQ220"])

cvd_vars = [c for c in ["MCQ160B","MCQ160C","MCQ160D","MCQ160E"] if c in d]
if cvd_vars:
    vals = pd.concat([clean_binary_yes_no(d[c]) for c in cvd_vars], axis=1)
    d["CVD_ANY"] = np.where(vals.eq(1).any(axis=1), 1.0,
                    np.where(vals.notna().any(axis=1), 0.0, np.nan))

if "LBXCRP" in d:
    x = pd.to_numeric(d["LBXCRP"], errors="coerce")
    d["LOG_CRP"] = np.log(x.where(x > 0))
for raw, new in [("LBXSATSI","LOG_ALT"),("LBXSASSI","LOG_AST")]:
    if raw in d:
        x = pd.to_numeric(d[raw], errors="coerce")
        d[new] = np.log1p(x.where(x >= 0))
if "LBXIN" in d:
    x = pd.to_numeric(d["LBXIN"], errors="coerce")
    d["LOG_IN"] = np.log(x.where(x > 0))

# Closest-paper comparator indices if ingredients exist.
if {"LBXHGB","LBXRDW"}.issubset(d.columns):
    hb = pd.to_numeric(d["LBXHGB"], errors="coerce")
    rdw = pd.to_numeric(d["LBXRDW"], errors="coerce")
    d["HRR"] = hb / rdw.where(rdw > 0)
if {"LBXSTR","LBXGLU"}.issubset(d.columns):
    tg = pd.to_numeric(d["LBXSTR"], errors="coerce")
    fg = pd.to_numeric(d["LBXGLU"], errors="coerce")
    d["TYG"] = np.log((tg*fg/2.0).where((tg > 0) & (fg > 0)))

# Covariate-vector atlas.
candidate_meta = {
    "RIDAGEYR": ("age","baseline/demographic"),
    "FEMALE": ("female sex","baseline/demographic"),
    "INDFMPIR": ("poverty-income ratio","socioeconomic"),
    "EDUC3": ("education category","socioeconomic"),
    "CURRENT_SMOKER": ("current smoking","behavioral"),
    "FORMER_SMOKER": ("former smoking","behavioral"),
    "BMXBMI": ("BMI","possible confounder-or-mediator"),
    "EGFR_2021": ("eGFR","possible confounder-or-mediator"),
    "PARTNERED": ("married/living with partner","social"),
    "ALCOHOL_DRINKS_DAY": ("average drinks/day","behavioral"),
    "SLEEP_HOURS": ("usual weekday sleep hours","symptom-overlap sensitivity"),
    "LOG_CRP": ("log CRP","inflammation"),
    "LBXWBCSI": ("white blood cell count","inflammation"),
    "LBXPLTSI": ("platelet count","hematologic/inflammation"),
    "LBXSAL": ("albumin","nutrition/systemic illness"),
    "LOG_ALT": ("log ALT","liver/metabolic"),
    "LOG_AST": ("log AST","liver/systemic"),
    "LBXSBU": ("blood urea nitrogen","renal/systemic"),
    "LBXSCH": ("total cholesterol","metabolic"),
    "LBXSUA": ("uric acid","metabolic/inflammatory"),
    "HYPERTENSION_SELFREPORT": ("hypertension","clinical/metabolic"),
    "DIABETES_SELFREPORT": ("diabetes","clinical/metabolic"),
    "INSULIN_USE": ("insulin use","treatment/disease severity"),
    "CVD_ANY": ("CVD history","clinical/systemic"),
    "CANCER_HISTORY": ("cancer history","clinical/systemic"),
}

disc = d[(d["PERIOD"]=="2005-2008") & (d["DOMAIN_SOMATIC_SCORE_X3"]==1)].copy()
avail_rows, vector_rows = [], []

for var, (label, role) in candidate_meta.items():
    if var not in d:
        avail_rows.append({"variable":var,"label":label,"role":role,"present":False,
                           "n_nonmissing_discovery":0,"weighted_variance":np.nan})
        continue
    n = int(disc[var].notna().sum())
    vv = weighted_var(disc[var], disc["SURVEY_WT"])
    avail_rows.append({"variable":var,"label":label,"role":role,"present":True,
                       "n_nonmissing_discovery":n,"weighted_variance":vv})
    if n < 200 or not np.isfinite(vv) or vv <= 0:
        continue
    row = {
        "variable":var, "label":label, "role":role,
        "n_nonmissing_discovery":n,
        "corr_somatic_PHQ":weighted_corr(disc[var],disc["SOMATIC_SCORE"],disc["SURVEY_WT"]),
        "corr_total_PHQ9":weighted_corr(disc[var],disc["PHQ9_TOTAL"],disc["SURVEY_WT"]),
    }
    vals = []
    for pc in A_PCS + G_PCS:
        rr = weighted_corr(disc[var],disc[pc],disc["SURVEY_WT"])
        row[f"corr_{pc}"] = rr
        if np.isfinite(rr):
            vals.append(rr)
    row["component_vector_norm"] = float(np.sqrt(np.sum(np.square(vals)))) if vals else np.nan
    row["max_abs_component_corr"] = float(np.max(np.abs(vals))) if vals else np.nan
    vector_rows.append(row)

availability = pd.DataFrame(avail_rows)
vectors = pd.DataFrame(vector_rows)
availability.to_csv(EDA_RESULTS/"62_candidate_covariate_availability.csv", index=False)
vectors.to_csv(EDA_RESULTS/"62_confounder_vector_atlas.csv", index=False)

if not vectors.empty:
    vp = vectors.sort_values("component_vector_norm", ascending=False).head(18)
    corr_cols = [f"corr_{x}" for x in A_PCS+G_PCS]
    fig, ax = plt.subplots(figsize=(10, max(6,0.38*len(vp))))
    im = ax.imshow(vp[corr_cols].to_numpy(float), vmin=-1, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["A-PC1","A-PC2","A-PC3","G-PC1","G-PC2","G-PC3"])
    ax.set_yticks(range(len(vp)))
    ax.set_yticklabels(vp["label"].tolist())
    ax.set_xlabel("Frozen physiological component")
    ax.set_ylabel("Candidate explanatory covariate")
    ax.set_title("Weighted covariate-to-component vectors, NHANES 2005–08")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Survey-weighted correlation")
    fig.tight_layout()
    fig.savefig(EDA_FIGURES/"62_confounder_vector_heatmap.png", dpi=300)
    plt.close(fig)

# Matched-sample diagnostic blocks.
block_defs = {
    "social_behavioral_extra":["PARTNERED","ALCOHOL_DRINKS_DAY"],
    "inflammation":["LOG_CRP","LBXWBCSI"],
    "systemic_nutrition_liver":["LBXSAL","LOG_ALT","LOG_AST","LBXSBU","LBXSCH","LBXSUA"],
    "clinical_comorbidity":["HYPERTENSION_SELFREPORT","DIABETES_SELFREPORT","CVD_ANY","CANCER_HISTORY"],
    "sleep_overlap_sensitivity_ONLY":["SLEEP_HOURS"],
}
valid_blocks = {}
for bn, vars_ in block_defs.items():
    keep = []
    for v in vars_:
        if v in disc:
            n = int(disc[v].notna().sum())
            vv = weighted_var(disc[v],disc["SURVEY_WT"])
            if n >= 300 and np.isfinite(vv) and vv > 0:
                keep.append(v)
    if keep:
        valid_blocks[bn] = keep

model_csv = EDA_RESULTS/"62_ddx_model_input.csv"
d.to_csv(model_csv, index=False)

# R survey stage.
rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])
if rscript is None:
    raise RuntimeError("Rscript not found.")

def r_vec(items):
    return "c(" + ",".join(json.dumps(x) for x in items) + ")"

block_r = "list(" + ",".join(
    f"{json.dumps(k)}={r_vec(v)}" for k,v in valid_blocks.items()
) + ")"

r_code = r'''
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
A <- __A_VEC__
G <- __G_VEC__
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
BLOCKS <- __BLOCKS__

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
'''

r_code = r_code.replace("__A_VEC__", r_vec(A_PCS))
r_code = r_code.replace("__G_VEC__", r_vec(G_PCS))
r_code = r_code.replace("__BLOCKS__", block_r)

r_path = AUDIT/"62_ddx_interaction_confounder_audit.R"
r_path.write_text(r_code, encoding="utf-8")

proc = subprocess.run(
    [rscript,str(r_path),str(model_csv),str(EDA_RESULTS),str(RLIB)],
    cwd=ROOT,capture_output=True,text=True
)
(EDA_RESULTS/"62_R_stdout.txt").write_text(proc.stdout or "",encoding="utf-8")
(EDA_RESULTS/"62_R_stderr.txt").write_text(proc.stderr or "",encoding="utf-8")
if proc.returncode != 0:
    print(proc.stdout)
    print(proc.stderr)
    raise RuntimeError("R stage failed. See EDA/Results/62_R_stderr.txt")

# Interaction figure
int_file = EDA_RESULTS/"62_AG_component_interaction_map.csv"
if int_file.exists():
    ii = pd.read_csv(int_file)
    p = ii[(ii["period"]=="2005-2008") & (ii["outcome"]=="SOMATIC_SCORE")].copy()
    if not p.empty:
        bm = p.pivot(index="A_component",columns="G_component",values="beta").reindex(A_PCS).reindex(columns=G_PCS)
        pm = p.pivot(index="A_component",columns="G_component",values="BH_p").reindex(A_PCS).reindex(columns=G_PCS)
        fig, ax = plt.subplots(figsize=(7,6))
        im = ax.imshow(bm.to_numpy(float))
        ax.set_xticks(range(3)); ax.set_xticklabels(["G-PC1","G-PC2","G-PC3"])
        ax.set_yticks(range(3)); ax.set_yticklabels(["A-PC1","A-PC2","A-PC3"])
        ax.set_xlabel("Frozen glycaemic component")
        ax.set_ylabel("Frozen haematological component")
        ax.set_title("A×G component interaction map: somatic PHQ, 2005–08")
        cb = fig.colorbar(im,ax=ax); cb.set_label("Interaction coefficient")
        for i in range(3):
            for j in range(3):
                if pd.notna(bm.iloc[i,j]):
                    ax.text(j,i,f"β={bm.iloc[i,j]:.3f}\nBH p={pm.iloc[i,j]:.3g}",
                            ha="center",va="center")
        fig.tight_layout()
        fig.savefig(EDA_FIGURES/"62_AG_component_interaction_map.png",dpi=300)
        plt.close(fig)

att_file = EDA_RESULTS/"62_matched_block_attenuation.csv"
if att_file.exists():
    aa = pd.read_csv(att_file)
    focus = aa[aa["component"].isin([A_PCS[2],G_PCS[0],G_PCS[1]])].copy()
    if not focus.empty:
        focus["rank_abs_attenuation"] = focus["absolute_magnitude_attenuation_percent"].abs()
        focus.sort_values("rank_abs_attenuation",ascending=False).to_csv(
            EDA_RESULTS/"62_ranked_attenuation_focus.csv",index=False
        )

manifest = {
    "script":"62_ddx_interaction_confounder_audit.py",
    "purpose":"DDx of component meaning, confounding/explanatory attenuation, A×G component interactions, and RBC-linked HbA1c coupling.",
    "A_components":A_PCS,
    "G_components":G_PCS,
    "primary_period":"2005-2008",
    "interaction_replication_period":"2009-2018",
    "valid_explanatory_blocks":valid_blocks,
    "causal_boundary":"Repeated cross-sectional NHANES cannot identify causal effects. These are causal-diagnostic/falsification analyses only.",
    "sleep_warning":"Sleep is sensitivity-only because somatic PHQ contains a sleep item.",
    "iron_subset_note":"Ferritin/TFR/iron are restored if available but reserved for a separate women 20-49 mechanism substudy because NHANES eligibility is restricted.",
    "closest_paper_comparators_restored_if_available":["HRR","TYG"],
    "locked_outputs_modified":False,
}
(EDA_RESULTS/"62_ddx_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

print("PASS  DDx interaction/confounder audit complete.")
print("PASS  No PCA refit; locked Results untouched.")
print("PASS  Confounder/covariate vector atlas written.")
print("PASS  Matched-sample block attenuation written.")
print("PASS  3x3 A-PC x G-PC interactions tested in 2005-08 and 2009-18.")
print("PASS  RBC-linked HbA1c/FPG/insulin coupling fingerprint written.")
print(f"PASS  Valid blocks: {valid_blocks}")
print("NEXT  Inspect the 62_* outputs before adding more models.")
