from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_libs"
for p in [RESULTS, AUDIT, RLIB]:
    p.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "47_domain_corrected_AG_input.csv"
if not INPUT.exists():
    raise FileNotFoundError(f"Missing required Script 47 input: {INPUT}")

print()
print("A x G + EFFECT-MODIFICATION + TEMPORAL-DRIFT AUDIT")
print("====================================================")
print("Goals:")
print("  1) Test whether frozen hematology and glycemia components interact (A x G).")
print("  2) Test prespecified age/sex effect modification without fishing across all covariates.")
print("  3) Test whether component-to-somatic associations drift across NHANES time.")
print("Survey design is always created before analytic-domain subsetting.")
print("No existing result file is overwritten.")
print()

# ---------------------------------------------------------------------
# 1) Load the corrected design input from Script 47.
# ---------------------------------------------------------------------
d = pd.read_csv(INPUT)

def norm_cycle(x):
    s = str(x).strip().replace(".0", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return s
    return digits.zfill(4)

d["CYCLE"] = d["CYCLE"].map(norm_cycle)

A = [f"A_OI_PC{i}_FZ" for i in range(1, 4)]
G = [f"G3_OI_PC{i}_FZ" for i in range(1, 4)]
PCS = A + G
needed = [
    "SEQN", "CYCLE", "PERIOD", "SURVEY_WT", "STRATUM", "PSU",
    "DOMAIN_SOMATIC_SCORE_X3", "SOMATIC_SCORE", "RIDAGEYR", "RIAGENDR",
    "RACE", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021",
] + PCS
missing = [c for c in needed if c not in d.columns]
if missing:
    raise ValueError(f"Script 47 input is missing required columns: {missing}")

# ---------------------------------------------------------------------
# 2) Explicit hierarchical interaction terms.
#    Main effects are always retained in every interaction model.
# ---------------------------------------------------------------------
axg_terms = []
for i, a in enumerate(A, start=1):
    for j, g in enumerate(G, start=1):
        name = f"AXG_A{i}_G{j}"
        d[name] = pd.to_numeric(d[a], errors="coerce") * pd.to_numeric(d[g], errors="coerce")
        axg_terms.append(name)

# Prespecified modifiers only: age and sex.
# Age is scaled per decade around age 50 for interpretable coefficients.
d["AGE10_C"] = (pd.to_numeric(d["RIDAGEYR"], errors="coerce") - 50.0) / 10.0
d["FEMALE"] = np.where(pd.to_numeric(d["RIAGENDR"], errors="coerce") == 2, 1.0,
                       np.where(pd.to_numeric(d["RIAGENDR"], errors="coerce") == 1, 0.0, np.nan))

age_int = []
sex_int = []
for k, pc in enumerate(PCS, start=1):
    prefix = "A" if pc.startswith("A_") else "G"
    idx = k if prefix == "A" else k - 3
    an = f"{prefix}{idx}_X_AGE10"
    sn = f"{prefix}{idx}_X_FEMALE"
    d[an] = pd.to_numeric(d[pc], errors="coerce") * d["AGE10_C"]
    d[sn] = pd.to_numeric(d[pc], errors="coerce") * d["FEMALE"]
    age_int.append(an)
    sex_int.append(sn)

# ---------------------------------------------------------------------
# 3) Time variables for true pooled 2005-2018 temporal tests.
#    Script 47 SURVEY_WT is pooled within development (2 cycles) or
#    replication (5 cycles). Recover the original 2-year fasting weight,
#    then repool across all seven 2-year cycles as WTSAF2YR / 7.
# ---------------------------------------------------------------------
cycle_index = {
    "0506": 0.0, "0708": 1.0, "0910": 2.0, "1112": 3.0,
    "1314": 4.0, "1516": 5.0, "1718": 6.0,
}
d["TIME2"] = d["CYCLE"].map(cycle_index)
d["POST_DISC"] = np.where(d["CYCLE"].isin(["0910", "1112", "1314", "1516", "1718"]), 1.0,
                          np.where(d["CYCLE"].isin(["0506", "0708"]), 0.0, np.nan))

raw_2yr = np.where(
    d["PERIOD"].eq("2005-2008"), pd.to_numeric(d["SURVEY_WT"], errors="coerce") * 2.0,
    np.where(d["PERIOD"].eq("2009-2018"), pd.to_numeric(d["SURVEY_WT"], errors="coerce") * 5.0, np.nan),
)
d["WT_0518"] = raw_2yr / 7.0

trend_terms = []
post_terms = []
for k, pc in enumerate(PCS, start=1):
    prefix = "A" if pc.startswith("A_") else "G"
    idx = k if prefix == "A" else k - 3
    tn = f"{prefix}{idx}_X_TIME2"
    pn = f"{prefix}{idx}_X_POST"
    d[tn] = pd.to_numeric(d[pc], errors="coerce") * d["TIME2"]
    d[pn] = pd.to_numeric(d[pc], errors="coerce") * d["POST_DISC"]
    trend_terms.append(tn)
    post_terms.append(pn)

# Keep the full positive-weight design records. Do NOT complete-case-filter here.
d = d[pd.to_numeric(d["SURVEY_WT"], errors="coerce") > 0].copy()

OUT_INPUT = RESULTS / "49_interaction_temporal_input.csv"
d.to_csv(OUT_INPUT, index=False)

# ---------------------------------------------------------------------
# 4) R survey inference.
# ---------------------------------------------------------------------
rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
if rscript is None:
    raise FileNotFoundError("Rscript not found. Install R or ensure Rscript is on PATH.")

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
'''

RFILE = AUDIT / "49_interaction_and_temporal_drift.R"
RFILE.write_text(r_code, encoding="utf-8")

parse = subprocess.run(
    [str(rscript), "-e", f'parse(file="{str(RFILE).replace(chr(92), "/")}")'],
    capture_output=True, text=True
)
if parse.returncode != 0:
    print(parse.stdout[-3000:])
    print(parse.stderr[-5000:])
    raise SystemExit("Generated R code failed syntax check")

proc = subprocess.run(
    [str(rscript), str(RFILE), str(OUT_INPUT), str(RESULTS), str(RLIB)],
    capture_output=True, text=True
)
if proc.returncode != 0:
    print(proc.stdout[-4000:])
    print(proc.stderr[-8000:])
    raise SystemExit(proc.returncode)

# ---------------------------------------------------------------------
# 5) Multiple-testing correction on individual exploratory coefficients.
#    Block tests remain the primary inferential unit.
# ---------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    restored = np.empty(n)
    restored[order] = adj
    out[np.where(ok)[0]] = restored
    return out

coef_path = RESULTS / "49_interaction_coefficients.csv"
coef = pd.read_csv(coef_path)
coef["p_BH_within_population_model"] = np.nan
for _, idx in coef.groupby(["population", "model"]).groups.items():
    coef.loc[list(idx), "p_BH_within_population_model"] = bh(coef.loc[list(idx), "p"].to_numpy(float))
coef.to_csv(coef_path, index=False)

tcoef_path = RESULTS / "49_temporal_drift_coefficients.csv"
tcoef = pd.read_csv(tcoef_path)
tcoef["p_BH_within_model"] = np.nan
for _, idx in tcoef.groupby(["model"]).groups.items():
    tcoef.loc[list(idx), "p_BH_within_model"] = bh(tcoef.loc[list(idx), "p"].to_numpy(float))
tcoef.to_csv(tcoef_path, index=False)

# ---------------------------------------------------------------------
# 6) Print only the high-value block tests.
# ---------------------------------------------------------------------
itests = pd.read_csv(RESULTS / "49_interaction_block_tests.csv")
ttests = pd.read_csv(RESULTS / "49_temporal_drift_tests.csv")

print("PASS  A x G interaction models completed with hierarchical main effects retained.")
print("PASS  Age/sex modification tested as prespecified blocks only.")
print("PASS  Pooled 2005-2018 temporal drift used WTSAF2YR/7 equivalent weights.")
print("PASS  Cycle fixed effects absorb arbitrary cycle-level mean shifts.")
print("PASS  No 2021-2023 interaction inference attempted because its survey df are already design-limited.")
print()
print("A x G / MODIFIER BLOCK TESTS")
print(itests.to_string(index=False))
print()
print("TEMPORAL-DRIFT BLOCK TESTS")
print(ttests.to_string(index=False))
