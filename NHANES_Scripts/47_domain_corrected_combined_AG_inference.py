from pathlib import Path
import json
import re
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

for p in [PROCESSED, RESULTS, AUDIT, RLIB]:
    p.mkdir(parents=True, exist_ok=True)

A_DISC_FILE = PROCESSED / "46_outcome_independent_A_discovery_scores.parquet"
G_DISC_FILE = PROCESSED / "46_outcome_independent_G3_discovery_scores.parquet"
A_TEMP_FILE = PROCESSED / "46_outcome_independent_A_temporal_scores.parquet"
G_TEMP_FILE = PROCESSED / "46_outcome_independent_G3_temporal_scores.parquet"
for p in [A_DISC_FILE, G_DISC_FILE, A_TEMP_FILE, G_TEMP_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"Run script 46 first: missing {p}")

DISC = {"0506": "D", "0708": "E"}
TEMP = {"0910": "F", "1112": "G", "1314": "H", "1516": "I", "1718": "J", "2123": "L"}
PHQ = [f"DPQ0{i}0" for i in range(1, 10)]
SOMATIC = ["DPQ030", "DPQ040", "DPQ050"]
X3 = ["RIDAGEYR", "RIAGENDR", "RACE", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"]
X0 = ["RIDAGEYR", "RIAGENDR", "RACE"]
A_PCS = [f"A_OI_PC{i}" for i in range(1, 4)]
G_PCS = [f"G3_OI_PC{i}" for i in range(1, 4)]

print()
print("DOMAIN-CORRECTED COMBINED A+G INFERENCE")
print("=========================================")
print("Survey design is built on the full positive-weight fasting sample FIRST.")
print("Analytic complete cases are then selected with survey::subset().")
print("Representation scores come only from the outcome-independent script 46 refreeze.")
print("Existing 44/45 outputs are not modified.")
print()


def xpt_path(base: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")


def base_for(cycle):
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def suffix_for(cycle):
    return (DISC if cycle in DISC else TEMP)[cycle]


def clean_phq(dpq):
    d = dpq[["SEQN"] + PHQ].copy()
    for c in PHQ:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[d[c].abs() < 1e-10, c] = 0
        d.loc[~d[c].isin([0, 1, 2, 3]), c] = np.nan
    d["PHQ9_TOTAL"] = d[PHQ].sum(axis=1, min_count=9)
    d["SOMATIC_SCORE"] = d[SOMATIC].sum(axis=1, min_count=3)
    d["COGAFF_SUM"] = d["PHQ9_TOTAL"] - d["SOMATIC_SCORE"]
    return d


def educ3(x):
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out[x.isin([1, 2])] = 1
    out[x == 3] = 2
    out[x.isin([4, 5])] = 3
    return out


def smoking3(df):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out[df["SMQ020"] == 2] = 0
    out[(df["SMQ020"] == 1) & (df["SMQ040"] == 3)] = 1
    out[(df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))] = 2
    return out


def egfr_2021(scr, age, sex):
    scr = pd.to_numeric(scr, errors="coerce").to_numpy(float)
    age = pd.to_numeric(age, errors="coerce").to_numpy(float)
    sex = pd.to_numeric(sex, errors="coerce").to_numpy(float)
    female = sex == 2
    k = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / k
    return 142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1.0)


def build_full_fasting_design_frame(cycle):
    """
    Build ALL positive-fasting-weight records first. Nothing is dropped for
    PHQ, exposure, representation or covariate missingness here.
    """
    s = suffix_for(cycle)
    base = base_for(cycle)
    demo = pd.read_sas(xpt_path(base, f"DEMO_{s}"), format="xport")
    glu = pd.read_sas(xpt_path(base, f"GLU_{s}"), format="xport")
    dpq = pd.read_sas(xpt_path(base, f"DPQ_{s}"), format="xport")
    cbc = pd.read_sas(xpt_path(base, f"CBC_{s}"), format="xport")
    ghb = pd.read_sas(xpt_path(base, f"GHB_{s}"), format="xport")
    bmx = pd.read_sas(xpt_path(base, f"BMX_{s}"), format="xport")
    bio = pd.read_sas(xpt_path(base, f"BIOPRO_{s}"), format="xport")
    smq = pd.read_sas(xpt_path(base, f"SMQ_{s}"), format="xport")

    required_demo = ["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "DMDEDUC2", "SDMVPSU", "SDMVSTRA"]
    missing = [c for c in required_demo if c not in demo.columns]
    if missing:
        raise ValueError(f"{cycle} DEMO missing {missing}")
    if "WTSAF2YR" not in glu.columns:
        raise ValueError(f"{cycle} GLU missing WTSAF2YR")

    race = "RIDRETH1" if "RIDRETH1" in demo.columns else ("RIDRETH3" if "RIDRETH3" in demo.columns else None)
    if race is None:
        raise ValueError(f"{cycle} has no race/ethnicity field")

    demo_cols = required_demo + [race] + (["RIDEXPRG"] if "RIDEXPRG" in demo.columns else [])
    d = demo[demo_cols].copy().rename(columns={race: "RACE_RAW"})
    d["RACE_SOURCE"] = race
    # Harmonize RIDRETH3 to the older 5-level RIDRETH1 scheme so pooled
    # 2009-2018 categories retain the same meaning across cycles.
    d["RACE"] = pd.to_numeric(d["RACE_RAW"], errors="coerce")
    if race == "RIDRETH3":
        d["RACE"] = d["RACE"].replace({6: 5, 7: 5})

    # Fasting sample design root. This is the ONLY pre-design restriction.
    d = d.merge(glu[["SEQN", "WTSAF2YR"]], on="SEQN", how="left", validate="one_to_one")
    d["WTSAF2YR"] = pd.to_numeric(d["WTSAF2YR"], errors="coerce")
    d = d[d["WTSAF2YR"] > 0].copy()

    # All subsequent files are LEFT joins so missingness remains represented
    # in the survey design and is handled only by the domain indicator.
    d = d.merge(clean_phq(dpq), on="SEQN", how="left", validate="one_to_one")
    d = d.merge(cbc[[c for c in ["SEQN", "LBXHGB"] if c in cbc.columns]], on="SEQN", how="left", validate="one_to_one")
    d = d.merge(ghb[[c for c in ["SEQN", "LBXGH"] if c in ghb.columns]], on="SEQN", how="left", validate="one_to_one")
    d = d.merge(bmx[[c for c in ["SEQN", "BMXBMI"] if c in bmx.columns]], on="SEQN", how="left", validate="one_to_one")
    d = d.merge(bio[[c for c in ["SEQN", "LBXSCR"] if c in bio.columns]], on="SEQN", how="left", validate="one_to_one")
    d = d.merge(smq[[c for c in ["SEQN", "SMQ020", "SMQ040"] if c in smq.columns]], on="SEQN", how="left", validate="one_to_one")

    d["CYCLE"] = cycle
    d["PERIOD"] = (
        "2005-2008" if cycle in DISC else
        "2021-2023" if cycle == "2123" else
        "2009-2018"
    )
    d["EDUC3"] = educ3(d["DMDEDUC2"])
    d["SMOKING3"] = smoking3(d)
    d["EGFR_2021"] = egfr_2021(d["LBXSCR"], d["RIDAGEYR"], d["RIAGENDR"])
    d["HB_THRESHOLD"] = np.where(d["RIAGENDR"] == 1, 13.0, np.where(d["RIAGENDR"] == 2, 12.0, np.nan))
    d["A"] = (d["HB_THRESHOLD"] - pd.to_numeric(d["LBXHGB"], errors="coerce")).clip(lower=0)
    d["G_HBA1C"] = (pd.to_numeric(d["LBXGH"], errors="coerce") - 5.7).clip(lower=0)

    age_ok = pd.to_numeric(d["RIDAGEYR"], errors="coerce") >= 20
    preg_ok = pd.Series(True, index=d.index)
    if "RIDEXPRG" in d.columns:
        preg_ok = d["RIDEXPRG"] != 1
    d["ELIGIBLE_ADULT_NONPREG"] = (age_ok & preg_ok).astype(int)

    d["SURVEY_WT"] = d["WTSAF2YR"]
    if cycle in DISC:
        d["SURVEY_WT"] = d["SURVEY_WT"] / 2.0
    elif cycle != "2123":
        d["SURVEY_WT"] = d["SURVEY_WT"] / 5.0

    d["STRATUM"] = d["CYCLE"].astype(str) + "_" + d["SDMVSTRA"].astype("Int64").astype(str)
    d["PSU"] = d["STRATUM"] + "_" + d["SDMVPSU"].astype("Int64").astype(str)
    return d


# Load NEW outcome-independent scores.
a_disc = pd.read_parquet(A_DISC_FILE)
g_disc = pd.read_parquet(G_DISC_FILE)
a_temp = pd.read_parquet(A_TEMP_FILE)
g_temp = pd.read_parquet(G_TEMP_FILE)

# Build full survey frames independently of analysis completeness.
frames = {cy: build_full_fasting_design_frame(cy) for cy in list(DISC) + list(TEMP)}
full = pd.concat(frames.values(), ignore_index=True)

# Left-merge frozen representation scores. Do not restrict the survey frame.
a_all = pd.concat([
    a_disc[["SEQN", "CYCLE"] + A_PCS],
    a_temp[["SEQN", "CYCLE"] + A_PCS],
], ignore_index=True)
g_all = pd.concat([
    g_disc[["SEQN", "CYCLE"] + G_PCS],
    g_temp[["SEQN", "CYCLE"] + G_PCS],
], ignore_index=True)

full = full.merge(a_all, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")
full = full.merge(g_all, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

# Discovery-frozen scaling is learned on the JOINT PHYSIOLOGY sample only,
# before any PHQ/covariate restriction. This is not outcome conditioning.
disc_phys = full[
    full["CYCLE"].isin(DISC)
    & full["ELIGIBLE_ADULT_NONPREG"].eq(1)
].dropna(subset=A_PCS + G_PCS + ["SURVEY_WT"]).copy()
disc_phys = disc_phys[disc_phys["SURVEY_WT"] > 0].copy()

scale_rows = []
A_Z, G_Z = [], []
for comp in A_PCS + G_PCS:
    x = pd.to_numeric(disc_phys[comp], errors="coerce").to_numpy(float)
    w = disc_phys["SURVEY_WT"].to_numpy(float)
    sw = w.sum()
    mu = np.sum(w * x) / sw
    sd = np.sqrt(np.sum(w * (x - mu) ** 2) / sw)
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Invalid frozen scale for {comp}")
    z = comp + "_FZ"
    full[z] = (pd.to_numeric(full[comp], errors="coerce") - mu) / sd
    scale_rows.append({"component": comp, "weighted_mean": mu, "weighted_sd": sd, "z_column": z})
    (A_Z if comp.startswith("A_") else G_Z).append(z)

pd.DataFrame(scale_rows).to_csv(RESULTS / "47_domain_corrected_component_scaling.csv", index=False)

# Outcome-specific domains are applied AFTER svydesign() in R.
# The primary somatic model therefore does not unnecessarily require the
# six non-somatic PHQ items to be observed.
base_rep_needed = ["A", "G_HBA1C"] + A_Z + G_Z
for adj, covars in [("X3", X3), ("X0", X0)]:
    for outcome in ["SOMATIC_SCORE", "PHQ9_TOTAL", "COGAFF_SUM"]:
        needed = [outcome] + base_rep_needed + covars
        full[f"DOMAIN_{outcome}_{adj}"] = (
            full["ELIGIBLE_ADULT_NONPREG"].eq(1)
            & full[needed].notna().all(axis=1)
        ).astype(int)

# Flow audit: the design sample is larger than the analytic domain by construction.
flow_rows = []
for period in ["2005-2008", "2009-2018", "2021-2023"]:
    q = full[full["PERIOD"].eq(period)]
    flow_rows.append({
        "period": period,
        "positive_fasting_weight_design_n": len(q),
        "adult_nonpreg_n": int(q["ELIGIBLE_ADULT_NONPREG"].sum()),
        "somatic_X3_domain_n": int(q["DOMAIN_SOMATIC_SCORE_X3"].sum()),
        "phq9_X3_domain_n": int(q["DOMAIN_PHQ9_TOTAL_X3"].sum()),
        "cogaff_X3_domain_n": int(q["DOMAIN_COGAFF_SUM_X3"].sum()),
        "somatic_X0_domain_n": int(q["DOMAIN_SOMATIC_SCORE_X0"].sum()),
        "design_psu_n": int(q["PSU"].nunique()),
        "design_strata_n": int(q["STRATUM"].nunique()),
    })
pd.DataFrame(flow_rows).to_csv(RESULTS / "47_domain_corrected_sample_flow.csv", index=False)

keep = [
    "SEQN", "CYCLE", "PERIOD", "SURVEY_WT", "SDMVPSU", "SDMVSTRA", "STRATUM", "PSU",
    "ELIGIBLE_ADULT_NONPREG",
    "DOMAIN_SOMATIC_SCORE_X3", "DOMAIN_PHQ9_TOTAL_X3", "DOMAIN_COGAFF_SUM_X3",
    "DOMAIN_SOMATIC_SCORE_X0", "DOMAIN_PHQ9_TOTAL_X0", "DOMAIN_COGAFF_SUM_X0",
    "PHQ9_TOTAL", "SOMATIC_SCORE", "COGAFF_SUM", "A", "G_HBA1C",
] + X3 + A_PCS + G_PCS + A_Z + G_Z
input_csv = RESULTS / "47_domain_corrected_AG_input.csv"
full[keep].to_csv(input_csv, index=False)

# ------------------------------------------------------------------
# R survey inference: DESIGN FIRST, DOMAIN SECOND.
# ------------------------------------------------------------------
rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])
if rscript is None:
    raise FileNotFoundError("Rscript not found")

A_VEC = "c(" + ",".join(json.dumps(x) for x in A_Z) + ")"
G_VEC = "c(" + ",".join(json.dumps(x) for x in G_Z) + ")"

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

A_PCS <- __A_VEC__
G_PCS <- __G_VEC__
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
'''
r_code = r_code.replace("__A_VEC__", A_VEC).replace("__G_VEC__", G_VEC)
r_path = AUDIT / "47_domain_corrected_AG.R"
r_path.write_text(r_code, encoding="utf-8")

r_path_r = str(r_path).replace("\\", "/")
parse = subprocess.run([str(rscript), "-e", f'parse(file="{r_path_r}")'], capture_output=True, text=True)
if parse.returncode != 0:
    print(parse.stdout[-3000:]); print(parse.stderr[-5000:])
    raise SystemExit("Generated R code failed syntax check")

proc = subprocess.run([str(rscript), str(r_path), str(input_csv), str(RESULTS), str(RLIB)], capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stdout[-4000:]); print(proc.stderr[-7000:])
    raise SystemExit(proc.returncode)

coef = pd.read_csv(RESULTS / "47_domain_corrected_coefficients.csv")
tests = pd.read_csv(RESULTS / "47_domain_corrected_block_tests.csv")
r2 = pd.read_csv(RESULTS / "47_domain_corrected_model_R2.csv")

# BH correction within population/adjustment/outcome/model component families.
def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return out
    order = np.argsort(pv); ranked = pv[order]; n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    restored = np.empty(n); restored[order] = adj
    out[np.where(ok)[0]] = restored
    return out

coef["p_BH_component_family"] = np.nan
for _, idx in coef.groupby(["population","adjustment","outcome","model"]).groups.items():
    comp = [i for i in idx if "_OI_PC" in str(coef.loc[i, "term"])]
    if comp:
        coef.loc[comp, "p_BH_component_family"] = bh(coef.loc[comp, "p"].to_numpy(float))
coef.to_csv(RESULTS / "47_domain_corrected_coefficients.csv", index=False)

# Incremental-information summary.
delta_rows = []
for (pop, adj, outcome), q in r2.groupby(["population","adjustment","outcome"]):
    v = dict(zip(q["model"], q["weighted_R2"]))
    delta_rows.append({
        "population": pop, "adjustment": adj, "outcome": outcome,
        "n": int(q["n"].iloc[0]),
        "delta_scalar_model_beyond_X": v["X_scalarA_scalarG"] - v["X"],
        "delta_all_PCs_beyond_X": v["X_Apcs_Gpcs"] - v["X"],
        "delta_all_PCs_beyond_scalar_model": v["X_scalars_Apcs_Gpcs"] - v["X_scalarA_scalarG"],
        "delta_scalars_beyond_all_PCs": v["X_scalars_Apcs_Gpcs"] - v["X_Apcs_Gpcs"],
        "delta_Apcs_scalarG_beyond_scalars": v["X_Apcs_scalarG"] - v["X_scalarA_scalarG"],
        "delta_Ascalar_Gpcs_beyond_scalars": v["X_scalarA_Gpcs"] - v["X_scalarA_scalarG"],
    })
pd.DataFrame(delta_rows).to_csv(RESULTS / "47_domain_corrected_incremental_information.csv", index=False)

# Old-vs-corrected comparison where labels match conceptually.
comparison_rows = []
old_paths = [RESULTS / "44_final_AG_block_tests.csv", RESULTS / "45_temporal_AG_block_tests.csv"]
for old_path in old_paths:
    if not old_path.exists():
        continue
    old = pd.read_csv(old_path)
    for _, nr in tests[tests["outcome"].eq("SOMATIC_SCORE")].iterrows():
        pop = nr["population"]
        if "44_" in old_path.name and pop != "2005-2008":
            continue
        if "45_" in old_path.name and pop == "2005-2008":
            continue
        oq = old[old["outcome"].eq("SOMATIC_SCORE") & old["test"].eq(nr["test"])]
        if "population" in old.columns:
            oq = oq[oq["population"].astype(str).eq(pop)]
        if "adjustment" in old.columns:
            oq = oq[oq["adjustment"].astype(str).eq(str(nr["adjustment"]))]
        if len(oq) == 1:
            comparison_rows.append({
                "population": pop, "adjustment": nr["adjustment"], "test": nr["test"],
                "old_p": float(oq.iloc[0]["p"]) if pd.notna(oq.iloc[0]["p"]) else np.nan,
                "corrected_p": nr["p"],
                "old_df_den": float(oq.iloc[0]["df_den"]) if "df_den" in oq and pd.notna(oq.iloc[0]["df_den"]) else np.nan,
                "corrected_df_den": nr["df_den"],
            })
if comparison_rows:
    pd.DataFrame(comparison_rows).to_csv(RESULTS / "47_old_vs_domain_corrected_tests.csv", index=False)

# Audit manifest.
audit = {
    "script": "47_domain_corrected_combined_AG_inference.py",
    "representation_source": "script 46 outcome-independent frozen scores",
    "survey_design_order": "positive fasting-weight sample -> svydesign -> survey::subset analytic domain -> svyglm",
    "pre_design_complete_case_filtering": False,
    "discovery_weight": "WTSAF2YR / 2",
    "2009_2018_weight": "WTSAF2YR / 5",
    "2021_2023_weight": "WTSAF2YR",
    "primary_adjustment": "X3",
    "modern_sensitivity": "X0",
    "old_44_45_outputs_overwritten": False,
}
(AUDIT / "47_domain_corrected_inference.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("PASS  Full positive-weight survey designs were created before domain subsetting.")
print("PASS  Corrected discovery/replication/modern-stress-test outputs saved under prefix 47.")
print()
show = tests[(tests["outcome"] == "SOMATIC_SCORE") & (tests["test"] == "all_PCs_extra_beyond_scalar_A_scalar_G_X")]
print(show[["population","adjustment","F","df_num","df_den","p","design_df","inference_status"]].to_string(index=False))
