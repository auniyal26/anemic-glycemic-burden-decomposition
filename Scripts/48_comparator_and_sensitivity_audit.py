from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"
for p in [RESULTS, AUDIT, RLIB]:
    p.mkdir(parents=True, exist_ok=True)

BASE = RESULTS / "47_domain_corrected_AG_input.csv"
A_DISC = PROCESSED / "46_outcome_independent_A_discovery_scores.parquet"
A_TEMP = PROCESSED / "46_outcome_independent_A_temporal_scores.parquet"
G_DISC = PROCESSED / "46_outcome_independent_G3_discovery_scores.parquet"
G_TEMP = PROCESSED / "46_outcome_independent_G3_temporal_scores.parquet"
for p in [BASE, A_DISC, A_TEMP, G_DISC, G_TEMP]:
    if not p.exists():
        raise FileNotFoundError(f"Run scripts 46 and 47 first: missing {p}")

A_RAW = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G_RAW = ["LBXGH", "LBXGLU", "LOG_IN"]
A_PCS = [f"A_OI_PC{i}_FZ" for i in range(1, 4)]
G3_PCS = [f"G3_OI_PC{i}_FZ" for i in range(1, 4)]
X3 = ["RIDAGEYR", "RIAGENDR", "RACE", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"]

print()
print("COMPARATOR + GLYCEMIC SENSITIVITY AUDIT")
print("=======================================")
print("Goals:")
print("  1) Compare threshold scalars with continuous, spline, raw multivariate, and PC representations.")
print("  2) Repeat the A+G test without insulin (HbA1c+FPG) and without HbA1c (FPG+insulin).")
print("  3) Check the primary component-vs-scalar result with quasi-Poisson somatic-score models.")
print("No existing result file is overwritten.")
print()


def weighted_pca_fit(df, variables, weight_col, positive_pc1=True):
    X = df[variables].to_numpy(float)
    w = df[weight_col].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X, w = X[ok], w[ok]
    if len(X) < 3 or w.sum() <= 0:
        raise RuntimeError(f"No valid weighted discovery rows for PCA variables {variables}")
    sw = w.sum()
    mu = np.sum(w[:, None] * X, axis=0) / sw
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise RuntimeError(f"Invalid PCA scale for {variables}")
    Z = (X - mu) / sd
    cov = (Z.T * w) @ Z / sw
    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    L = vecs[:, order]
    for j in range(L.shape[1]):
        anchor = np.argmax(np.abs(L[:, j]))
        if L[anchor, j] < 0:
            L[:, j] *= -1
    if positive_pc1 and L[:, 0].sum() < 0:
        L[:, 0] *= -1
    return {"mean": mu, "sd": sd, "L": L, "evr": vals / vals.sum()}


def weighted_mean_sd(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0 or w.sum() <= 0:
        raise RuntimeError("No valid positive-weight observations for weighted scaling")
    sw = w.sum(); mu = np.sum(w*x)/sw
    sd = np.sqrt(np.sum(w*(x-mu)**2)/sw)
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Invalid weighted scale")
    return float(mu), float(sd)


def project(raw, variables, model):
    X = raw[variables].to_numpy(float)
    Z = (X - model["mean"]) / model["sd"]
    return Z @ model["L"]


# ---------------------------------------------------------------------
# Raw marker assembly.
# CSV round-tripping converts 0506 -> 506 and 0910 -> 910, so normalize
# cycle labels before ANY merge. This was the cause of the first failed run.
# ---------------------------------------------------------------------
DISC = {"0506": "D", "0708": "E"}
TEMP = {"0910": "F", "1112": "G", "1314": "H", "1516": "I", "1718": "J", "2123": "L"}
ALL_CYCLES = {**DISC, **TEMP}
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"


def norm_cycle(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .replace({"506": "0506", "708": "0708", "910": "0910"})
    )


def xpt_path(base_dir: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        path = base_dir / name
        if path.exists():
            return path
    matches = list(base_dir.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base_dir}")


def cycle_base(cycle: str) -> Path:
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def insulin_path(cycle: str) -> Path:
    s = ALL_CYCLES[cycle]
    stem = f"GLU_{s}" if cycle in {"0506", "0708", "0910", "1112"} else f"INS_{s}"
    return xpt_path(cycle_base(cycle), stem)


def read_raw_markers(cycle: str) -> pd.DataFrame:
    """Read raw physiology without requiring any outcome or cross-lab completeness."""
    s = ALL_CYCLES[cycle]
    b = cycle_base(cycle)
    cbc = pd.read_sas(xpt_path(b, f"CBC_{s}"), format="xport")
    ghb = pd.read_sas(xpt_path(b, f"GHB_{s}"), format="xport")
    glu = pd.read_sas(xpt_path(b, f"GLU_{s}"), format="xport")
    ins = pd.read_sas(insulin_path(cycle), format="xport")

    for name, frame, cols in [
        ("CBC", cbc, ["SEQN"] + A_RAW),
        ("GHB", ghb, ["SEQN", "LBXGH"]),
        ("GLU", glu, ["SEQN", "LBXGLU"]),
        ("INS", ins, ["SEQN", "LBXIN"]),
    ]:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{cycle} {name} missing {missing}")

    # OUTER merges are deliberate: G2A must not require insulin, and G2B
    # must not require HbA1c. Completeness is imposed only within each
    # sensitivity-specific physiological fit/domain.
    raw = cbc[["SEQN"] + A_RAW].merge(
        ghb[["SEQN", "LBXGH"]], on="SEQN", how="outer", validate="one_to_one"
    ).merge(
        glu[["SEQN", "LBXGLU"]], on="SEQN", how="outer", validate="one_to_one"
    ).merge(
        ins[["SEQN", "LBXIN"]], on="SEQN", how="outer", validate="one_to_one"
    )
    raw.loc[pd.to_numeric(raw["LBXIN"], errors="coerce") <= 0, "LBXIN"] = np.nan
    raw["LOG_IN"] = np.log(pd.to_numeric(raw["LBXIN"], errors="coerce"))
    raw["CYCLE"] = cycle
    return raw


base = pd.read_csv(BASE)
base["CYCLE"] = norm_cycle(base["CYCLE"])
base["PERIOD"] = base["PERIOD"].astype(str)

raw = pd.concat([read_raw_markers(cy) for cy in ALL_CYCLES], ignore_index=True)
raw["CYCLE"] = norm_cycle(raw["CYCLE"])
if raw.duplicated(["SEQN", "CYCLE"]).any():
    raise RuntimeError("Raw physiology contains duplicate SEQN/CYCLE rows")

# LEFT merge preserves the full positive-fasting-weight survey frame from 47.
d = base.merge(raw, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")
d["G_FPG"] = (pd.to_numeric(d["LBXGLU"], errors="coerce") - 100.0).clip(lower=0)

# Strict raw7 scaling for the SAME-SAMPLE comparator. This deliberately
# requires all 7 raw markers, but still no outcome/covariate completeness.
phys_disc_raw7 = d[
    d["PERIOD"].eq("2005-2008") & d["ELIGIBLE_ADULT_NONPREG"].eq(1)
].dropna(subset=A_RAW + G_RAW + ["SURVEY_WT"]).copy()
phys_disc_raw7 = phys_disc_raw7[phys_disc_raw7["SURVEY_WT"] > 0].copy()
if phys_disc_raw7.empty:
    raise RuntimeError("Discovery raw7 physiology sample is empty after cycle normalization")

raw_z_cols = []
raw_scale_rows = []
for var in A_RAW + G_RAW:
    mu, sd = weighted_mean_sd(phys_disc_raw7[var], phys_disc_raw7["SURVEY_WT"])
    z = var + "_DISC_Z"
    d[z] = (pd.to_numeric(d[var], errors="coerce") - mu) / sd
    raw_z_cols.append(z)
    raw_scale_rows.append({"representation": "raw7", "variable": var, "mean": mu, "sd": sd, "fit_n": len(phys_disc_raw7)})

# G2 sensitivity 1: HbA1c + fasting glucose. IMPORTANT: fit on every
# eligible discovery participant with those two markers -- insulin is NOT required.
g2a_vars = ["LBXGH", "LBXGLU"]
g2a_fit_data = d[
    d["PERIOD"].eq("2005-2008") & d["ELIGIBLE_ADULT_NONPREG"].eq(1)
].dropna(subset=g2a_vars + ["SURVEY_WT"]).copy()
g2a_fit_data = g2a_fit_data[g2a_fit_data["SURVEY_WT"] > 0].copy()
g2a_model = weighted_pca_fit(g2a_fit_data, g2a_vars, "SURVEY_WT")
score = np.full((len(d), 2), np.nan)
ok = d[g2a_vars].notna().all(axis=1)
score[ok.to_numpy(), :] = project(d.loc[ok], g2a_vars, g2a_model)
for j in range(2):
    d[f"G2A_PC{j+1}"] = score[:, j]

# G2 sensitivity 2: fasting glucose + log insulin. Fit independently of HbA1c.
g2b_vars = ["LBXGLU", "LOG_IN"]
g2b_fit_data = d[
    d["PERIOD"].eq("2005-2008") & d["ELIGIBLE_ADULT_NONPREG"].eq(1)
].dropna(subset=g2b_vars + ["SURVEY_WT"]).copy()
g2b_fit_data = g2b_fit_data[g2b_fit_data["SURVEY_WT"] > 0].copy()
g2b_model = weighted_pca_fit(g2b_fit_data, g2b_vars, "SURVEY_WT")
score = np.full((len(d), 2), np.nan)
ok = d[g2b_vars].notna().all(axis=1)
score[ok.to_numpy(), :] = project(d.loc[ok], g2b_vars, g2b_model)
for j in range(2):
    d[f"G2B_PC{j+1}"] = score[:, j]

# Scale G2 scores on their OWN outcome-independent discovery physiology
# populations, then freeze those scales for later periods.
G2A_Z, G2B_Z = [], []
for prefix, dest, fit_mask, fit_n in [
    ("G2A_PC", G2A_Z, g2a_fit_data.index, len(g2a_fit_data)),
    ("G2B_PC", G2B_Z, g2b_fit_data.index, len(g2b_fit_data)),
]:
    for j in range(1, 3):
        col = f"{prefix}{j}"
        q = d.loc[fit_mask, [col, "SURVEY_WT"]].dropna()
        mu, sd = weighted_mean_sd(q[col], q["SURVEY_WT"])
        z = col + "_FZ"
        d[z] = (d[col] - mu) / sd
        dest.append(z)
        raw_scale_rows.append({"representation": prefix, "variable": col, "mean": mu, "sd": sd, "fit_n": fit_n})

pd.DataFrame(raw_scale_rows).to_csv(RESULTS / "48_comparator_frozen_scaling.csv", index=False)

print(f"PASS  Cycle labels normalized before raw-marker merge.")
print(f"PASS  Strict raw7 discovery physiology n={len(phys_disc_raw7):,}.")
print(f"PASS  G2 no-insulin discovery fit n={len(g2a_fit_data):,} (insulin not required).")
print(f"PASS  G2 no-HbA1c discovery fit n={len(g2b_fit_data):,} (HbA1c not required).")

# Domains are analysis-specific and will be applied with survey::subset AFTER design creation.
base_common = ["SOMATIC_SCORE", "A", "G_HBA1C"] + X3

d["DOMAIN_COMPARE_X3"] = (
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[base_common + A_PCS + G3_PCS + A_RAW + G_RAW + raw_z_cols].notna().all(axis=1)
).astype(int)

d["DOMAIN_G2A_X3"] = (
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[base_common + A_PCS + G2A_Z].notna().all(axis=1)
).astype(int)

base_noa1c = ["SOMATIC_SCORE", "A", "G_FPG"] + X3

d["DOMAIN_G2B_X3"] = (
    d["ELIGIBLE_ADULT_NONPREG"].eq(1)
    & d[base_noa1c + A_PCS + G2B_Z].notna().all(axis=1)
).astype(int)

# Selection audit for final comparator sample.
flow_rows = []
for period in ["2005-2008", "2009-2018", "2021-2023"]:
    q = d[d["PERIOD"].eq(period)]
    flow_rows.append({
        "period": period,
        "positive_fasting_weight_design_n": len(q),
        "adult_nonpreg_n": int(q["ELIGIBLE_ADULT_NONPREG"].sum()),
        "main_47_somatic_X3_domain_n": int(q["DOMAIN_SOMATIC_SCORE_X3"].sum()),
        "strict_same_raw7_comparator_domain_n": int(q["DOMAIN_COMPARE_X3"].sum()),
        "G2_no_insulin_domain_n": int(q["DOMAIN_G2A_X3"].sum()),
        "G2_no_HbA1c_domain_n": int(q["DOMAIN_G2B_X3"].sum()),
    })
pd.DataFrame(flow_rows).to_csv(RESULTS / "48_sensitivity_sample_flow.csv", index=False)

# Included/excluded descriptives for the MAIN 47 X3 domain, using available values.
desc_rows = []
for period in ["2005-2008", "2009-2018", "2021-2023"]:
    q = d[d["PERIOD"].eq(period)].copy()
    for grp, flag in [("included", 1), ("excluded", 0)]:
        z = q[q["DOMAIN_SOMATIC_SCORE_X3"].eq(flag)]
        w = pd.to_numeric(z["SURVEY_WT"], errors="coerce")
        for var in ["RIDAGEYR", "BMXBMI", "LBXHGB", "LBXGH", "LBXGLU"]:
            x = pd.to_numeric(z[var], errors="coerce")
            ok = x.notna() & w.notna() & (w > 0)
            mean = np.sum(w[ok] * x[ok]) / np.sum(w[ok]) if ok.any() else np.nan
            desc_rows.append({
                "period": period, "group": grp, "variable": var,
                "n_observed": int(ok.sum()), "weighted_mean": mean,
                "missing_fraction": float(x.isna().mean()) if len(x) else np.nan,
            })
pd.DataFrame(desc_rows).to_csv(RESULTS / "48_included_excluded_weighted_descriptives.csv", index=False)

keep = list(dict.fromkeys([
    "SEQN","CYCLE","PERIOD","SURVEY_WT","STRATUM","PSU","ELIGIBLE_ADULT_NONPREG",
    "DOMAIN_SOMATIC_SCORE_X3","DOMAIN_COMPARE_X3","DOMAIN_G2A_X3","DOMAIN_G2B_X3",
    "PHQ9_TOTAL","SOMATIC_SCORE","COGAFF_SUM","A","G_HBA1C","G_FPG",
] + X3 + A_PCS + G3_PCS + A_RAW + G_RAW + raw_z_cols + G2A_Z + G2B_Z))
input_csv = RESULTS / "48_comparator_sensitivity_input.csv"
d[keep].to_csv(input_csv, index=False)

# Save sensitivity transforms for exact reproducibility.
param = {
    "G2A_no_insulin": {
        "variables": g2a_vars,
        "mean": g2a_model["mean"].tolist(), "sd": g2a_model["sd"].tolist(),
        "loadings": g2a_model["L"].tolist(), "evr": g2a_model["evr"].tolist(),
    },
    "G2B_no_HbA1c": {
        "variables": g2b_vars,
        "mean": g2b_model["mean"].tolist(), "sd": g2b_model["sd"].tolist(),
        "loadings": g2b_model["L"].tolist(), "evr": g2b_model["evr"].tolist(),
    },
}
(AUDIT / "48_glycemic_sensitivity_transforms.json").write_text(json.dumps(param, indent=2), encoding="utf-8")

rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])
if rscript is None:
    raise FileNotFoundError("Rscript not found")

vec = lambda xs: "c(" + ",".join(json.dumps(x) for x in xs) + ")"
repl = {
    "__A__": vec(A_PCS), "__G3__": vec(G3_PCS), "__RAW7__": vec(raw_z_cols),
    "__G2A__": vec(G2A_Z), "__G2B__": vec(G2B_Z),
}

r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]; results_dir <- args[2]; user_lib <- args[3]
dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib,.libPaths()))
if (!requireNamespace("survey", quietly=TRUE)) install.packages("survey",repos="https://cloud.r-project.org",lib=user_lib)
library(survey); library(splines)
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])
A <- __A__; G3 <- __G3__; RAW7 <- __RAW7__; G2A <- __G2A__; G2B <- __G2B__
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
'''
for k, v in repl.items():
    r_code = r_code.replace(k, v)

r_path = AUDIT / "48_comparator_sensitivity.R"
r_path.write_text(r_code, encoding="utf-8")
r_path_r = str(r_path).replace("\\", "/")
parse = subprocess.run([str(rscript), "-e", f'parse(file="{r_path_r}")'], capture_output=True, text=True)
if parse.returncode != 0:
    print(parse.stdout[-3000:]); print(parse.stderr[-5000:])
    raise SystemExit("Generated R script failed syntax check")
proc = subprocess.run([str(rscript), str(r_path), str(input_csv), str(RESULTS), str(RLIB)], capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stdout[-4000:]); print(proc.stderr[-7000:])
    raise SystemExit(proc.returncode)

# Compact machine-readable audit.
audit = {
    "script": "48_comparator_and_sensitivity_audit.py",
    "comparators_same_sample": [
        "threshold A + threshold HbA1c G",
        "continuous Hb + HbA1c",
        "natural splines Hb + HbA1c (3 df each)",
        "raw standardized 7-marker multivariate model",
        "frozen A3 + G3 component model",
    ],
    "glycemic_sensitivities": {
        "G2_no_insulin": "HbA1c + fasting glucose, outcome-independent frozen PCA",
        "G2_no_HbA1c": "fasting glucose + log insulin, outcome-independent frozen PCA",
    },
    "outcome_robustness": "survey quasi-Poisson log-link sensitivity for SOMATIC_SCORE",
    "survey_domain_order": "design first; subset domain second",
    "cycle_normalization_before_raw_merge": True,
    "G2A_fit_does_not_require_insulin": True,
    "G2B_fit_does_not_require_HbA1c": True,
    "overwrites_prior_outputs": False,
}
(AUDIT / "48_comparator_sensitivity.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print("PASS  Same-sample representation comparators completed.")
print("PASS  No-insulin and no-HbA1c glycemic sensitivities completed.")
print("PASS  Quasi-Poisson somatic-score robustness completed.")
print("PASS  Selection-flow/descriptive audit saved.")

comp = pd.read_csv(RESULTS / "48_representation_comparator_R2.csv")
tests_out = pd.read_csv(RESULTS / "48_comparator_sensitivity_tests.csv")
qp = pd.read_csv(RESULTS / "48_quasipoisson_somatic_tests.csv")

print()
print("SAME-SAMPLE REPRESENTATION COMPARATORS")
print(comp[comp["population"].isin(["2005-2008", "2009-2018"])].to_string(index=False))
print()
print("KEY COMPARATOR / GLYCEMIC SENSITIVITY TESTS")
key = tests_out[tests_out["population"].isin(["2005-2008", "2009-2018"])].copy()
print(key.to_string(index=False))
print()
print("QUASI-POISSON ROBUSTNESS")
print(qp[qp["population"].isin(["2005-2008", "2009-2018"])].to_string(index=False))
