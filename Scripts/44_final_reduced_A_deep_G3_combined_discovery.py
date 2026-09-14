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

PARITY = RESULTS / "25_survey_parity_input.csv"
A_SCORES = PROCESSED / "39_frozen_reduced_A_scores.parquet"
G3_SCORES = PROCESSED / "42_G3_discovery_frozen_scores.parquet"

for p in [PARITY, A_SCORES, G3_SCORES]:
    if not p.exists():
        raise FileNotFoundError(p)

print()
print("FINAL REDUCED A + DEEP G3 DISCOVERY TEST")
print("========================================")
print("A: frozen reduced Hb/RBC/MCV/RDW representation, retained PC1-PC3")
print("G3: frozen HbA1c/fasting-glucose/log-insulin representation, PC1-PC3")
print("Primary phenotype: somatic PHQ symptom score")
print("Primary adjustment: frozen X3")
print("Formal inference: R survey::svyglm + survey::regTermTest")
print()

X3 = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "INDFMPIR",
    "EDUC3",
    "SMOKING3",
    "BMXBMI",
    "EGFR_2021",
]

A_PCS_RAW = [f"A_FROZEN_PC{i}" for i in range(1, 4)]
G_PCS_RAW = [f"G3_FROZEN_PC{i}" for i in range(1, 4)]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def norm_cycle(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .replace({"506": "0506", "708": "0708"})
    )

def weighted_mean_sd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan, np.nan
    sw = w.sum()
    m = np.sum(w * x) / sw
    v = np.sum(w * (x - m) ** 2) / sw
    return float(m), float(np.sqrt(v))

def weighted_corr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return np.nan
    sw = w.sum()
    mx = np.sum(w*x)/sw
    my = np.sum(w*y)/sw
    cov = np.sum(w*(x-mx)*(y-my))/sw
    vx = np.sum(w*(x-mx)**2)/sw
    vy = np.sum(w*(y-my)**2)/sw
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx*vy))

# ---------------------------------------------------------------------
# Base survey cohort
# ---------------------------------------------------------------------
parity = pd.read_csv(PARITY)
parity["CYCLE"] = norm_cycle(parity["CYCLE"])

required = [
    "SEQN", "CYCLE", "STRATUM", "PSU", "WTMEC4YR",
    "SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C"
] + X3

missing = [c for c in required if c not in parity.columns]
if missing:
    raise ValueError(f"Parity input missing: {missing}")

# ---------------------------------------------------------------------
# Frozen A scores
# ---------------------------------------------------------------------
a = pd.read_parquet(A_SCORES).copy()
a["CYCLE"] = norm_cycle(a["CYCLE"])

need_a = ["SEQN", "CYCLE", "WTMEC2YR", "WTMEC4YR_FREEZE"] + A_PCS_RAW
missing = [c for c in need_a if c not in a.columns]
if missing:
    raise ValueError(f"A frozen score file missing: {missing}")

a = a[need_a].copy()

# Full-MEC, outcome-independent standardization for final A validation.
A_MEC_Z = []
scale_rows = []

for pc in A_PCS_RAW:
    m, sd = weighted_mean_sd(a[pc], a["WTMEC4YR_FREEZE"])
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Invalid full-MEC A SD for {pc}")
    z = pc + "_MEC_Z"
    a[z] = (a[pc] - m) / sd
    A_MEC_Z.append(z)
    scale_rows.append({
        "analysis": "A_final_MEC",
        "component": pc,
        "weighted_mean": m,
        "weighted_sd": sd,
        "z_column": z,
    })

mec = parity.merge(
    a[["SEQN", "CYCLE"] + A_MEC_Z],
    on=["SEQN", "CYCLE"],
    how="inner",
    validate="one_to_one",
)

mec_complete = (
    ["SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C",
     "STRATUM", "PSU", "WTMEC4YR"]
    + X3 + A_MEC_Z
)
mec = mec.dropna(subset=mec_complete).copy()
mec = mec[mec["WTMEC4YR"] > 0].copy()

# ---------------------------------------------------------------------
# Frozen G3 scores + joint A/G physiology intersection.
# Standardize BOTH representations on the joint fasting physiology sample,
# before applying PHQ/outcome complete-case restrictions.
# ---------------------------------------------------------------------
g = pd.read_parquet(G3_SCORES).copy()
g["CYCLE"] = norm_cycle(g["CYCLE"])

need_g = ["SEQN", "CYCLE", "WTSAF2YR"] + G_PCS_RAW
missing = [c for c in need_g if c not in g.columns]
if missing:
    raise ValueError(f"G3 frozen score file missing: {missing}")

g = g[need_g].copy()

joint_phys = a[
    ["SEQN", "CYCLE"] + A_PCS_RAW
].merge(
    g,
    on=["SEQN", "CYCLE"],
    how="inner",
    validate="one_to_one",
)

joint_phys["WTFAST4YR"] = (
    pd.to_numeric(joint_phys["WTSAF2YR"], errors="coerce") / 2.0
)
joint_phys = joint_phys.dropna(
    subset=A_PCS_RAW + G_PCS_RAW + ["WTFAST4YR"]
).copy()
joint_phys = joint_phys[joint_phys["WTFAST4YR"] > 0].copy()

A_JOINT_Z = []
G_JOINT_Z = []

for pc in A_PCS_RAW:
    m, sd = weighted_mean_sd(joint_phys[pc], joint_phys["WTFAST4YR"])
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Invalid joint fasting A SD for {pc}")
    z = pc + "_FZ"
    joint_phys[z] = (joint_phys[pc] - m) / sd
    A_JOINT_Z.append(z)
    scale_rows.append({
        "analysis": "A_G3_joint_fasting",
        "component": pc,
        "weighted_mean": m,
        "weighted_sd": sd,
        "z_column": z,
    })

for pc in G_PCS_RAW:
    m, sd = weighted_mean_sd(joint_phys[pc], joint_phys["WTFAST4YR"])
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Invalid joint fasting G SD for {pc}")
    z = pc + "_FZ"
    joint_phys[z] = (joint_phys[pc] - m) / sd
    G_JOINT_Z.append(z)
    scale_rows.append({
        "analysis": "A_G3_joint_fasting",
        "component": pc,
        "weighted_mean": m,
        "weighted_sd": sd,
        "z_column": z,
    })

pd.DataFrame(scale_rows).to_csv(
    RESULTS / "44_final_AG_component_scaling.csv",
    index=False,
)

# Cross-system physiological correlations, before conditioning on PHQ.
corr_rows = []
for apc in A_JOINT_Z:
    for gpc in G_JOINT_Z:
        corr_rows.append({
            "A_component": apc,
            "G_component": gpc,
            "weighted_correlation": weighted_corr(
                joint_phys[apc],
                joint_phys[gpc],
                joint_phys["WTFAST4YR"],
            ),
            "n_joint_physiology": len(joint_phys),
        })

pd.DataFrame(corr_rows).to_csv(
    RESULTS / "44_final_AG_crossblock_correlations.csv",
    index=False,
)

joint = parity.merge(
    joint_phys[
        ["SEQN", "CYCLE", "WTFAST4YR"] + A_JOINT_Z + G_JOINT_Z
    ],
    on=["SEQN", "CYCLE"],
    how="inner",
    validate="one_to_one",
)

joint_complete = (
    ["SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C",
     "STRATUM", "PSU", "WTFAST4YR"]
    + X3 + A_JOINT_Z + G_JOINT_Z
)
joint = joint.dropna(subset=joint_complete).copy()
joint = joint[joint["WTFAST4YR"] > 0].copy()

# Optional crude score corresponding to the six non-somatic PHQ items.
# This is secondary/descriptive, not a replacement for the validated latent CFA.
joint["COGAFF_SUM"] = joint["PHQ9_TOTAL"] - joint["SOMATIC_SCORE"]
mec["COGAFF_SUM"] = mec["PHQ9_TOTAL"] - mec["SOMATIC_SCORE"]

mec_path = RESULTS / "44_final_A_MEC_input.csv"
joint_path = RESULTS / "44_final_AG_joint_fasting_input.csv"
mec.to_csv(mec_path, index=False)
joint.to_csv(joint_path, index=False)

print(f"PASS  Final reduced-A MEC X3 sample: n={len(mec):,}")
print(f"PASS  Joint frozen A+G3 fasting physiology sample: n={len(joint_phys):,}")
print(f"PASS  Joint frozen A+G3 fasting X3 phenotype sample: n={len(joint):,}")
print()

# ---------------------------------------------------------------------
# Static R survey analysis
# ---------------------------------------------------------------------
rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])

if rscript is None:
    raise FileNotFoundError("Rscript not found")

r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
mec_file <- args[1]
joint_file <- args[2]
results_dir <- args[3]
user_lib <- args[4]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("survey", quietly=TRUE)) {
  install.packages("survey", repos="https://cloud.r-project.org", lib=user_lib)
}
library(survey)
options(survey.lonely.psu="adjust")

mec <- read.csv(mec_file, stringsAsFactors=FALSE)
joint <- read.csv(joint_file, stringsAsFactors=FALSE)

normalize_cycle <- function(x) {
  z <- as.character(x)
  z[z %in% c("506","0506")] <- "0506"
  z[z %in% c("708","0708")] <- "0708"
  z
}

prep <- function(d) {
  d$CYCLE <- factor(normalize_cycle(d$CYCLE))
  d$RIAGENDR <- factor(d$RIAGENDR)
  d$RIDRETH1 <- factor(d$RIDRETH1)
  d$EDUC3 <- factor(d$EDUC3)
  d$SMOKING3 <- factor(d$SMOKING3)
  d
}

mec <- prep(mec)
joint <- prep(joint)

X3 <- c(
  "RIDAGEYR",
  "RIAGENDR",
  "RIDRETH1",
  "INDFMPIR",
  "EDUC3",
  "SMOKING3",
  "BMXBMI",
  "EGFR_2021",
  "CYCLE"
)

A_MEC <- c(
  "A_FROZEN_PC1_MEC_Z",
  "A_FROZEN_PC2_MEC_Z",
  "A_FROZEN_PC3_MEC_Z"
)

A_JOINT <- c(
  "A_FROZEN_PC1_FZ",
  "A_FROZEN_PC2_FZ",
  "A_FROZEN_PC3_FZ"
)

G_JOINT <- c(
  "G3_FROZEN_PC1_FZ",
  "G3_FROZEN_PC2_FZ",
  "G3_FROZEN_PC3_FZ"
)

make_formula <- function(outcome, terms) {
  as.formula(
    paste(outcome, "~", paste(terms, collapse=" + "))
  )
}

make_design <- function(d, weight_col) {
  svydesign(
    ids=~PSU,
    strata=~STRATUM,
    weights=as.formula(paste0("~", weight_col)),
    nest=TRUE,
    data=d
  )
}

weighted_r2 <- function(fit, d, weight_col, outcome) {
  y <- d[[outcome]]
  pred <- as.numeric(predict(fit, newdata=d))
  w <- d[[weight_col]]
  mu <- sum(w*y) / sum(w)
  sse <- sum(w*(y-pred)^2)
  sst <- sum(w*(y-mu)^2)
  if (sst <= 0) return(NA_real_)
  1 - sse/sst
}

extract_coef <- function(fit, analysis, outcome, model, terms) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]
  out <- data.frame()

  for (term in terms) {
    if (!(term %in% rownames(sm))) next
    out <- rbind(out, data.frame(
      analysis=analysis,
      outcome=outcome,
      model=model,
      term=term,
      beta=sm[term,"Estimate"],
      se=sm[term,"Std. Error"],
      ci_low=ci[term,1],
      ci_high=ci[term,2],
      p=sm[term,pcol],
      stringsAsFactors=FALSE
    ))
  }
  out
}

safe_test <- function(fit, terms, analysis, outcome, label) {
  z <- tryCatch(
    regTermTest(
      fit,
      as.formula(paste("~", paste(terms, collapse=" + "))),
      method="Wald"
    ),
    error=function(e) NULL
  )

  if (is.null(z)) {
    return(data.frame(
      analysis=analysis, outcome=outcome, test=label,
      F=NA_real_, df_num=NA_real_, df_den=NA_real_, p=NA_real_,
      design_df=degf(fit$survey.design)
    ))
  }

  data.frame(
    analysis=analysis,
    outcome=outcome,
    test=label,
    F=as.numeric(z$Ftest),
    df_num=as.numeric(z$df),
    df_den=as.numeric(z$ddf),
    p=as.numeric(z$p),
    design_df=degf(fit$survey.design)
  )
}

add_r2 <- function(store, fit, analysis, outcome, model, d, weight_col) {
  rbind(store, data.frame(
    analysis=analysis,
    outcome=outcome,
    model=model,
    weighted_R2=weighted_r2(fit, d, weight_col, outcome),
    n=nrow(d),
    design_df=degf(fit$survey.design)
  ))
}

coef_rows <- data.frame()
test_rows <- data.frame()
r2_rows <- data.frame()

# =====================================================================
# Analysis 1: manuscript-grade final reduced A on the full MEC sample.
# =====================================================================
des_mec <- make_design(mec, "WTMEC4YR")

for (outcome in c("SOMATIC_SCORE","PHQ9_TOTAL","COGAFF_SUM")) {

  fit_x <- svyglm(
    make_formula(outcome, X3),
    design=des_mec, family=gaussian()
  )

  fit_scalar <- svyglm(
    make_formula(outcome, c("A","G_HBA1C",X3)),
    design=des_mec, family=gaussian()
  )

  fit_apc <- svyglm(
    make_formula(outcome, c(A_MEC,"G_HBA1C",X3)),
    design=des_mec, family=gaussian()
  )

  fit_union <- svyglm(
    make_formula(outcome, c("A",A_MEC,"G_HBA1C",X3)),
    design=des_mec, family=gaussian()
  )

  coef_rows <- rbind(
    coef_rows,
    extract_coef(
      fit_apc, "A_final_MEC", outcome, "A_PCs_plus_scalar_G",
      c(A_MEC,"G_HBA1C")
    ),
    extract_coef(
      fit_union, "A_final_MEC", outcome, "union",
      c("A",A_MEC,"G_HBA1C")
    )
  )

  test_rows <- rbind(
    test_rows,
    safe_test(
      fit_apc, A_MEC, "A_final_MEC", outcome,
      "A_PCs_joint_given_scalar_G_X"
    ),
    safe_test(
      fit_apc, A_MEC[-1], "A_final_MEC", outcome,
      "A_PC2_3_extra_beyond_PC1_scalar_G_X"
    ),
    safe_test(
      fit_union, A_MEC, "A_final_MEC", outcome,
      "A_PCs_extra_beyond_scalar_A_G_X"
    ),
    safe_test(
      fit_union, c("A"), "A_final_MEC", outcome,
      "scalar_A_extra_beyond_A_PCs_G_X"
    )
  )

  r2_rows <- add_r2(
    r2_rows, fit_x, "A_final_MEC", outcome, "X", mec, "WTMEC4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_scalar, "A_final_MEC", outcome,
    "X_scalarA_scalarG", mec, "WTMEC4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_apc, "A_final_MEC", outcome,
    "X_Apcs_scalarG", mec, "WTMEC4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_union, "A_final_MEC", outcome,
    "X_scalarA_Apcs_scalarG", mec, "WTMEC4YR"
  )
}

# =====================================================================
# Analysis 2: final combined frozen A + G3 on SAME fasting sample.
# =====================================================================
des_joint <- make_design(joint, "WTFAST4YR")

for (outcome in c("SOMATIC_SCORE","PHQ9_TOTAL","COGAFF_SUM")) {

  fit_x <- svyglm(
    make_formula(outcome, X3),
    design=des_joint, family=gaussian()
  )

  fit_scalars <- svyglm(
    make_formula(outcome, c("A","G_HBA1C",X3)),
    design=des_joint, family=gaussian()
  )

  fit_Apcs_Gscalar <- svyglm(
    make_formula(outcome, c(A_JOINT,"G_HBA1C",X3)),
    design=des_joint, family=gaussian()
  )

  fit_Ascalar_Gpcs <- svyglm(
    make_formula(outcome, c("A",G_JOINT,X3)),
    design=des_joint, family=gaussian()
  )

  fit_pcs <- svyglm(
    make_formula(outcome, c(A_JOINT,G_JOINT,X3)),
    design=des_joint, family=gaussian()
  )

  fit_union <- svyglm(
    make_formula(
      outcome,
      c("A","G_HBA1C",A_JOINT,G_JOINT,X3)
    ),
    design=des_joint, family=gaussian()
  )

  coef_rows <- rbind(
    coef_rows,
    extract_coef(
      fit_pcs, "AG3_joint_fasting", outcome, "all_PCs",
      c(A_JOINT,G_JOINT)
    ),
    extract_coef(
      fit_union, "AG3_joint_fasting", outcome, "union",
      c("A","G_HBA1C",A_JOINT,G_JOINT)
    )
  )

  test_rows <- rbind(
    test_rows,

    safe_test(
      fit_pcs, A_JOINT, "AG3_joint_fasting", outcome,
      "A_PCs_joint_given_G_PCs_X"
    ),

    safe_test(
      fit_pcs, G_JOINT, "AG3_joint_fasting", outcome,
      "G_PCs_joint_given_A_PCs_X"
    ),

    safe_test(
      fit_pcs, A_JOINT[-1], "AG3_joint_fasting", outcome,
      "A_PC2_3_extra_beyond_A_PC1_G_PCs_X"
    ),

    safe_test(
      fit_pcs, G_JOINT[-1], "AG3_joint_fasting", outcome,
      "G_PC2_3_extra_beyond_G_PC1_A_PCs_X"
    ),

    safe_test(
      fit_union, A_JOINT, "AG3_joint_fasting", outcome,
      "A_PCs_extra_beyond_scalar_A_scalar_G_G_PCs_X"
    ),

    safe_test(
      fit_union, G_JOINT, "AG3_joint_fasting", outcome,
      "G_PCs_extra_beyond_scalar_G_scalar_A_A_PCs_X"
    ),

    safe_test(
      fit_union, c(A_JOINT,G_JOINT), "AG3_joint_fasting", outcome,
      "all_PCs_extra_beyond_scalar_A_scalar_G_X"
    ),

    safe_test(
      fit_union, c("A","G_HBA1C"), "AG3_joint_fasting", outcome,
      "both_scalars_extra_beyond_all_PCs_X"
    ),

    safe_test(
      fit_union, c("A"), "AG3_joint_fasting", outcome,
      "scalar_A_extra_beyond_all_PCs_scalar_G_X"
    ),

    safe_test(
      fit_union, c("G_HBA1C"), "AG3_joint_fasting", outcome,
      "scalar_G_extra_beyond_all_PCs_scalar_A_X"
    )
  )

  r2_rows <- add_r2(
    r2_rows, fit_x, "AG3_joint_fasting", outcome,
    "X", joint, "WTFAST4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_scalars, "AG3_joint_fasting", outcome,
    "X_scalarA_scalarG", joint, "WTFAST4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_Apcs_Gscalar, "AG3_joint_fasting", outcome,
    "X_Apcs_scalarG", joint, "WTFAST4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_Ascalar_Gpcs, "AG3_joint_fasting", outcome,
    "X_scalarA_Gpcs", joint, "WTFAST4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_pcs, "AG3_joint_fasting", outcome,
    "X_Apcs_Gpcs", joint, "WTFAST4YR"
  )
  r2_rows <- add_r2(
    r2_rows, fit_union, "AG3_joint_fasting", outcome,
    "X_scalars_Apcs_Gpcs", joint, "WTFAST4YR"
  )
}

write.csv(
  coef_rows,
  file.path(results_dir, "44_final_AG_coefficients.csv"),
  row.names=FALSE
)

write.csv(
  test_rows,
  file.path(results_dir, "44_final_AG_block_tests.csv"),
  row.names=FALSE
)

write.csv(
  r2_rows,
  file.path(results_dir, "44_final_AG_model_R2.csv"),
  row.names=FALSE
)
'''

r_path = AUDIT / "44_final_AG_discovery.R"
r_path.write_text(r_code, encoding="utf-8")

# Syntax check before model execution.
r_path_r = str(r_path).replace("\\", "/")
parse_proc = subprocess.run(
    [str(rscript), "-e", f'parse(file="{r_path_r}")'],
    capture_output=True,
    text=True,
)

if parse_proc.returncode != 0:
    print("FAIL  Generated R script did not parse.")
    print(parse_proc.stdout[-3000:])
    print(parse_proc.stderr[-3000:])
    raise SystemExit(parse_proc.returncode)

print("PASS  Generated R script syntax check.")

proc = subprocess.run(
    [
        str(rscript),
        str(r_path),
        str(mec_path),
        str(joint_path),
        str(RESULTS),
        str(RLIB),
    ],
    capture_output=True,
    text=True,
)

if proc.returncode != 0:
    print("FAIL  R survey analysis failed.")
    print(proc.stdout[-4000:])
    print(proc.stderr[-6000:])
    raise SystemExit(proc.returncode)

coef = pd.read_csv(RESULTS / "44_final_AG_coefficients.csv")
tests = pd.read_csv(RESULTS / "44_final_AG_block_tests.csv")
r2 = pd.read_csv(RESULTS / "44_final_AG_model_R2.csv")

# BH correction for individual component coefficients within each
# analysis/outcome/model component family.
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
    adj = ranked * n / np.arange(1, n+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    restore = np.empty(n)
    restore[order] = adj
    out[np.where(ok)[0]] = restore
    return out

coef["p_BH_component_family"] = np.nan

for (analysis, outcome, model), idx in coef.groupby(
    ["analysis","outcome","model"]
).groups.items():
    component_idx = [
        i for i in idx
        if "FROZEN_PC" in str(coef.loc[i, "term"])
    ]
    if component_idx:
        coef.loc[component_idx, "p_BH_component_family"] = bh(
            coef.loc[component_idx, "p"].to_numpy(float)
        )

coef.to_csv(RESULTS / "44_final_AG_coefficients.csv", index=False)

# Compact R2 increments.
delta_rows = []

for (analysis, outcome), q in r2.groupby(["analysis","outcome"]):
    v = dict(zip(q["model"], q["weighted_R2"]))

    if analysis == "A_final_MEC":
        delta_rows.append({
            "analysis": analysis,
            "outcome": outcome,
            "delta_scalar_model_beyond_X":
                v["X_scalarA_scalarG"] - v["X"],
            "delta_Apcs_scalarG_beyond_X":
                v["X_Apcs_scalarG"] - v["X"],
            "delta_Apcs_beyond_scalar_model":
                v["X_scalarA_Apcs_scalarG"] - v["X_scalarA_scalarG"],
            "delta_scalarA_beyond_Apcs_scalarG":
                v["X_scalarA_Apcs_scalarG"] - v["X_Apcs_scalarG"],
        })

    else:
        delta_rows.append({
            "analysis": analysis,
            "outcome": outcome,
            "delta_scalar_model_beyond_X":
                v["X_scalarA_scalarG"] - v["X"],
            "delta_all_PCs_beyond_X":
                v["X_Apcs_Gpcs"] - v["X"],
            "delta_all_PCs_beyond_scalar_model":
                v["X_scalars_Apcs_Gpcs"] - v["X_scalarA_scalarG"],
            "delta_scalars_beyond_all_PCs":
                v["X_scalars_Apcs_Gpcs"] - v["X_Apcs_Gpcs"],
            "delta_Apcs_scalarG_beyond_scalars":
                v["X_Apcs_scalarG"] - v["X_scalarA_scalarG"],
            "delta_Ascalar_Gpcs_beyond_scalars":
                v["X_scalarA_Gpcs"] - v["X_scalarA_scalarG"],
        })

delta = pd.DataFrame(delta_rows)
delta.to_csv(
    RESULTS / "44_final_AG_incremental_information.csv",
    index=False,
)

audit = {
    "script": "44_final_reduced_A_deep_G3_combined_discovery.py",
    "primary_outcome": "SOMATIC_SCORE",
    "secondary_outcomes": ["PHQ9_TOTAL", "COGAFF_SUM"],
    "primary_adjustment": "X3_kidney_direct_effect + cycle",
    "A_representation": "39 frozen reduced A PC1-PC3",
    "G_representation": "42 frozen G3 PC1-PC3",
    "A_MEC_n": int(len(mec)),
    "joint_physiology_n": int(len(joint_phys)),
    "joint_fasting_phenotype_n": int(len(joint)),
    "joint_weight": "WTSAF2YR/2",
    "representation_refit": False,
    "outcome_used_for_scaling": False,
    "interactions_tested": False,
    "formal_inference": "survey::svyglm + survey::regTermTest",
}

(AUDIT / "44_final_AG_discovery.json").write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8",
)

print()
print("FINAL REDUCED A — FULL MEC SOMATIC")
print(
    tests[
        tests["analysis"].eq("A_final_MEC")
        & tests["outcome"].eq("SOMATIC_SCORE")
    ][
        ["test","F","df_num","df_den","p","design_df"]
    ].to_string(index=False)
)
print()

print("COMBINED FROZEN A + G3 — SOMATIC")
print(
    tests[
        tests["analysis"].eq("AG3_joint_fasting")
        & tests["outcome"].eq("SOMATIC_SCORE")
    ][
        ["test","F","df_num","df_den","p","design_df"]
    ].to_string(index=False)
)
print()

print("COMBINED INCREMENTAL INFORMATION — SOMATIC")
print(
    delta[
        delta["analysis"].eq("AG3_joint_fasting")
        & delta["outcome"].eq("SOMATIC_SCORE")
    ].to_string(index=False)
)
print()

print("CROSS-BLOCK A/G PHYSIOLOGICAL CORRELATIONS")
corr = pd.read_csv(RESULTS / "44_final_AG_crossblock_correlations.csv")
print(corr.to_string(index=False))
print()

print("Saved:")
print("  Results/44_final_AG_component_scaling.csv")
print("  Results/44_final_AG_crossblock_correlations.csv")
print("  Results/44_final_AG_coefficients.csv")
print("  Results/44_final_AG_block_tests.csv")
print("  Results/44_final_AG_model_R2.csv")
print("  Results/44_final_AG_incremental_information.csv")
print("  Results/44_final_A_MEC_input.csv")
print("  Results/44_final_AG_joint_fasting_input.csv")
print("  Audit/44_final_AG_discovery.R")
print("  Audit/44_final_AG_discovery.json")
