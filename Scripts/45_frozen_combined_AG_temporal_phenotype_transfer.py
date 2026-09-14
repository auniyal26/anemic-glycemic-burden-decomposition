from pathlib import Path
import json
import re
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

BASE = PROCESSED / "31_transfer_harmonized_FINAL_PREMODEL.parquet"
A_TEMP = PROCESSED / "40_frozen_A_temporal_projection.parquet"
G_TEMP = PROCESSED / "42_G3_temporal_frozen_scores.parquet"
SCALING = RESULTS / "44_final_AG_component_scaling.csv"

for p in [BASE, A_TEMP, G_TEMP, SCALING]:
    if not p.exists():
        raise FileNotFoundError(p)

print()
print("FROZEN COMBINED A + G3 TEMPORAL PHENOTYPE TRANSFER")
print("==================================================")
print("No representation refit.")
print("No temporal re-standardization.")
print("A and G3 component coordinates/scales remain discovery-frozen.")
print("Primary replication: 2009-2018 pooled, X3.")
print("Modern holdout: 2021-2023 X3 descriptive/inferential where design allows; X0 sensitivity.")
print()

X3 = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH_FROZEN",
    "INDFMPIR",
    "EDUC3",
    "SMOKING3",
    "BMXBMI",
    "EGFR_2021",
]
X0 = ["RIDAGEYR", "RIAGENDR", "RIDRETH_FROZEN"]

A_CANON = [f"A_FROZEN_PC{i}" for i in range(1, 4)]
G_CANON = [f"G3_FROZEN_PC{i}" for i in range(1, 4)]

def norm_cycle(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .replace({
            "910": "0910",
            "1112": "1112",
            "1314": "1314",
            "1516": "1516",
            "1718": "1718",
            "2123": "2123",
        })
    )

def find_component_columns(df, prefix, n=3):
    cols = list(df.columns)
    found = []
    for i in range(1, n + 1):
        exact = f"{prefix}{i}"
        if exact in cols:
            found.append(exact)
            continue

        # Conservative fallback for local naming differences.
        pats = [
            rf"^{re.escape(prefix)}{i}$",
            rf"^{re.escape(prefix.rstrip('_'))}.*PC{i}$",
            rf"^.*{re.escape(prefix.split('_')[0])}.*FROZEN.*PC{i}$",
        ]
        matches = []
        for c in cols:
            if any(re.search(p, c, flags=re.I) for p in pats):
                matches.append(c)
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely identify {prefix}{i}. Candidates={matches}"
            )
        found.append(matches[0])
    return found

def weighted_corr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3:
        return np.nan
    sw = w.sum()
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    vx = np.sum(w * (x - mx) ** 2) / sw
    vy = np.sum(w * (y - my) ** 2) / sw
    if vx <= 0 or vy <= 0:
        return np.nan
    cov = np.sum(w * (x - mx) * (y - my)) / sw
    return float(cov / np.sqrt(vx * vy))

# ---------------------------------------------------------------------
# Load temporal source files.
# ---------------------------------------------------------------------
base = pd.read_parquet(BASE).copy()
a = pd.read_parquet(A_TEMP).copy()
g = pd.read_parquet(G_TEMP).copy()

for d in [base, a, g]:
    d["CYCLE"] = norm_cycle(d["CYCLE"])

A_SOURCE = find_component_columns(a, "A_FROZEN_PC", 3)
G_SOURCE = find_component_columns(g, "G3_FROZEN_PC", 3)

print("Resolved temporal A score columns:", A_SOURCE)
print("Resolved temporal G3 score columns:", G_SOURCE)
print()

# Normalize to canonical discovery names before scaling.
a_keep = ["SEQN", "CYCLE"] + A_SOURCE
a2 = a[a_keep].copy()
a2 = a2.rename(columns=dict(zip(A_SOURCE, A_CANON)))

g_need = ["SEQN", "CYCLE", "WTSAF2YR"] + G_SOURCE
missing_g = [c for c in g_need if c not in g.columns]
if missing_g:
    raise RuntimeError(f"G3 temporal file missing: {missing_g}")

g2 = g[g_need].copy()
g2 = g2.rename(columns=dict(zip(G_SOURCE, G_CANON)))

# ---------------------------------------------------------------------
# Apply EXACT discovery joint-fasting scaling from script 44.
# ---------------------------------------------------------------------
scale = pd.read_csv(SCALING)
scale = scale[scale["analysis"].eq("A_G3_joint_fasting")].copy()

need_scale_components = A_CANON + G_CANON
missing_scale = [c for c in need_scale_components if c not in set(scale["component"])]
if missing_scale:
    raise RuntimeError(
        f"Discovery joint-fasting scaling missing components: {missing_scale}"
    )

phys = a2.merge(
    g2,
    on=["SEQN", "CYCLE"],
    how="inner",
    validate="one_to_one",
)

phys["WTFAST_TEMP"] = np.where(
    phys["CYCLE"].isin(["0910","1112","1314","1516","1718"]),
    pd.to_numeric(phys["WTSAF2YR"], errors="coerce") / 5.0,
    pd.to_numeric(phys["WTSAF2YR"], errors="coerce"),
)

A_Z = []
G_Z = []

for comp in need_scale_components:
    row = scale[scale["component"].eq(comp)]
    if len(row) != 1:
        raise RuntimeError(f"Expected one scaling row for {comp}, found {len(row)}")
    mu = float(row.iloc[0]["weighted_mean"])
    sd = float(row.iloc[0]["weighted_sd"])
    zcol = str(row.iloc[0]["z_column"])

    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Invalid frozen discovery SD for {comp}")

    phys[zcol] = (pd.to_numeric(phys[comp], errors="coerce") - mu) / sd

    if comp.startswith("A_"):
        A_Z.append(zcol)
    else:
        G_Z.append(zcol)

if len(A_Z) != 3 or len(G_Z) != 3:
    raise RuntimeError(f"Unexpected frozen Z columns A={A_Z}, G={G_Z}")

# ---------------------------------------------------------------------
# Merge outcomes/covariates/scalars.
# ---------------------------------------------------------------------
base_required = [
    "SEQN", "CYCLE",
    "SOMATIC_SCORE", "PHQ9_TOTAL",
    "A", "G_HBA1C",
    "SDMVPSU", "SDMVSTRA",
    "STRATUM_TRANSFER", "PSU_TRANSFER",
] + X3

missing_base = [c for c in base_required if c not in base.columns]
if missing_base:
    raise RuntimeError(f"Transfer base missing: {missing_base}")

d = base[base_required].merge(
    phys[
        ["SEQN","CYCLE","WTFAST_TEMP"]
        + A_CANON + G_CANON + A_Z + G_Z
    ],
    on=["SEQN","CYCLE"],
    how="inner",
    validate="one_to_one",
)

d["PERIOD"] = np.where(
    d["CYCLE"].isin(["0910","1112","1314","1516","1718"]),
    "2009-2018",
    np.where(d["CYCLE"].eq("2123"), "2021-2023", "OTHER"),
)
d = d[d["PERIOD"].isin(["2009-2018","2021-2023"])].copy()

# Secondary crude cognitive-affective sum for continuity with script 44.
d["COGAFF_SUM"] = d["PHQ9_TOTAL"] - d["SOMATIC_SCORE"]

# Save cross-block correlations before outcome complete-case filtering.
corr_rows = []
for period in ["2009-2018","2021-2023"]:
    q = d[d["PERIOD"].eq(period)].dropna(
        subset=A_Z + G_Z + ["WTFAST_TEMP"]
    ).copy()
    q = q[q["WTFAST_TEMP"] > 0]
    for apc in A_Z:
        for gpc in G_Z:
            corr_rows.append({
                "period": period,
                "A_component": apc,
                "G_component": gpc,
                "weighted_correlation": weighted_corr(
                    q[apc], q[gpc], q["WTFAST_TEMP"]
                ),
                "n_physiology": len(q),
            })

pd.DataFrame(corr_rows).to_csv(
    RESULTS / "45_temporal_AG_crossblock_correlations.csv",
    index=False,
)

# Complete-case input is deferred to R separately for X3/X0.
csv_path = RESULTS / "45_temporal_AG_input.csv"
d.to_csv(csv_path, index=False)

print(
    "PASS  Temporal joint physiology rows: "
    f"2009-2018={int((d['PERIOD']=='2009-2018').sum()):,}; "
    f"2021-2023={int((d['PERIOD']=='2021-2023').sum()):,}"
)
print("PASS  Discovery-frozen joint scaling applied unchanged.")
print("PASS  No temporal score centering/scaling performed.")
print()

# ---------------------------------------------------------------------
# R survey analysis.
# ---------------------------------------------------------------------
rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])
if rscript is None:
    raise FileNotFoundError("Rscript not found")

# Substitute only literal column-name vectors, not formulas.
a_vec = "c(" + ",".join(json.dumps(x) for x in A_Z) + ")"
g_vec = "c(" + ",".join(json.dumps(x) for x in G_Z) + ")"

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

d$CYCLE <- as.character(d$CYCLE)
d$PERIOD <- as.character(d$PERIOD)

for (v in c("RIAGENDR","RIDRETH_FROZEN","EDUC3","SMOKING3")) {
  d[[v]] <- factor(d[[v]])
}

A_PCS <- __A_VEC__
G_PCS <- __G_VEC__

X3 <- c(
  "RIDAGEYR",
  "RIAGENDR",
  "RIDRETH_FROZEN",
  "INDFMPIR",
  "EDUC3",
  "SMOKING3",
  "BMXBMI",
  "EGFR_2021"
)

X0 <- c(
  "RIDAGEYR",
  "RIAGENDR",
  "RIDRETH_FROZEN"
)

make_formula <- function(outcome, terms) {
  as.formula(paste(outcome, "~", paste(terms, collapse=" + ")))
}

weighted_r2 <- function(fit, dat, outcome, weight_col) {
  pred <- as.numeric(predict(fit, newdata=dat, type="response"))
  y <- dat[[outcome]]
  w <- dat[[weight_col]]
  mu <- sum(w*y) / sum(w)
  sse <- sum(w*(y-pred)^2)
  sst <- sum(w*(y-mu)^2)
  if (sst <= 0) return(NA_real_)
  1 - sse/sst
}

extract_coef <- function(fit, population, adjustment, outcome, model, terms) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]
  out <- data.frame()

  for (term in terms) {
    if (!(term %in% rownames(sm))) next
    out <- rbind(out, data.frame(
      population=population,
      adjustment=adjustment,
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

safe_test <- function(fit, terms, population, adjustment, outcome, label) {
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
      population=population,
      adjustment=adjustment,
      outcome=outcome,
      test=label,
      F=NA_real_,
      df_num=NA_real_,
      df_den=NA_real_,
      p=NA_real_,
      design_df=degf(fit$survey.design),
      inference_status="unavailable"
    ))
  }

  ddf <- as.numeric(z$ddf)
  status <- ifelse(
    is.na(ddf),
    "unavailable",
    ifelse(ddf <= 3, "design_limited", "available")
  )

  data.frame(
    population=population,
    adjustment=adjustment,
    outcome=outcome,
    test=label,
    F=as.numeric(z$Ftest),
    df_num=as.numeric(z$df),
    df_den=ddf,
    p=as.numeric(z$p),
    design_df=degf(fit$survey.design),
    inference_status=status
  )
}

run_period <- function(dat, population, adjustment, pooled_cycles) {

  covars <- if (adjustment=="X3") X3 else X0

  needed <- unique(c(
    "SOMATIC_SCORE","PHQ9_TOTAL","COGAFF_SUM",
    "A","G_HBA1C",
    A_PCS,G_PCS,
    covars,
    "WTFAST_TEMP",
    "SDMVPSU","SDMVSTRA",
    "STRATUM_TRANSFER","PSU_TRANSFER"
  ))

  dat <- dat[complete.cases(dat[,needed]), ]
  dat <- dat[dat$WTFAST_TEMP > 0, ]

  if (pooled_cycles) {
    dat$CYCLE <- factor(dat$CYCLE)
    des <- svydesign(
      ids=~PSU_TRANSFER,
      strata=~STRATUM_TRANSFER,
      weights=~WTFAST_TEMP,
      nest=TRUE,
      data=dat
    )
    X <- c(covars, "CYCLE")
  } else {
    des <- svydesign(
      ids=~SDMVPSU,
      strata=~SDMVSTRA,
      weights=~WTFAST_TEMP,
      nest=TRUE,
      data=dat
    )
    X <- covars
  }

  coef_rows <- data.frame()
  test_rows <- data.frame()
  r2_rows <- data.frame()

  for (outcome in c("SOMATIC_SCORE","PHQ9_TOTAL","COGAFF_SUM")) {

    fit_x <- svyglm(
      make_formula(outcome, X),
      design=des, family=gaussian()
    )

    fit_scalars <- svyglm(
      make_formula(outcome, c("A","G_HBA1C",X)),
      design=des, family=gaussian()
    )

    fit_Apcs_Gscalar <- svyglm(
      make_formula(outcome, c(A_PCS,"G_HBA1C",X)),
      design=des, family=gaussian()
    )

    fit_Ascalar_Gpcs <- svyglm(
      make_formula(outcome, c("A",G_PCS,X)),
      design=des, family=gaussian()
    )

    fit_pcs <- svyglm(
      make_formula(outcome, c(A_PCS,G_PCS,X)),
      design=des, family=gaussian()
    )

    fit_union <- svyglm(
      make_formula(
        outcome,
        c("A","G_HBA1C",A_PCS,G_PCS,X)
      ),
      design=des, family=gaussian()
    )

    coef_rows <- rbind(
      coef_rows,
      extract_coef(
        fit_pcs, population, adjustment, outcome, "all_PCs",
        c(A_PCS,G_PCS)
      ),
      extract_coef(
        fit_union, population, adjustment, outcome, "union",
        c("A","G_HBA1C",A_PCS,G_PCS)
      )
    )

    test_rows <- rbind(
      test_rows,
      safe_test(
        fit_pcs, A_PCS, population, adjustment, outcome,
        "A_PCs_joint_given_G_PCs_X"
      ),
      safe_test(
        fit_pcs, G_PCS, population, adjustment, outcome,
        "G_PCs_joint_given_A_PCs_X"
      ),
      safe_test(
        fit_pcs, A_PCS[-1], population, adjustment, outcome,
        "A_PC2_3_extra_beyond_A_PC1_G_PCs_X"
      ),
      safe_test(
        fit_pcs, G_PCS[-1], population, adjustment, outcome,
        "G_PC2_3_extra_beyond_G_PC1_A_PCs_X"
      ),
      safe_test(
        fit_union, A_PCS, population, adjustment, outcome,
        "A_PCs_extra_beyond_scalar_A_scalar_G_G_PCs_X"
      ),
      safe_test(
        fit_union, G_PCS, population, adjustment, outcome,
        "G_PCs_extra_beyond_scalar_G_scalar_A_A_PCs_X"
      ),
      safe_test(
        fit_union, c(A_PCS,G_PCS), population, adjustment, outcome,
        "all_PCs_extra_beyond_scalar_A_scalar_G_X"
      ),
      safe_test(
        fit_union, c("A","G_HBA1C"), population, adjustment, outcome,
        "both_scalars_extra_beyond_all_PCs_X"
      ),
      safe_test(
        fit_union, "A", population, adjustment, outcome,
        "scalar_A_extra_beyond_all_PCs_scalar_G_X"
      ),
      safe_test(
        fit_union, "G_HBA1C", population, adjustment, outcome,
        "scalar_G_extra_beyond_all_PCs_scalar_A_X"
      )
    )

    fits <- list(
      X=fit_x,
      X_scalarA_scalarG=fit_scalars,
      X_Apcs_scalarG=fit_Apcs_Gscalar,
      X_scalarA_Gpcs=fit_Ascalar_Gpcs,
      X_Apcs_Gpcs=fit_pcs,
      X_scalars_Apcs_Gpcs=fit_union
    )

    for (nm in names(fits)) {
      r2_rows <- rbind(r2_rows, data.frame(
        population=population,
        adjustment=adjustment,
        outcome=outcome,
        model=nm,
        weighted_R2=weighted_r2(
          fits[[nm]], dat, outcome, "WTFAST_TEMP"
        ),
        n=nrow(dat),
        design_df=degf(des),
        stringsAsFactors=FALSE
      ))
    }
  }

  list(coef=coef_rows, tests=test_rows, r2=r2_rows)
}

d0918 <- d[d$PERIOD=="2009-2018", ]
d2123 <- d[d$PERIOD=="2021-2023", ]

# Primary temporal replication.
r0918_x3 <- run_period(
  d0918, "2009-2018", "X3", pooled_cycles=TRUE
)

# Modern holdout: same X3 plus reduced-adjustment sensitivity.
r2123_x3 <- run_period(
  d2123, "2021-2023", "X3", pooled_cycles=FALSE
)

r2123_x0 <- run_period(
  d2123, "2021-2023", "X0", pooled_cycles=FALSE
)

coef_all <- rbind(
  r0918_x3$coef,
  r2123_x3$coef,
  r2123_x0$coef
)

tests_all <- rbind(
  r0918_x3$tests,
  r2123_x3$tests,
  r2123_x0$tests
)

r2_all <- rbind(
  r0918_x3$r2,
  r2123_x3$r2,
  r2123_x0$r2
)

write.csv(
  coef_all,
  file.path(results_dir, "45_temporal_AG_coefficients.csv"),
  row.names=FALSE
)

write.csv(
  tests_all,
  file.path(results_dir, "45_temporal_AG_block_tests.csv"),
  row.names=FALSE
)

write.csv(
  r2_all,
  file.path(results_dir, "45_temporal_AG_model_R2.csv"),
  row.names=FALSE
)
'''

r_code = r_code.replace("__A_VEC__", a_vec).replace("__G_VEC__", g_vec)

r_path = AUDIT / "45_temporal_AG_transfer.R"
r_path.write_text(r_code, encoding="utf-8")

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
        str(csv_path),
        str(RESULTS),
        str(RLIB),
    ],
    capture_output=True,
    text=True,
)

if proc.returncode != 0:
    print("FAIL  Temporal R survey analysis failed.")
    print(proc.stdout[-4000:])
    print(proc.stderr[-7000:])
    raise SystemExit(proc.returncode)

coef = pd.read_csv(RESULTS / "45_temporal_AG_coefficients.csv")
tests = pd.read_csv(RESULTS / "45_temporal_AG_block_tests.csv")
r2 = pd.read_csv(RESULTS / "45_temporal_AG_model_R2.csv")

# ---------------------------------------------------------------------
# BH correction for individual component coefficients within period/model.
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

coef["p_BH_component_family"] = np.nan
for keys, idx in coef.groupby(
    ["population","adjustment","outcome","model"]
).groups.items():
    comp_idx = [
        i for i in idx
        if "FROZEN_PC" in str(coef.loc[i, "term"])
    ]
    if comp_idx:
        coef.loc[comp_idx, "p_BH_component_family"] = bh(
            coef.loc[comp_idx, "p"].to_numpy(float)
        )

coef.to_csv(
    RESULTS / "45_temporal_AG_coefficients.csv",
    index=False,
)

# ---------------------------------------------------------------------
# R2 increment summary.
# ---------------------------------------------------------------------
delta_rows = []
for (pop, adj, outcome), q in r2.groupby(
    ["population","adjustment","outcome"]
):
    v = dict(zip(q["model"], q["weighted_R2"]))
    delta_rows.append({
        "population": pop,
        "adjustment": adj,
        "outcome": outcome,
        "n": int(q["n"].iloc[0]),
        "design_df": float(q["design_df"].iloc[0]),
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
    RESULTS / "45_temporal_AG_incremental_information.csv",
    index=False,
)

audit = {
    "script": "45_frozen_combined_AG_temporal_phenotype_transfer.py",
    "A_source": str(A_TEMP.relative_to(ROOT)),
    "G3_source": str(G_TEMP.relative_to(ROOT)),
    "discovery_scaling_source": str(SCALING.relative_to(ROOT)),
    "A_temporal_columns_resolved": A_SOURCE,
    "G_temporal_columns_resolved": G_SOURCE,
    "A_z_columns": A_Z,
    "G_z_columns": G_Z,
    "representation_refit": False,
    "temporal_restandardization": False,
    "2009_2018_weight": "WTSAF2YR / 5",
    "2021_2023_weight": "WTSAF2YR",
    "primary_2009_2018_adjustment": "X3 + cycle",
    "modern_holdout_adjustments": ["X3", "X0 sensitivity"],
    "interactions_tested": False,
    "primary_outcome": "SOMATIC_SCORE",
    "secondary_outcomes": ["PHQ9_TOTAL","COGAFF_SUM"],
}

(AUDIT / "45_temporal_AG_transfer.json").write_text(
    json.dumps(audit, indent=2),
    encoding="utf-8",
)

# ---------------------------------------------------------------------
# Terminal report.
# ---------------------------------------------------------------------
print()
print("TEMPORAL COMBINED A+G3 — SOMATIC BLOCK TESTS")
show = tests[
    tests["outcome"].eq("SOMATIC_SCORE")
    & (
        ((tests["population"]=="2009-2018") & (tests["adjustment"]=="X3"))
        | ((tests["population"]=="2021-2023") & (tests["adjustment"].isin(["X3","X0"])))
    )
].copy()

print(
    show[
        ["population","adjustment","test","F","df_num","df_den","p",
         "design_df","inference_status"]
    ].to_string(index=False)
)

print()
print("TEMPORAL COMBINED A+G3 — SOMATIC INCREMENTAL INFORMATION")
print(
    delta[
        delta["outcome"].eq("SOMATIC_SCORE")
    ].to_string(index=False)
)

print()
print("TEMPORAL ALL-PC INDIVIDUAL COEFFICIENTS — SOMATIC")
print(
    coef[
        coef["outcome"].eq("SOMATIC_SCORE")
        & coef["model"].eq("all_PCs")
    ][
        ["population","adjustment","term","beta","se","ci_low","ci_high",
         "p","p_BH_component_family"]
    ].to_string(index=False)
)

print()
print("CROSS-BLOCK CORRELATIONS")
print(
    pd.read_csv(
        RESULTS / "45_temporal_AG_crossblock_correlations.csv"
    ).to_string(index=False)
)

print()
print("Saved:")
print("  Results/45_temporal_AG_input.csv")
print("  Results/45_temporal_AG_crossblock_correlations.csv")
print("  Results/45_temporal_AG_coefficients.csv")
print("  Results/45_temporal_AG_block_tests.csv")
print("  Results/45_temporal_AG_model_R2.csv")
print("  Results/45_temporal_AG_incremental_information.csv")
print("  Audit/45_temporal_AG_transfer.R")
print("  Audit/45_temporal_AG_transfer.json")
