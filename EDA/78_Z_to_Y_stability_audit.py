#!/usr/bin/env python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    RESULTS / "73_frozen_phenotype_map_input.csv",
    ROOT / "73_frozen_phenotype_map_input.csv",
]
INPUT = next((p for p in INPUT_CANDIDATES if p.exists()), None)

if INPUT is None:
    raise FileNotFoundError(
        "Need EDA/Results/73_frozen_phenotype_map_input.csv. "
        "Run Script 73 first."
    )

PHYS = [
    "SP_SHARED1", "SP_SHARED2",
    "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]

OUTCOME_ORDER = ["SOMATIC", "COGAFF", "PHQ9_TOTAL"]
PERIOD_ORDER = ["2005-2008", "2009-2018", "2021-2023"]

R_CODE = r'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
out_dir <- args[2]

suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)

for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3")) {
  if (v %in% names(d)) d[[v]] <- factor(d[[v]])
}

PHYS <- c(
  "SP_SHARED1","SP_SHARED2",
  "SP_DISCORD1","SP_DISCORD2",
  "SP_A_PRIVATE3","SP_A_PRIVATE4",
  "SP_G_PRIVATE3"
)

COVARS <- c(
  "RIDAGEYR","RIAGENDR","RACE","INDFMPIR",
  "EDUC3","SMOKING3","BMXBMI","EGFR_2021"
)

OUTCOMES <- c(
  SOMATIC="SOMATIC_SCORE",
  COGAFF="COGAFF_SUM",
  PHQ9_TOTAL="PHQ9_TOTAL"
)

PERIODS <- c("2005-2008","2009-2018","2021-2023")

make_formula <- function(y, terms) {
  as.formula(paste(y, "~", paste(terms, collapse=" + ")))
}

weighted_metrics <- function(y, pred, w) {
  ok <- is.finite(y) & is.finite(pred) & is.finite(w) & w > 0
  y <- y[ok]
  pred <- pred[ok]
  w <- w[ok]
  w <- w / sum(w)

  my <- sum(w*y)
  mp <- sum(w*pred)

  vy <- sum(w*(y-my)^2)
  vp <- sum(w*(pred-mp)^2)
  covyp <- sum(w*(y-my)*(pred-mp))

  sse <- sum(w*(y-pred)^2)
  rmse <- sqrt(sse)
  r2 <- ifelse(vy > 0, 1 - sse/vy, NA_real_)
  corr <- ifelse(vy > 0 && vp > 0, covyp/sqrt(vy*vp), NA_real_)
  slope <- ifelse(vp > 0, covyp/vp, NA_real_)
  intercept <- ifelse(is.finite(slope), my - slope*mp, NA_real_)

  c(
    weighted_R2=r2,
    weighted_corr=corr,
    RMSE=rmse,
    mean_bias=sum(w*(pred-y)),
    calibration_intercept=intercept,
    calibration_slope=slope,
    observed_mean=my,
    predicted_mean=mp
  )
}

coef_extract <- function(fit, outcome, period, map_type) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]

  out <- data.frame()

  for (term in PHYS) {
    if (!(term %in% rownames(sm))) next

    out <- rbind(
      out,
      data.frame(
        outcome=outcome,
        period=period,
        map_type=map_type,
        term=term,
        beta=sm[term,"Estimate"],
        se=sm[term,"Std. Error"],
        ci_low=ci[term,1],
        ci_high=ci[term,2],
        p=sm[term,pcol],
        stringsAsFactors=FALSE
      )
    )
  }

  out
}

all_coef <- data.frame()
all_perf <- data.frame()
all_n <- data.frame()

for (oname in names(OUTCOMES)) {

  yname <- OUTCOMES[[oname]]
  need <- unique(c(
    yname, PHYS, COVARS,
    "SURVEY_WT","STRATUM","PSU",
    "ELIGIBLE_ADULT_NONPREG"
  ))

  # Discovery map: same transportable formula used by Script 73.
  dd <- d[
    d$PERIOD=="2005-2008" &
    d$ELIGIBLE_ADULT_NONPREG==1,
  ]

  okd <- complete.cases(dd[,need]) &
         is.finite(dd$SURVEY_WT) &
         dd$SURVEY_WT > 0

  dd$MODEL_OK <- okd

  des_full_d <- svydesign(
    ids=~PSU,
    strata=~STRATUM,
    weights=~SURVEY_WT,
    nest=TRUE,
    data=dd
  )

  des_d <- subset(des_full_d, MODEL_OK)

  fit_frozen <- svyglm(
    make_formula(yname, c(COVARS,PHYS)),
    design=des_d,
    family=gaussian()
  )

  all_coef <- rbind(
    all_coef,
    coef_extract(
      fit_frozen,
      oname,
      "2005-2008",
      "frozen_discovery"
    )
  )

  # Each period: evaluate the frozen map and independently refit h.
  # X->Z remains fixed throughout.
  for (period in PERIODS) {

    dp <- d[
      d$PERIOD==period &
      d$ELIGIBLE_ADULT_NONPREG==1,
    ]

    okp <- complete.cases(dp[,need]) &
           is.finite(dp$SURVEY_WT) &
           dp$SURVEY_WT > 0

    dp$MODEL_OK <- okp
    dm <- dp[okp,]

    if (nrow(dm) < 50) next

    des_full_p <- svydesign(
      ids=~PSU,
      strata=~STRATUM,
      weights=~SURVEY_WT,
      nest=TRUE,
      data=dp
    )

    des_p <- subset(des_full_p, MODEL_OK)

    fit_refit <- svyglm(
      make_formula(yname, c(COVARS,PHYS)),
      design=des_p,
      family=gaussian()
    )

    all_coef <- rbind(
      all_coef,
      coef_extract(
        fit_refit,
        oname,
        period,
        "period_refit"
      )
    )

    pred_frozen <- as.numeric(
      predict(
        fit_frozen,
        newdata=dm,
        type="response"
      )
    )

    pred_refit <- as.numeric(
      predict(
        fit_refit,
        newdata=dm,
        type="response"
      )
    )

    mf <- weighted_metrics(
      dm[[yname]],
      pred_frozen,
      dm$SURVEY_WT
    )

    mr <- weighted_metrics(
      dm[[yname]],
      pred_refit,
      dm$SURVEY_WT
    )

    all_perf <- rbind(
      all_perf,
      data.frame(
        outcome=oname,
        period=period,
        map="frozen_2005_08",
        t(mf),
        stringsAsFactors=FALSE
      ),
      data.frame(
        outcome=oname,
        period=period,
        map="period_refit",
        t(mr),
        stringsAsFactors=FALSE
      )
    )

    all_n <- rbind(
      all_n,
      data.frame(
        outcome=oname,
        period=period,
        n=nrow(dm),
        design_df=degf(des_p),
        stringsAsFactors=FALSE
      )
    )
  }
}

write.csv(
  all_coef,
  file.path(out_dir,"78_period_specific_ZY_coefficients.csv"),
  row.names=FALSE
)

write.csv(
  all_perf,
  file.path(out_dir,"78_frozen_vs_refit_performance.csv"),
  row.names=FALSE
)

write.csv(
  all_n,
  file.path(out_dir,"78_ZY_sample_audit.csv"),
  row.names=FALSE
)
'''

r_path = RESULTS / "78_ZY_stability_audit.R"
r_path.write_text(R_CODE, encoding="utf-8")

rscript = shutil.which("Rscript")
if rscript is None:
    raise FileNotFoundError("Rscript not found")

proc = subprocess.run(
    [rscript, str(r_path), str(INPUT), str(RESULTS)],
    text=True,
    capture_output=True,
)

if proc.returncode != 0:
    print(proc.stdout)
    print(proc.stderr)
    raise SystemExit(1)

coef = pd.read_csv(RESULTS / "78_period_specific_ZY_coefficients.csv")
perf = pd.read_csv(RESULTS / "78_frozen_vs_refit_performance.csv")
ns = pd.read_csv(RESULTS / "78_ZY_sample_audit.csv")

# Guardrail: Script 78 must reproduce Script 73's discovery Z->Y map.
ref73 = RESULTS / "73_frozen_phenotype_map_coefficients.csv"
if not ref73.exists():
    raise FileNotFoundError(
        "73_frozen_phenotype_map_coefficients.csv not found. Run Script 73 first."
    )

r73 = pd.read_csv(ref73)
r73 = r73[r73["term"].isin(PHYS)].copy()

anchor = coef[
    (coef["period"] == "2005-2008")
    & (coef["map_type"] == "frozen_discovery")
].copy()

chk = anchor.merge(
    r73[["outcome","term","beta"]].rename(columns={"beta":"beta_script73"}),
    on=["outcome","term"],
    how="left"
)

chk["abs_diff"] = (chk["beta"] - chk["beta_script73"]).abs()

if chk["beta_script73"].isna().any():
    raise RuntimeError("Could not match all Script-78 discovery terms to Script 73.")

maxdiff = float(chk["abs_diff"].max())

print("=" * 112)
print("SCRIPT 78 — Z -> Y STABILITY AUDIT")
print("=" * 112)
print(f"Input: {INPUT}")
print(f"SCRIPT-73 ANCHOR CHECK: max |beta_78 - beta_73| = {maxdiff:.3e}")

if maxdiff > 1e-8:
    print(chk.to_string(index=False))
    raise RuntimeError(
        "STOP: Script 78 does not reproduce Script 73's frozen discovery map."
    )

print("PASS — Script 78 reproduces the frozen Script-73 Z -> Y discovery map exactly.")
print()

# Coefficient drift table.
rows = []

for outcome in OUTCOME_ORDER:
    q = coef[coef["outcome"] == outcome]

    base = q[
        (q["period"] == "2005-2008")
        & (q["map_type"] == "frozen_discovery")
    ].set_index("term")

    for term in PHYS:
        if term not in base.index:
            continue

        b0 = float(base.loc[term, "beta"])

        row = {
            "outcome": outcome,
            "component": term,
            "beta_2005_08": b0,
        }

        for period, tag in [
            ("2009-2018","2009_18"),
            ("2021-2023","2021_23"),
        ]:
            z = q[
                (q["period"] == period)
                & (q["map_type"] == "period_refit")
                & (q["term"] == term)
            ]

            if z.empty:
                row[f"beta_{tag}"] = np.nan
                row[f"sign_match_{tag}"] = np.nan
                row[f"abs_drift_{tag}"] = np.nan
                row[f"ratio_to_discovery_{tag}"] = np.nan
                continue

            b = float(z.iloc[0]["beta"])

            row[f"beta_{tag}"] = b
            row[f"sign_match_{tag}"] = (
                np.sign(b) == np.sign(b0)
                if abs(b0) > 1e-12 and abs(b) > 1e-12
                else np.nan
            )
            row[f"abs_drift_{tag}"] = b - b0
            row[f"ratio_to_discovery_{tag}"] = (
                b / b0 if abs(b0) > 1e-12 else np.nan
            )

        rows.append(row)

drift = pd.DataFrame(rows)
drift.to_csv(
    RESULTS / "78_component_coefficient_drift.csv",
    index=False
)

# Frozen-vs-refit performance gap.
perf_rows = []

for outcome in OUTCOME_ORDER:
    for period in PERIOD_ORDER:
        q = perf[
            (perf["outcome"] == outcome)
            & (perf["period"] == period)
        ]

        fz = q[q["map"] == "frozen_2005_08"]
        rf = q[q["map"] == "period_refit"]

        if fz.empty or rf.empty:
            continue

        f = fz.iloc[0]
        r = rf.iloc[0]

        perf_rows.append({
            "outcome": outcome,
            "period": period,
            "frozen_R2": f["weighted_R2"],
            "refit_R2": r["weighted_R2"],
            "R2_gain_from_refit": r["weighted_R2"] - f["weighted_R2"],
            "frozen_calibration_intercept": f["calibration_intercept"],
            "frozen_calibration_slope": f["calibration_slope"],
            "frozen_bias": f["mean_bias"],
            "frozen_corr": f["weighted_corr"],
        })

perf_gap = pd.DataFrame(perf_rows)
perf_gap.to_csv(
    RESULTS / "78_ZY_stability_summary.csv",
    index=False
)

print("COEFFICIENT DRIFT")
print(drift.to_string(index=False))

print()
print("FROZEN MAP vs SAME-PERIOD REFIT")
print(perf_gap.to_string(index=False))

print()
print("READING GUIDE")
print("  sign_match=True  -> component points in the same phenotype direction.")
print("  ratio ~ 1        -> coefficient magnitude is close to discovery.")
print("  R2 gain ~ 0      -> refitting the phenotype map adds little.")
print("  slope ~ 1        -> frozen effect scale is well calibrated.")
print("  intercept ~ 0    -> frozen prediction level needs little recalibration.")
print()
print("This script does NOT change X->Z, does NOT change the phenotype definition,")
print("and does NOT make causal claims. It audits only Z->Y temporal stability.")
print("=" * 112)
