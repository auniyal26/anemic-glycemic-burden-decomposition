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
        "Need EDA/Results/73_frozen_phenotype_map_input.csv. Run Script 73 first."
    )

R_CODE = r"""
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

wmetrics <- function(y, pred, w) {
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
  r2 <- ifelse(vy > 0, 1 - sse/vy, NA_real_)
  corr <- ifelse(vy > 0 && vp > 0, covyp/sqrt(vy*vp), NA_real_)
  slope <- ifelse(vp > 0, covyp/vp, NA_real_)
  intercept <- ifelse(is.finite(slope), my - slope*mp, NA_real_)

  c(
    weighted_R2=r2,
    weighted_corr=corr,
    RMSE=sqrt(sse),
    mean_bias=sum(w*(pred-y)),
    calibration_intercept=intercept,
    calibration_slope=slope
  )
}

coef_rows <- data.frame()
metric_rows <- data.frame()
summary_rows <- data.frame()

for (oname in names(OUTCOMES)) {

  yname <- OUTCOMES[[oname]]

  need <- unique(c(
    yname, PHYS, COVARS,
    "SURVEY_WT","STRATUM","PSU",
    "ELIGIBLE_ADULT_NONPREG"
  ))

  # ----------------------------------------------------------
  # 1) Freeze the original 2005-08 Z -> Y map.
  # ----------------------------------------------------------
  dd <- d[
    d$PERIOD=="2005-2008" &
    d$ELIGIBLE_ADULT_NONPREG==1,
  ]

  okd <- complete.cases(dd[,need]) &
         is.finite(dd$SURVEY_WT) &
         dd$SURVEY_WT > 0

  dd$MODEL_OK <- okd

  des_full_d <- svydesign(
    ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT,
    nest=TRUE, data=dd
  )
  des_d <- subset(des_full_d, MODEL_OK)

  fit_frozen <- svyglm(
    make_formula(yname, c(COVARS,PHYS)),
    design=des_d,
    family=gaussian()
  )

  # ----------------------------------------------------------
  # 2) Travel to each period.
  #    A) frozen GPS prediction
  #    B) GPS recalibration only: Y ~ alpha + gamma * frozen_prediction
  #    C) full same-period Z -> Y refit, only as the upper benchmark
  # ----------------------------------------------------------
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

    frozen_pred <- as.numeric(
      predict(fit_frozen, newdata=dm, type="response")
    )

    dm$FROZEN_PRED <- frozen_pred

    des_m <- svydesign(
      ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT,
      nest=TRUE, data=dm
    )

    # GPS v1: ONLY intercept + slope are allowed to move.
    fit_cal <- svyglm(
      as.formula(paste(yname, "~ FROZEN_PRED")),
      design=des_m,
      family=gaussian()
    )

    cal_pred <- as.numeric(
      predict(fit_cal, newdata=dm, type="response")
    )

    # Full period-specific refit: benchmark for how much structure
    # cannot be recovered by simple GPS recalibration.
    fit_refit <- svyglm(
      make_formula(yname, c(COVARS,PHYS)),
      design=des_m,
      family=gaussian()
    )

    refit_pred <- as.numeric(
      predict(fit_refit, newdata=dm, type="response")
    )

    mf <- wmetrics(dm[[yname]], frozen_pred, dm$SURVEY_WT)
    mc <- wmetrics(dm[[yname]], cal_pred, dm$SURVEY_WT)
    mr <- wmetrics(dm[[yname]], refit_pred, dm$SURVEY_WT)

    metric_rows <- rbind(
      metric_rows,
      data.frame(
        outcome=oname, period=period, model="frozen",
        t(mf), stringsAsFactors=FALSE
      ),
      data.frame(
        outcome=oname, period=period, model="gps_recalibrated",
        t(mc), stringsAsFactors=FALSE
      ),
      data.frame(
        outcome=oname, period=period, model="full_period_refit",
        t(mr), stringsAsFactors=FALSE
      )
    )

    sm <- summary(fit_cal)$coefficients
    ci <- suppressMessages(confint(fit_cal))
    df <- degf(des_m)

    alpha <- unname(coef(fit_cal)[1])
    gamma <- unname(coef(fit_cal)[2])
    se_alpha <- sm[1,"Std. Error"]
    se_gamma <- sm[2,"Std. Error"]

    t_gamma1 <- (gamma - 1) / se_gamma
    p_gamma1 <- 2 * pt(-abs(t_gamma1), df=df)

    coef_rows <- rbind(
      coef_rows,
      data.frame(
        outcome=oname,
        period=period,
        n=nrow(dm),
        design_df=df,
        alpha=alpha,
        alpha_se=se_alpha,
        alpha_ci_low=ci[1,1],
        alpha_ci_high=ci[1,2],
        gamma=gamma,
        gamma_se=se_gamma,
        gamma_ci_low=ci[2,1],
        gamma_ci_high=ci[2,2],
        p_gamma_equals_1=p_gamma1,
        stringsAsFactors=FALSE
      )
    )

    total_possible_gain <- mr["weighted_R2"] - mf["weighted_R2"]
    recovered_gain <- mc["weighted_R2"] - mf["weighted_R2"]

    recovery_fraction <- ifelse(
      is.finite(total_possible_gain) && abs(total_possible_gain) > 1e-12,
      recovered_gain / total_possible_gain,
      NA_real_
    )

    summary_rows <- rbind(
      summary_rows,
      data.frame(
        outcome=oname,
        period=period,
        frozen_R2=mf["weighted_R2"],
        gps_R2=mc["weighted_R2"],
        full_refit_R2=mr["weighted_R2"],
        gps_gain_over_frozen=recovered_gain,
        full_refit_gain_over_frozen=total_possible_gain,
        fraction_of_refit_gain_recovered_by_GPS=recovery_fraction,
        remaining_R2_gap_after_GPS=mr["weighted_R2"] - mc["weighted_R2"],
        alpha=alpha,
        gamma=gamma,
        p_gamma_equals_1=p_gamma1,
        stringsAsFactors=FALSE
      )
    )
  }
}

write.csv(
  coef_rows,
  file.path(out_dir,"79_gps_recalibration_coefficients.csv"),
  row.names=FALSE
)

write.csv(
  metric_rows,
  file.path(out_dir,"79_gps_model_metrics.csv"),
  row.names=FALSE
)

write.csv(
  summary_rows,
  file.path(out_dir,"79_gps_summary.csv"),
  row.names=FALSE
)
"""

r_path = RESULTS / "79_time_travel_gps_recalibration.R"
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

summary = pd.read_csv(RESULTS / "79_gps_summary.csv")
coefs = pd.read_csv(RESULTS / "79_gps_recalibration_coefficients.csv")

print("=" * 108)
print("SCRIPT 79 — TIME-TRAVEL GPS v1")
print("=" * 108)
print("Frozen physiology coordinates: unchanged.")
print("Frozen 2005-08 Z -> Y map: unchanged.")
print("Only GPS recalibration alpha + gamma is allowed to move in later periods.")
print()

for outcome in ["SOMATIC","COGAFF","PHQ9_TOTAL"]:
    print(f"[{outcome}]")
    s = summary[summary["outcome"] == outcome]

    for period in ["2005-2008","2009-2018","2021-2023"]:
        q = s[s["period"] == period]
        if q.empty:
            continue
        r = q.iloc[0]

        print(
            f"  {period}: "
            f"frozen R2={r.frozen_R2:.4f} | "
            f"GPS R2={r.gps_R2:.4f} | "
            f"full-refit R2={r.full_refit_R2:.4f} | "
            f"GPS recovered={r.fraction_of_refit_gain_recovered_by_GPS:.3f} | "
            f"remaining gap={r.remaining_R2_gap_after_GPS:.4f} | "
            f"alpha={r.alpha:+.3f}, gamma={r.gamma:.3f}"
        )
    print()

print("HOW TO READ THIS")
print("  GPS recovered ~1   -> intercept/slope recalibration explains most temporal drift.")
print("  GPS recovered ~0   -> simple recalibration does little; component relationships changed.")
print("  remaining gap ~0   -> little benefit from allowing every Z coefficient to refit.")
print("  gamma near 1       -> phenotype scale is close to discovery.")
print()
print("IMPORTANT")
print("  This is a structural temporal-transport test, not a causal model.")
print("  Recalibration is estimated and evaluated within each target period.")
print("  If GPS v1 looks promising, the next test should validate it out-of-period/cross-cycle.")
print("=" * 108)
