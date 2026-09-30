from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

MASTER_CANDIDATES = [
    RESULTS / "63_deep_ddx_input.csv",
    ROOT / "63_deep_ddx_input.csv",
    Path("/mnt/data/63_deep_ddx_input.csv"),
]
RAW_CANDIDATES = [
    RESULTS / "69_temporal_raw_bank.csv",
    ROOT / "69_temporal_raw_bank.csv",
    Path("/mnt/data/69_temporal_raw_bank.csv"),
]
TRANSFORM_CANDIDATES = [
    RESULTS / "72_shared_private_frozen_transform.npz",
    ROOT / "72_shared_private_frozen_transform.npz",
    Path("/mnt/data/72_shared_private_frozen_transform(1).npz"),
    Path("/mnt/data/72_shared_private_frozen_transform.npz"),
]

MASTER = next((p for p in MASTER_CANDIDATES if p.exists()), None)
RAW = next((p for p in RAW_CANDIDATES if p.exists()), None)
TRANSFORM = next((p for p in TRANSFORM_CANDIDATES if p.exists()), None)

if MASTER is None:
    raise FileNotFoundError("Could not find 63_deep_ddx_input.csv")
if RAW is None:
    raise FileNotFoundError("Could not find 69_temporal_raw_bank.csv")
if TRANSFORM is None:
    raise FileNotFoundError("Could not find 72_shared_private_frozen_transform.npz")

A_VARS = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G_VARS = ["LBXGH", "LBXGLU", "LOG_IN"]
PHYS = [
    "SP_SHARED1", "SP_SHARED2",
    "SP_DISCORD1", "SP_DISCORD2",
    "SP_A_PRIVATE3", "SP_A_PRIVATE4",
    "SP_G_PRIVATE3",
]
COVARS = [
    "RIDAGEYR", "RIAGENDR", "RACE", "INDFMPIR",
    "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021",
]
OUTCOMES = {
    "SOMATIC": "SOMATIC_SCORE",
    "COGAFF": "COGAFF_SUM",
    "PHQ9_TOTAL": "PHQ9_TOTAL",
}


def transform_block(XA_raw, XG_raw, fit, shared_rank):
    A = (XA_raw - fit["A_mean"]) / fit["A_sd"]
    G = (XG_raw - fit["G_mean"]) / fit["G_sd"]
    U = A @ fit["WA"]
    V = G @ fit["WG"]
    rho = fit["rho"]

    out = {}
    for j in range(3):
        k = j + 1
        if j < shared_rank:
            out[f"SP_SHARED{k}"] = (U[:, j] + V[:, j]) / np.sqrt(2 * (1 + rho[j]))
            out[f"SP_DISCORD{k}"] = (U[:, j] - V[:, j]) / np.sqrt(2 * (1 - rho[j]))
        else:
            out[f"SP_A_PRIVATE{k}"] = U[:, j]
            out[f"SP_G_PRIVATE{k}"] = V[:, j]

    out["SP_A_PRIVATE4"] = U[:, 3]
    return pd.DataFrame(out)


master = pd.read_csv(MASTER)
raw = pd.read_csv(RAW)

drop_phys = [c for c in set(A_VARS + G_VARS + ["LBXIN"]) if c in master.columns]
d = master.drop(columns=drop_phys).merge(raw, on="SEQN", how="left", validate="one_to_one")

npz = np.load(TRANSFORM)
fit = {k: npz[k] for k in ["A_mean", "A_sd", "G_mean", "G_sd", "WA", "WG", "rho"]}
shared_rank = int(npz["shared_rank"].ravel()[0])

for c in PHYS:
    d[c] = np.nan

for period in ["2005-2008", "2009-2018", "2021-2023"]:
    mask = (
        (d["PERIOD"] == period)
        & (d["ELIGIBLE_ADULT_NONPREG"] == 1)
        & d[A_VARS + G_VARS].notna().all(axis=1)
    )
    if not mask.any():
        continue

    z = transform_block(
        d.loc[mask, A_VARS].to_numpy(float),
        d.loc[mask, G_VARS].to_numpy(float),
        fit,
        shared_rank,
    )
    d.loc[mask, PHYS] = z[PHYS].to_numpy()

model_input = RESULTS / "73_frozen_phenotype_map_input.csv"
d.to_csv(model_input, index=False)

r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
out_dir <- args[2]

suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3")) d[[v]] <- factor(d[[v]])

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
  y <- y[ok]; pred <- pred[ok]; w <- w[ok]
  w <- w / sum(w)

  my <- sum(w*y)
  mp <- sum(w*pred)
  vy <- sum(w*(y-my)^2)
  vp <- sum(w*(pred-mp)^2)
  covyp <- sum(w*(y-my)*(pred-mp))

  rmse <- sqrt(sum(w*(y-pred)^2))
  mae <- sum(w*abs(y-pred))
  r2 <- ifelse(vy > 0, 1 - sum(w*(y-pred)^2)/vy, NA)
  corr <- ifelse(vy > 0 && vp > 0, covyp/sqrt(vy*vp), NA)
  slope <- ifelse(vp > 0, covyp/vp, NA)
  intercept <- ifelse(is.finite(slope), my - slope*mp, NA)

  c(
    weighted_R2=r2,
    weighted_corr=corr,
    RMSE=rmse,
    NRMSE=ifelse(vy>0, rmse/sqrt(vy), NA),
    MAE=mae,
    mean_bias=sum(w*(pred-y)),
    calibration_intercept=intercept,
    calibration_slope=slope,
    observed_mean=my,
    predicted_mean=mp
  )
}

all_metrics <- data.frame()
all_coefs <- data.frame()
all_n <- data.frame()

for (oname in names(OUTCOMES)) {
  yname <- OUTCOMES[[oname]]

  dd <- d[
    d$PERIOD=="2005-2008" &
    d$ELIGIBLE_ADULT_NONPREG==1,
  ]

  need <- unique(c(
    yname, PHYS, COVARS,
    "SURVEY_WT","STRATUM","PSU"
  ))
  ok <- complete.cases(dd[, need]) &
        is.finite(dd$SURVEY_WT) &
        dd$SURVEY_WT > 0
  dd$MODEL_OK <- ok

  des_full <- svydesign(
    ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT,
    nest=TRUE, data=dd
  )
  des <- subset(des_full, MODEL_OK)

  fit0 <- svyglm(
    make_formula(yname, COVARS),
    design=des, family=gaussian()
  )
  fit1 <- svyglm(
    make_formula(yname, c(COVARS, PHYS)),
    design=des, family=gaussian()
  )

  sm <- summary(fit1)$coefficients
  coef_df <- data.frame(
    outcome=oname,
    term=rownames(sm),
    beta=sm[, "Estimate"],
    se=sm[, "Std. Error"],
    row.names=NULL
  )
  all_coefs <- rbind(all_coefs, coef_df)

  for (period in PERIODS) {
    dp <- d[
      d$PERIOD==period &
      d$ELIGIBLE_ADULT_NONPREG==1,
    ]

    okp <- complete.cases(dp[, need]) &
           is.finite(dp$SURVEY_WT) &
           dp$SURVEY_WT > 0
    dm <- dp[okp,]

    if (nrow(dm) < 50) next

    p0 <- as.numeric(predict(fit0, newdata=dm, type="response"))
    p1 <- as.numeric(predict(fit1, newdata=dm, type="response"))

    m0 <- wmetrics(dm[[yname]], p0, dm$SURVEY_WT)
    m1 <- wmetrics(dm[[yname]], p1, dm$SURVEY_WT)

    all_metrics <- rbind(
      all_metrics,
      data.frame(
        outcome=oname,
        period=period,
        map="covariates_only_frozen_2005_08",
        t(m0),
        delta_R2_vs_covariates=0,
        stringsAsFactors=FALSE
      ),
      data.frame(
        outcome=oname,
        period=period,
        map="shared_private_plus_covariates_frozen_2005_08",
        t(m1),
        delta_R2_vs_covariates=m1["weighted_R2"]-m0["weighted_R2"],
        stringsAsFactors=FALSE
      )
    )

    all_n <- rbind(
      all_n,
      data.frame(outcome=oname, period=period, n=nrow(dm))
    )
  }
}

write.csv(
  all_metrics,
  file.path(out_dir,"73_frozen_phenotype_map_metrics.csv"),
  row.names=FALSE
)
write.csv(
  all_coefs,
  file.path(out_dir,"73_frozen_phenotype_map_coefficients.csv"),
  row.names=FALSE
)
write.csv(
  all_n,
  file.path(out_dir,"73_frozen_phenotype_map_n.csv"),
  row.names=FALSE
)
'''

r_path = RESULTS / "73_frozen_phenotype_map.R"
r_path.write_text(r_code, encoding="utf-8")

rscript = shutil.which("Rscript")
status = "not_run"
error = ""

if rscript:
    proc = subprocess.run(
        [rscript, str(r_path), str(model_input), str(RESULTS)],
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        status = "pass"
    else:
        status = "fail"
        error = (proc.stderr or proc.stdout)[-4000:]
else:
    status = "Rscript_not_found"

manifest = {
    "script": "73_frozen_phenotype_map_time_travel.py",
    "question": "Does the phenotype map learned in NHANES 2005-08 survive when both T and h are frozen and applied to later NHANES periods?",
    "decomposition_source": str(TRANSFORM),
    "T_fit_period": "2005-2008 only",
    "h_fit_period": "2005-2008 only",
    "future_T_refit": False,
    "future_h_refit": False,
    "adult_nonpregnant_only": True,
    "physiology_terms": PHYS,
    "covariates": COVARS,
    "outcomes": OUTCOMES,
    "cycle_as_predictor": False,
    "cycle_reason": "future cycle categories are unseen; a frozen transport map cannot depend on them",
    "r_status": status,
    "r_error": error,
}
(RESULTS / "73_frozen_phenotype_map_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

if status != "pass":
    print("=" * 76)
    print("SCRIPT 73 PHENOTYPE-MAP TIME TRAVEL")
    print("=" * 76)
    print(f"R status: {status}")
    if error:
        print(error)
    print("=" * 76)
    raise SystemExit(1)

metrics = pd.read_csv(RESULTS / "73_frozen_phenotype_map_metrics.csv")
ns = pd.read_csv(RESULTS / "73_frozen_phenotype_map_n.csv")

print("=" * 82)
print("SCRIPT 73 — FROZEN PHENOTYPE-MAP TIME TRAVEL")
print("=" * 82)
print("TRAINING: T and h learned from 2005-08 only.")
print("TRANSFER: 2009-18 and 2021-23 use NO representation or coefficient refit.")
print(f"Shared/private rank from Script 72 = {shared_rank}")
print()

for outcome in ["SOMATIC", "COGAFF", "PHQ9_TOTAL"]:
    print(f"[{outcome}]")
    sub = metrics[metrics["outcome"] == outcome]
    for period in ["2005-2008", "2009-2018", "2021-2023"]:
        full = sub[
            (sub["period"] == period)
            & (sub["map"] == "shared_private_plus_covariates_frozen_2005_08")
        ]
        base = sub[
            (sub["period"] == period)
            & (sub["map"] == "covariates_only_frozen_2005_08")
        ]
        if full.empty or base.empty:
            continue

        f = full.iloc[0]
        b = base.iloc[0]
        nrow = ns[
            (ns["outcome"] == outcome)
            & (ns["period"] == period)
        ]["n"]
        n = int(nrow.iloc[0]) if len(nrow) else -1

        print(
            f"  {period} n={n}: "
            f"R2={f.weighted_R2:.4f} "
            f"(cov={b.weighted_R2:.4f}, delta={f.delta_R2_vs_covariates:+.4f}); "
            f"corr={f.weighted_corr:.4f}; "
            f"NRMSE={f.NRMSE:.4f}; "
            f"bias={f.mean_bias:+.4f}; "
            f"calibration slope={f.calibration_slope:.3f}"
        )
    print()

print("READING GUIDE")
print("  delta R2 > 0  : frozen physiology adds information beyond frozen covariates.")
print("  corr retained : ranking relationship survives temporal transfer.")
print("  slope ~ 1     : effect scale/calibration is stable.")
print("  bias ~ 0      : average prediction level transfers well.")
print("  negative R2   : frozen predictions are worse than using that period's mean.")
print()
print("PASS Script 73 complete.")
print("=" * 82)
