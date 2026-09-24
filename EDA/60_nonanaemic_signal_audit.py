#!/usr/bin/env python
# =============================================================================
# 60_nonanaemic_signal_audit.py
# =============================================================================

from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

LOCKED = ROOT / "Results" / "47_domain_corrected_AG_input.csv"
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"
RLIB = ROOT / "R_library"

OUT = ROOT / "EDA" / "Results"
FIG = ROOT / "EDA" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DISC = {"0506": "D", "0708": "E"}
TEMP = {
    "0910": "F",
    "1112": "G",
    "1314": "H",
    "1516": "I",
    "1718": "J",
    "2123": "L",
}
ALL_CYCLES = list(DISC) + list(TEMP)
PERIODS = ["2005-2008", "2009-2018", "2021-2023"]

A_PCS = [f"A_OI_PC{i}_FZ" for i in range(1, 4)]
G_PCS = [f"G3_OI_PC{i}_FZ" for i in range(1, 4)]

X3 = [
    "RIDAGEYR", "RIAGENDR", "RACE", "INDFMPIR",
    "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021",
]

CBC = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
NON_HB_CBC = ["LBXRBCSI", "LBXMCVSI", "LBXRDW"]

OUTCOMES = [
    ("SOMATIC_SCORE", "DOMAIN_SOMATIC_SCORE_X3"),
    ("PHQ9_TOTAL", "DOMAIN_PHQ9_TOTAL_X3"),
    ("COGAFF_SUM", "DOMAIN_COGAFF_SUM_X3"),
]


def base_for(cycle):
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def suffix_for(cycle):
    return (DISC if cycle in DISC else TEMP)[cycle]


def xpt_path(base: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base / name
        if p.exists():
            return p
    matches = list(base.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base}")


def weighted_mean(x, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return np.sum(x[ok] * w[ok]) / np.sum(w[ok])


def weighted_sd(x, w):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.nan
    x = x[ok]
    w = w[ok]
    mu = np.sum(w * x) / np.sum(w)
    return np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w))


def weighted_percent(mask, w):
    mask = pd.Series(mask).fillna(False).astype(bool).to_numpy()
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(float)
    ok = np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return 100 * np.sum(w[ok] * mask[ok]) / np.sum(w[ok])


if not LOCKED.exists():
    raise FileNotFoundError(f"Missing locked input: {LOCKED}")

d = pd.read_csv(LOCKED)

required = (
    ["SEQN", "CYCLE", "PERIOD", "SURVEY_WT", "STRATUM", "PSU",
     "ELIGIBLE_ADULT_NONPREG", "A", "G_HBA1C"]
    + X3 + A_PCS + G_PCS
    + [x for pair in OUTCOMES for x in pair]
)

missing = [c for c in required if c not in d.columns]
if missing:
    raise ValueError(f"Locked input missing required columns: {missing}")

d["SEQN"] = pd.to_numeric(d["SEQN"], errors="coerce")
d["CYCLE"] = d["CYCLE"].astype(str).str.zfill(4)

cbc_frames = []

for cycle in ALL_CYCLES:
    base = base_for(cycle)
    suffix = suffix_for(cycle)
    p = xpt_path(base, f"CBC_{suffix}")
    cbc = pd.read_sas(p, format="xport")

    missing_cbc = [c for c in CBC if c not in cbc.columns]
    if missing_cbc:
        raise ValueError(f"{cycle}: CBC missing {missing_cbc}")

    q = cbc[["SEQN"] + CBC].copy()
    q["CYCLE"] = cycle
    cbc_frames.append(q)

cbc = pd.concat(cbc_frames, ignore_index=True)
cbc["SEQN"] = pd.to_numeric(cbc["SEQN"], errors="coerce")
cbc["CYCLE"] = cbc["CYCLE"].astype(str).str.zfill(4)

d = d.merge(
    cbc,
    on=["SEQN", "CYCLE"],
    how="left",
    validate="one_to_one",
)

d["HB_THRESHOLD"] = np.where(
    d["RIAGENDR"].eq(1),
    13.0,
    np.where(d["RIAGENDR"].eq(2), 12.0, np.nan),
)

d["HB_MARGIN"] = pd.to_numeric(d["LBXHGB"], errors="coerce") - d["HB_THRESHOLD"]

d["NONANAEMIC"] = (
    pd.to_numeric(d["A"], errors="coerce").abs() < 1e-12
).astype(int)

d["ANAEMIC"] = (d["NONANAEMIC"] == 0).astype(int)

floor_rows = []

for period in PERIODS:
    q = d[
        d["PERIOD"].eq(period)
        & d["DOMAIN_SOMATIC_SCORE_X3"].eq(1)
    ].copy()

    floor_rows.append({
        "period": period,
        "analytic_n": len(q),
        "nonanaemic_A_eq_0_n": int(q["NONANAEMIC"].sum()),
        "nonanaemic_A_eq_0_percent": 100 * q["NONANAEMIC"].mean(),
        "nonanaemic_A_eq_0_weighted_percent":
            weighted_percent(q["NONANAEMIC"].eq(1), q["SURVEY_WT"]),
        "anaemic_A_gt_0_n": int(q["ANAEMIC"].sum()),
        "anaemic_A_gt_0_percent": 100 * q["ANAEMIC"].mean(),
        "anaemic_A_gt_0_weighted_percent":
            weighted_percent(q["ANAEMIC"].eq(1), q["SURVEY_WT"]),
    })

floor = pd.DataFrame(floor_rows)
floor.to_csv(OUT / "60_scalar_floor_summary.csv", index=False)

variation_rows = []

for period in PERIODS:
    q = d[
        d["PERIOD"].eq(period)
        & d["DOMAIN_SOMATIC_SCORE_X3"].eq(1)
        & d["NONANAEMIC"].eq(1)
    ].copy()

    for v in A_PCS + ["HB_MARGIN"] + NON_HB_CBC:
        x = pd.to_numeric(q[v], errors="coerce")

        variation_rows.append({
            "period": period,
            "variable": v,
            "n": int(x.notna().sum()),
            "mean": x.mean(),
            "sd": x.std(ddof=1),
            "weighted_mean": weighted_mean(x, q["SURVEY_WT"]),
            "weighted_sd": weighted_sd(x, q["SURVEY_WT"]),
            "min": x.min(),
            "max": x.max(),
        })

variation = pd.DataFrame(variation_rows)
variation.to_csv(
    OUT / "60_nonanaemic_component_variation.csv",
    index=False,
)

model_input = OUT / "60_nonanaemic_model_input.csv"
d.to_csv(model_input, index=False)

rscript = shutil.which("Rscript")

if rscript is None:
    candidates = sorted(
        Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe")
    )
    if candidates:
        rscript = str(candidates[-1])

if rscript is None:
    raise FileNotFoundError("Rscript not found")

r_code = r'''
args <- commandArgs(trailingOnly=TRUE)

input_file <- args[1]
out_dir <- args[2]
rlib <- args[3]

.libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)

for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) {
  d[[v]] <- factor(d[[v]])
}

A_PCS <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G_PCS <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")

X3 <- c(
  "RIDAGEYR","RIAGENDR","RACE","INDFMPIR",
  "EDUC3","SMOKING3","BMXBMI","EGFR_2021"
)

NON_HB <- c("LBXRBCSI","LBXMCVSI","LBXRDW")

make_formula <- function(outcome, terms) {
  as.formula(paste(outcome, "~", paste(terms, collapse=" + ")))
}

weighted_r2 <- function(fit, domain_data, outcome) {
  pred <- as.numeric(predict(fit, newdata=domain_data, type="response"))
  y <- domain_data[[outcome]]
  w <- domain_data$SURVEY_WT

  ok <- is.finite(y) & is.finite(w) & is.finite(pred) & w > 0

  y <- y[ok]
  w <- w[ok]
  pred <- pred[ok]

  if (length(y) < 2) return(NA_real_)

  mu <- sum(w * y) / sum(w)
  sse <- sum(w * (y - pred)^2)
  sst <- sum(w * (y - mu)^2)

  if (sst <= 0) return(NA_real_)

  1 - sse / sst
}

safe_test <- function(fit, terms, period, outcome, label) {
  z <- tryCatch(
    regTermTest(
      fit,
      as.formula(paste("~", paste(terms, collapse=" + "))),
      method="Wald"
    ),
    error=function(e) NULL
  )

  if (is.null(z)) {
    return(
      data.frame(
        period=period,
        outcome=outcome,
        test=label,
        F=NA_real_,
        df_num=NA_real_,
        df_den=NA_real_,
        p=NA_real_,
        inference_status="unavailable"
      )
    )
  }

  ddf <- as.numeric(z$ddf)

  status <- ifelse(
    is.na(ddf),
    "unavailable",
    ifelse(ddf <= 3, "design_limited", "available")
  )

  data.frame(
    period=period,
    outcome=outcome,
    test=label,
    F=as.numeric(z$Ftest),
    df_num=as.numeric(z$df),
    df_den=ddf,
    p=as.numeric(z$p),
    inference_status=status
  )
}

extract_coef <- function(fit, period, outcome, model, terms) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]

  out <- data.frame()

  for (term in terms) {
    if (!(term %in% rownames(sm))) next

    out <- rbind(
      out,
      data.frame(
        period=period,
        outcome=outcome,
        model=model,
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

run_period <- function(dat, period, include_cycle) {
  des_full <- svydesign(
    ids=~PSU,
    strata=~STRATUM,
    weights=~SURVEY_WT,
    nest=TRUE,
    data=dat
  )

  covars <- X3

  if (include_cycle) {
    covars <- c(covars, "CYCLE")
  }

  tests <- data.frame()
  coefs <- data.frame()
  r2 <- data.frame()
  audit <- data.frame()

  outcome_domains <- list(
    SOMATIC_SCORE="DOMAIN_SOMATIC_SCORE_X3",
    PHQ9_TOTAL="DOMAIN_PHQ9_TOTAL_X3",
    COGAFF_SUM="DOMAIN_COGAFF_SUM_X3"
  )

  for (outcome in names(outcome_domains)) {
    domain_var <- outcome_domains[[outcome]]
    keep <- dat[[domain_var]] == 1 & dat$NONANAEMIC == 1

    des <- subset(des_full, keep)
    dom <- dat[keep, ]

    audit <- rbind(
      audit,
      data.frame(
        period=period,
        outcome=outcome,
        full_design_n=nrow(dat),
        nonanaemic_domain_n=nrow(dom),
        full_design_df=degf(des_full),
        nonanaemic_domain_df=degf(des)
      )
    )

    base_terms <- c(G_PCS, covars)

    fit_base <- svyglm(
      make_formula(outcome, base_terms),
      design=des,
      family=gaussian()
    )

    fit_hb <- svyglm(
      make_formula(outcome, c("HB_MARGIN", base_terms)),
      design=des,
      family=gaussian()
    )

    fit_apc <- svyglm(
      make_formula(outcome, c(A_PCS, base_terms)),
      design=des,
      family=gaussian()
    )

    fit_rawA <- svyglm(
      make_formula(outcome, c("HB_MARGIN", NON_HB, base_terms)),
      design=des,
      family=gaussian()
    )

    fit_hb_apc <- svyglm(
      make_formula(outcome, c("HB_MARGIN", A_PCS, base_terms)),
      design=des,
      family=gaussian()
    )

    tests <- rbind(
      tests,

      safe_test(
        fit_apc,
        A_PCS,
        period,
        outcome,
        "A_PCs_joint_within_nonanaemic_given_G_PCs_X"
      ),

      safe_test(
        fit_apc,
        A_PCS[2:3],
        period,
        outcome,
        "A_PC2_3_extra_beyond_A_PC1_within_nonanaemic"
      ),

      safe_test(
        fit_hb,
        "HB_MARGIN",
        period,
        outcome,
        "continuous_Hb_margin_given_G_PCs_X"
      ),

      safe_test(
        fit_rawA,
        NON_HB,
        period,
        outcome,
        "RBC_MCV_RDW_extra_beyond_Hb_margin_G_PCs_X"
      ),

      safe_test(
        fit_rawA,
        "HB_MARGIN",
        period,
        outcome,
        "Hb_margin_extra_beyond_RBC_MCV_RDW_G_PCs_X"
      ),

      safe_test(
        fit_hb_apc,
        A_PCS,
        period,
        outcome,
        "A_PCs_extra_beyond_continuous_Hb_margin_G_PCs_X"
      ),

      safe_test(
        fit_hb_apc,
        "HB_MARGIN",
        period,
        outcome,
        "continuous_Hb_margin_extra_beyond_A_PCs_G_PCs_X"
      )
    )

    coefs <- rbind(
      coefs,

      extract_coef(
        fit_apc,
        period,
        outcome,
        "A_PCs_nonanaemic",
        A_PCS
      ),

      extract_coef(
        fit_rawA,
        period,
        outcome,
        "raw_reduced_A_nonanaemic",
        c("HB_MARGIN", NON_HB)
      ),

      extract_coef(
        fit_hb_apc,
        period,
        outcome,
        "Hb_plus_A_PCs_nonanaemic",
        c("HB_MARGIN", A_PCS)
      )
    )

    fits <- list(
      base_Gpcs_X=fit_base,
      continuous_Hb_margin=fit_hb,
      frozen_A_PCs=fit_apc,
      raw_Hb_RBC_MCV_RDW=fit_rawA,
      Hb_margin_plus_A_PCs=fit_hb_apc
    )

    for (nm in names(fits)) {
      r2 <- rbind(
        r2,
        data.frame(
          period=period,
          outcome=outcome,
          model=nm,
          weighted_R2=weighted_r2(fits[[nm]], dom, outcome),
          n=nrow(dom),
          design_df=degf(des)
        )
      )
    }
  }

  list(
    tests=tests,
    coefs=coefs,
    r2=r2,
    audit=audit
  )
}

r1 <- run_period(
  d[d$PERIOD=="2005-2008",],
  "2005-2008",
  TRUE
)

r2 <- run_period(
  d[d$PERIOD=="2009-2018",],
  "2009-2018",
  TRUE
)

r3 <- run_period(
  d[d$PERIOD=="2021-2023",],
  "2021-2023",
  FALSE
)

write.csv(
  rbind(r1$tests,r2$tests,r3$tests),
  file.path(out_dir,"60_nonanaemic_block_tests.csv"),
  row.names=FALSE
)

write.csv(
  rbind(r1$coefs,r2$coefs,r3$coefs),
  file.path(out_dir,"60_nonanaemic_coefficients.csv"),
  row.names=FALSE
)

write.csv(
  rbind(r1$r2,r2$r2,r3$r2),
  file.path(out_dir,"60_nonanaemic_model_R2.csv"),
  row.names=FALSE
)

write.csv(
  rbind(r1$audit,r2$audit,r3$audit),
  file.path(out_dir,"60_nonanaemic_design_audit.csv"),
  row.names=FALSE
)
'''

r_path = OUT / "60_nonanaemic_signal_audit.R"
r_path.write_text(r_code, encoding="utf-8")

r_path_r = str(r_path).replace("\\", "/")

parse = subprocess.run(
    [str(rscript), "-e", f'parse(file="{r_path_r}")'],
    capture_output=True,
    text=True,
)

if parse.returncode != 0:
    print(parse.stdout[-3000:])
    print(parse.stderr[-5000:])
    raise SystemExit("Generated R code failed syntax check")

proc = subprocess.run(
    [
        str(rscript),
        str(r_path),
        str(model_input),
        str(OUT),
        str(RLIB),
    ],
    capture_output=True,
    text=True,
)

if proc.returncode != 0:
    print(proc.stdout[-4000:])
    print(proc.stderr[-7000:])
    raise SystemExit(proc.returncode)

r2 = pd.read_csv(OUT / "60_nonanaemic_model_R2.csv")

contrast_rows = []

for (period, outcome), q in r2.groupby(["period", "outcome"]):
    vals = dict(zip(q["model"], q["weighted_R2"]))

    base = vals.get("base_Gpcs_X", np.nan)
    hb = vals.get("continuous_Hb_margin", np.nan)
    apc = vals.get("frozen_A_PCs", np.nan)
    raw = vals.get("raw_Hb_RBC_MCV_RDW", np.nan)
    hb_apc = vals.get("Hb_margin_plus_A_PCs", np.nan)

    contrast_rows.extend([
        {
            "period": period,
            "outcome": outcome,
            "contrast": "continuous_Hb_margin_beyond_G_PCs_X",
            "delta_R2": hb - base,
        },
        {
            "period": period,
            "outcome": outcome,
            "contrast": "frozen_A_PCs_beyond_G_PCs_X",
            "delta_R2": apc - base,
        },
        {
            "period": period,
            "outcome": outcome,
            "contrast": "raw_Hb_RBC_MCV_RDW_beyond_G_PCs_X",
            "delta_R2": raw - base,
        },
        {
            "period": period,
            "outcome": outcome,
            "contrast": "RBC_MCV_RDW_beyond_continuous_Hb_margin",
            "delta_R2": raw - hb,
        },
        {
            "period": period,
            "outcome": outcome,
            "contrast": "A_PCs_beyond_continuous_Hb_margin_descriptive",
            "delta_R2": hb_apc - hb,
        },
    ])

contrasts = pd.DataFrame(contrast_rows)

contrasts.to_csv(
    OUT / "60_nonanaemic_incremental_R2.csv",
    index=False,
)

coef = pd.read_csv(OUT / "60_nonanaemic_coefficients.csv")


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
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


coef["p_BH_within_model"] = np.nan

for _, idx in coef.groupby(
    ["period", "outcome", "model"]
).groups.items():

    idx = list(idx)

    pc_idx = [
        i for i in idx
        if str(coef.loc[i, "term"]).startswith("A_OI_PC")
    ]

    if pc_idx:
        coef.loc[pc_idx, "p_BH_within_model"] = bh_adjust(
            coef.loc[pc_idx, "p"].to_numpy(float)
        )

coef.to_csv(
    OUT / "60_nonanaemic_coefficients.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    floor["period"],
    floor["nonanaemic_A_eq_0_weighted_percent"],
)
ax.set_xlabel("NHANES analysis period")
ax.set_ylabel(
    "Survey-weighted percentage with scalar anaemia burden A = 0 (%)"
)
ax.set_title(
    "Scalar anaemia burden has a large zero-valued region"
)
fig.tight_layout()
fig.savefig(
    FIG / "60_scalar_A_zero_floor_by_period.png",
    dpi=300,
)
plt.close(fig)

v = variation[
    variation["variable"].isin(A_PCS)
].copy()

x = np.arange(len(A_PCS))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

for i, period in enumerate(PERIODS):

    q = (
        v[v["period"].eq(period)]
        .set_index("variable")
        .reindex(A_PCS)
    )

    ax.bar(
        x + (i - 1) * width,
        q["weighted_sd"].to_numpy(),
        width=width,
        label=period,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [
        "Haematology PC1",
        "Haematology PC2",
        "Haematology PC3",
    ]
)

ax.set_xlabel(
    "Frozen haematological component among participants with A = 0"
)
ax.set_ylabel(
    "Survey-weighted standard deviation of component score"
)
ax.set_title(
    "Multivariate haematological variation remains after scalar A collapses to zero"
)
ax.legend(title="NHANES analysis period")

fig.tight_layout()
fig.savefig(
    FIG / "60_nonanaemic_A_PC_variation.png",
    dpi=300,
)
plt.close(fig)

plot_contrasts = [
    "continuous_Hb_margin_beyond_G_PCs_X",
    "frozen_A_PCs_beyond_G_PCs_X",
    "raw_Hb_RBC_MCV_RDW_beyond_G_PCs_X",
]

labels = {
    "continuous_Hb_margin_beyond_G_PCs_X":
        "Continuous Hb margin",
    "frozen_A_PCs_beyond_G_PCs_X":
        "Frozen haematology PCs",
    "raw_Hb_RBC_MCV_RDW_beyond_G_PCs_X":
        "Raw Hb + RBC + MCV + RDW",
}

z = contrasts[
    contrasts["outcome"].eq("SOMATIC_SCORE")
    & contrasts["contrast"].isin(plot_contrasts)
    & contrasts["period"].isin(["2005-2008", "2009-2018"])
].copy()

x = np.arange(len(plot_contrasts))
width = 0.34

fig, ax = plt.subplots(figsize=(10, 6))

for i, period in enumerate(["2005-2008", "2009-2018"]):

    q = (
        z[z["period"].eq(period)]
        .set_index("contrast")
        .reindex(plot_contrasts)
    )

    ax.bar(
        x + (i - 0.5) * width,
        q["delta_R2"].to_numpy(),
        width=width,
        label=period,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    [labels[c] for c in plot_contrasts],
    rotation=15,
    ha="right",
)

ax.set_xlabel(
    "Additional haematological representation among participants with A = 0"
)
ax.set_ylabel(
    "Increase in survey-weighted R² for somatic PHQ symptoms"
)
ax.set_title(
    "What explains phenotype information when threshold-defined anaemia is absent?"
)
ax.legend(title="NHANES analysis period")

fig.tight_layout()
fig.savefig(
    FIG / "60_nonanaemic_incremental_information.png",
    dpi=300,
)
plt.close(fig)

manifest = {
    "script": "60_nonanaemic_signal_audit.py",
    "question":
        "Do frozen haematological PCs retain phenotype-associated information "
        "among participants for whom scalar anaemia burden A is exactly zero?",
    "nonanaemic_definition":
        "A == 0 under locked sex-specific Hb threshold scalar",
    "A_representation":
        "Frozen outcome-independent A_OI_PC1-3_FZ from Script 47; no PCA refit",
    "raw_A_variables":
        ["Hb margin above sex-specific threshold", "RBC", "MCV", "RDW"],
    "primary_outcome":
        "SOMATIC_SCORE",
    "secondary_outcomes":
        ["PHQ9_TOTAL", "COGAFF_SUM"],
    "survey_rule":
        "Full period survey design first, then outcome analytic domain, then non-anaemic subpopulation",
    "causal_claim":
        "None. This is an associative representation-mechanism audit.",
    "locked_outputs_modified":
        False,
}

(OUT / "60_nonanaemic_signal_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("PASS  Non-anaemic signal audit complete.")
print("PASS  PCA was NOT refit.")
print("PASS  Survey design created before A=0 subgroup restriction.")
print("PASS  Tested whether A PCs still predict phenotype when scalar A is fixed at zero.")
print("PASS  Tested RBC/MCV/RDW beyond continuous Hb margin.")
print("PASS  Discovery + 2009-18 replication + modern descriptive holdout included.")
print("PASS  Outputs written only under EDA/Results and EDA/Figures.")
print("NOTE  This tests why the representation may work; it does not establish causality.")
