
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
