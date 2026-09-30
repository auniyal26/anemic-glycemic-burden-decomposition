
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
