
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
