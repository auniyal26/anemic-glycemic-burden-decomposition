
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

A_PCS <- c("A_FROZEN_PC1_FZ","A_FROZEN_PC2_FZ","A_FROZEN_PC3_FZ")
G_PCS <- c("G3_FROZEN_PC1_FZ","G3_FROZEN_PC2_FZ","G3_FROZEN_PC3_FZ")

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
