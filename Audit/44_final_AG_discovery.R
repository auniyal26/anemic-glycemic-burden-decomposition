
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
