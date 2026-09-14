
args <- commandArgs(trailingOnly=TRUE)
g3_file <- args[1]
g4_file <- args[2]
results_dir <- args[3]
user_lib <- args[4]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("survey", quietly=TRUE)) {
  install.packages("survey", repos="https://cloud.r-project.org", lib=user_lib)
}
library(survey)
options(survey.lonely.psu="adjust")

g3 <- read.csv(g3_file, stringsAsFactors=FALSE)
g4 <- read.csv(g4_file, stringsAsFactors=FALSE)

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

g3 <- prep(g3)
g4 <- prep(g4)

make_design <- function(d, weight_col) {
  svydesign(
    ids=~PSU,
    strata=~STRATUM,
    weights=as.formula(paste0("~", weight_col)),
    nest=TRUE,
    data=d
  )
}

X3 <- c(
  "RIDAGEYR",
  "RIAGENDR",
  "RIDRETH1",
  "INDFMPIR",
  "EDUC3",
  "SMOKING3",
  "BMXBMI",
  "EGFR_2021"
)

make_formula <- function(outcome, terms) {
  txt <- paste(outcome, "~", paste(terms, collapse=" + "))
  message("FORMULA: ", txt)
  as.formula(txt)
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

extract_coef <- function(fit, block, outcome, model, keep_terms) {
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]
  out <- data.frame()

  for (term in keep_terms) {
    if (!(term %in% rownames(sm))) next
    out <- rbind(out, data.frame(
      block=block,
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

run_term_test <- function(fit, terms, block, outcome, test_name) {
  test_formula <- as.formula(
    paste("~", paste(terms, collapse=" + "))
  )

  z <- tryCatch(
    regTermTest(fit, test_formula, method="Wald"),
    error=function(e) {
      message("TEST FAILURE ", test_name, ": ", conditionMessage(e))
      NULL
    }
  )

  if (is.null(z)) {
    return(data.frame(
      block=block,
      outcome=outcome,
      test=test_name,
      F=NA_real_,
      df_num=NA_real_,
      df_den=NA_real_,
      p=NA_real_,
      design_df=degf(fit$survey.design)
    ))
  }

  data.frame(
    block=block,
    outcome=outcome,
    test=test_name,
    F=as.numeric(z$Ftest),
    df_num=as.numeric(z$df),
    df_den=as.numeric(z$ddf),
    p=as.numeric(z$p),
    design_df=degf(fit$survey.design)
  )
}

run_block <- function(d, block, pcs, weight_col) {
  des <- make_design(d, weight_col)

  coefficients <- data.frame()
  tests <- data.frame()
  r2s <- data.frame()

  for (outcome in c("SOMATIC_SCORE","PHQ9_TOTAL")) {

    base_terms <- c("A", X3, "CYCLE")
    scalar_terms <- c("A", "G_HBA1C", X3, "CYCLE")
    pc1_terms <- c("A", pcs[1], X3, "CYCLE")
    pc_terms <- c("A", pcs, X3, "CYCLE")
    union_terms <- c("A", "G_HBA1C", pcs, X3, "CYCLE")

    fit_base <- svyglm(
      make_formula(outcome, base_terms), design=des, family=gaussian()
    )
    fit_scalar <- svyglm(
      make_formula(outcome, scalar_terms), design=des, family=gaussian()
    )
    fit_pc1 <- svyglm(
      make_formula(outcome, pc1_terms), design=des, family=gaussian()
    )
    fit_pc <- svyglm(
      make_formula(outcome, pc_terms), design=des, family=gaussian()
    )
    fit_union <- svyglm(
      make_formula(outcome, union_terms), design=des, family=gaussian()
    )

    coefficients <- rbind(
      coefficients,
      extract_coef(fit_pc, block, outcome, "PC_block", pcs),
      extract_coef(
        fit_union, block, outcome, "union_scalar_plus_PCs",
        c("G_HBA1C", pcs)
      )
    )

    tests <- rbind(
      tests,
      run_term_test(
        fit_pc, pcs, block, outcome,
        "G_PCs_joint_given_A_X"
      ),
      run_term_test(
        fit_pc, pcs[-1], block, outcome,
        "G_PC2plus_extra_beyond_PC1_A_X"
      ),
      run_term_test(
        fit_union, pcs, block, outcome,
        "G_PCs_extra_beyond_scalar_G_A_X"
      ),
      run_term_test(
        fit_union, c("G_HBA1C"), block, outcome,
        "scalar_G_extra_beyond_G_PCs_A_X"
      )
    )

    fits <- list(
      X_plus_A=fit_base,
      scalar_G=fit_scalar,
      PC1_only=fit_pc1,
      PC_block=fit_pc,
      union_scalar_plus_PCs=fit_union
    )

    for (nm in names(fits)) {
      r2s <- rbind(r2s, data.frame(
        block=block,
        outcome=outcome,
        model=nm,
        weighted_R2=weighted_r2(fits[[nm]], d, weight_col, outcome),
        n=nrow(d),
        design_df=degf(des)
      ))
    }
  }

  list(coef=coefficients, tests=tests, r2=r2s)
}

g3_pcs <- c(
  "G3_FROZEN_PC1_Z",
  "G3_FROZEN_PC2_Z",
  "G3_FROZEN_PC3_Z"
)

g4_pcs <- c(
  "G4_FROZEN_PC1_Z",
  "G4_FROZEN_PC2_Z",
  "G4_FROZEN_PC3_Z",
  "G4_FROZEN_PC4_Z"
)

r3 <- run_block(g3, "G3_core", g3_pcs, "WTFAST4YR_PHENO")
r4 <- run_block(g4, "G4_extended", g4_pcs, "WTOGTT4YR_PHENO")

write.csv(
  rbind(r3$coef, r4$coef),
  file.path(results_dir, "43_G_component_coefficients.csv"),
  row.names=FALSE
)

write.csv(
  rbind(r3$tests, r4$tests),
  file.path(results_dir, "43_G_component_block_tests.csv"),
  row.names=FALSE
)

write.csv(
  rbind(r3$r2, r4$r2),
  file.path(results_dir, "43_G_model_weighted_R2.csv"),
  row.names=FALSE
)
