
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]
if (!requireNamespace("survey", quietly=TRUE)) {
  stop("R package 'survey' is not installed in the active environment. Install with: conda install -n physiodecomp -c conda-forge r-survey")
}
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)

# Preserve/restore canonical 4-character NHANES cycle labels.
# read.csv() may type-convert "0506" to numeric 506, which would make
# d$CYCLE %in% c("0506","0708") silently return no rows.
cy <- trimws(as.character(d$CYCLE))
cy_num <- suppressWarnings(as.integer(cy))
is_numeric_cycle <- grepl("^[0-9]+$", cy) & !is.na(cy_num)
cy[is_numeric_cycle] <- sprintf("%04d", cy_num[is_numeric_cycle])
d$CYCLE <- cy

for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])

cat("Canonical CYCLE levels read by R:", paste(levels(d$CYCLE), collapse=", "), "\\n")

A <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G3 <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")
G4 <- c("G4_PC1_FZ","G4_PC2_FZ","G4_PC3_FZ")
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
AXG <- c("A1_X_G4_1","A1_X_G4_2","A1_X_G4_3","A2_X_G4_1","A2_X_G4_2","A2_X_G4_3","A3_X_G4_1","A3_X_G4_2","A3_X_G4_3")
mods <- c("AGE10_C","FEMALE","BMI5_C","EGFR10_C")

make_formula <- function(outcome, terms) as.formula(paste(outcome, "~", paste(unique(terms), collapse=" + ")))

safe_test <- function(fit, terms, population, family, label) {
  z <- tryCatch(regTermTest(fit, as.formula(paste("~", paste(terms, collapse=" + "))), method="Wald"), error=function(e) NULL)
  if (is.null(z)) return(data.frame(population=population,family=family,test=label,F=NA,df_num=NA,df_den=NA,p=NA,design_df=degf(fit$survey.design),inference_status="unavailable"))
  ddf <- as.numeric(z$ddf)
  status <- ifelse(is.na(ddf),"unavailable",ifelse(ddf<=3,"design_limited","available"))
  data.frame(population=population,family=family,test=label,F=as.numeric(z$Ftest),df_num=as.numeric(z$df),df_den=ddf,p=as.numeric(z$p),design_df=degf(fit$survey.design),inference_status=status)
}

wr2 <- function(fit) {
  y <- fit$y; w <- weights(fit$survey.design, "sampling"); pred <- fitted(fit)
  ok <- is.finite(y) & is.finite(w) & is.finite(pred) & w>0
  y<-y[ok]; w<-w[ok]; pred<-pred[ok]
  ym <- sum(w*y)/sum(w)
  1 - sum(w*(y-pred)^2)/sum(w*(y-ym)^2)
}

run_pop <- function(dat, pop) {
  if (nrow(dat) == 0) stop(paste0(pop, ": zero rows reached run_pop(); check CYCLE normalization/filtering."))
  positive <- dat[!is.na(dat$OGTT_WT_PERIOD) & dat$OGTT_WT_PERIOD>0,]
  if (nrow(positive) == 0) stop(paste0(pop, ": no positive OGTT survey weights after cycle filtering."))
  des_full <- svydesign(ids=~PSU, strata=~STRATUM, weights=~OGTT_WT_PERIOD, nest=TRUE, data=positive)
  des <- subset(des_full, DOMAIN_G4_X3==1)
  cat(pop, " R-side rows: cycle subset=", nrow(dat),
      "; positive weight=", nrow(des_full$variables),
      "; analytic domain=", nrow(des$variables), "\n", sep="")
  if (nrow(des$variables) == 0) stop(paste0(pop, ": analytic survey domain unexpectedly empty in R."))

  # A factor with only one observed level in an analytic domain cannot be
  # included in model.matrix(). Drop only such non-identifiable adjustment
  # terms; this does not alter the analytic sample or the survey design.
  usable_term <- function(v, vars) {
    x <- vars[[v]]
    x <- x[!is.na(x)]
    if (length(x) == 0) return(FALSE)
    if (is.factor(x) || is.character(x)) return(length(unique(as.character(x))) >= 2)
    return(length(unique(x)) >= 2)
  }

  requested_covars <- c(X3,"CYCLE")
  covars <- requested_covars[sapply(requested_covars, usable_term, vars=des$variables)]
  dropped_covars <- setdiff(requested_covars, covars)

  if (length(dropped_covars) > 0) {
    cat("\n", pop, "dropping non-identifiable covariate(s):",
        paste(dropped_covars, collapse=", "), "\n")
  }

  factor_diag <- sapply(c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE"), function(v) {
    length(unique(as.character(des$variables[[v]][!is.na(des$variables[[v]])])))
  })
  cat(pop, " observed factor levels:",
      paste(names(factor_diag), factor_diag, sep="=", collapse=", "), "\n")

  fit_g3 <- svyglm(make_formula("SOMATIC_SCORE", c(A,G3,covars)), design=des, family=gaussian())
  fit_g4 <- svyglm(make_formula("SOMATIC_SCORE", c(A,G4,covars)), design=des, family=gaussian())
  fit_g3ogtt <- svyglm(make_formula("SOMATIC_SCORE", c(A,G3,"OGTT2H_DISC_Z",covars)), design=des, family=gaussian())
  fit_axg <- svyglm(make_formula("SOMATIC_SCORE", c(A,G4,AXG,covars)), design=des, family=gaussian())

  tests <- rbind(
    safe_test(fit_g3ogtt, "OGTT2H_DISC_Z", pop, "G4_extension", "2h_OGTT_extra_beyond_full_G3"),
    safe_test(fit_g4, G4, pop, "G4_extension", "G4_components_joint_given_A_X"),
    safe_test(fit_g4, A, pop, "G4_extension", "A_components_joint_given_G4_X"),
    safe_test(fit_axg, "A1_X_G4_1", pop, "A_x_G4", "PC1_x_PC1"),
    safe_test(fit_axg, AXG, pop, "A_x_G4", "all_9_AxG4_interactions")
  )

  for (m in mods) {
    Aint <- paste0(c("A1_X_","A2_X_","A3_X_"),m)
    Gint <- paste0(c("G4_1_X_","G4_2_X_","G4_3_X_"),m)
    # Do not add m separately: its parent main effect is already present in X3
    # (AGE10_C~RIDAGEYR, FEMALE~RIAGENDR, BMI5_C~BMXBMI, EGFR10_C~EGFR_2021).
    # This preserves hierarchy without exact/affine collinearity.
    f <- svyglm(make_formula("SOMATIC_SCORE", c(A,G4,Aint,Gint,covars)), design=des, family=gaussian())
    tests <- rbind(tests,
      safe_test(f,Aint,pop,"effect_modification",paste0("A_components_x_",m)),
      safe_test(f,Gint,pop,"effect_modification",paste0("G4_components_x_",m)),
      safe_test(f,c(Aint,Gint),pop,"effect_modification",paste0("all_components_x_",m))
    )
  }

  r2 <- data.frame(population=pop, model=c("A_plus_G3","A_plus_G4","A_plus_G3_plus_2hOGTT"),
                   weighted_R2=c(wr2(fit_g3),wr2(fit_g4),wr2(fit_g3ogtt)), n=nrow(des$variables), design_df=degf(des))
  flow <- data.frame(
    population=pop,
    full_positive_OGTT_weight_n=nrow(des_full$variables),
    analytic_domain_n=nrow(des$variables),
    design_df=degf(des),
    dropped_nonidentifiable_covariates=ifelse(length(dropped_covars)==0,"",paste(dropped_covars,collapse=";"))
  )
  list(tests=tests,r2=r2,flow=flow)
}

r1 <- run_pop(d[d$CYCLE %in% c("0506","0708"),], "2005-2008")
r2 <- run_pop(d[d$CYCLE %in% c("0910","1112","1314","1516"),], "2009-2016")

tests <- rbind(r1$tests,r2$tests)
# FDR only within the modifier family, preserving raw p-values.
idx <- which(tests$family=="effect_modification" & is.finite(tests$p))
tests$q_BH_modifier_family <- NA_real_
if (length(idx)>0) tests$q_BH_modifier_family[idx] <- p.adjust(tests$p[idx], method="BH")

write.csv(tests,file.path(results_dir,"50_g4_modifier_tests.csv"),row.names=FALSE)
write.csv(rbind(r1$r2,r2$r2),file.path(results_dir,"50_g4_same_sample_R2.csv"),row.names=FALSE)
write.csv(rbind(r1$flow,r2$flow),file.path(results_dir,"50_g4_sample_flow.csv"),row.names=FALSE)

cat("\nSAME-SAMPLE G3 vs G4\n")
print(rbind(r1$r2,r2$r2),row.names=FALSE)
cat("\nG4 / A x G4 / X-MODIFIER TESTS\n")
print(tests,row.names=FALSE)
