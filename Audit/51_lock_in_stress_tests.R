
args <- commandArgs(trailingOnly=TRUE)
input47 <- args[1]
input49 <- args[2]
adj_out <- args[3]
time_out <- args[4]

if (!requireNamespace("survey", quietly=TRUE)) {
  stop("R package 'survey' is required")
}
library(survey)
options(survey.lonely.psu="adjust")

safe_test <- function(fit, terms) {
  z <- tryCatch(
    regTermTest(fit, as.formula(paste("~", paste(terms, collapse=" + "))), method="Wald"),
    error=function(e) NULL
  )
  if (is.null(z)) {
    return(c(F=NA_real_, df_num=NA_real_, df_den=NA_real_, p=NA_real_))
  }
  c(F=as.numeric(z$Ftest), df_num=as.numeric(z$df),
    df_den=as.numeric(z$ddf), p=as.numeric(z$p))
}

make_formula <- function(outcome, terms) {
  as.formula(paste(outcome, "~", paste(unique(terms), collapse=" + ")))
}

wr2 <- function(fit, d, outcome) {
  y <- d[[outcome]]
  pred <- as.numeric(predict(fit, newdata=d, type="response"))
  w <- d$SURVEY_WT
  mu <- sum(w*y)/sum(w)
  sse <- sum(w*(y-pred)^2)
  sst <- sum(w*(y-mu)^2)
  if (sst <= 0) return(NA_real_)
  1 - sse/sst
}

# A) SAME-SAMPLE ADJUSTMENT LADDER
normalize_cycle <- function(x) {
  z <- gsub("\\.0$", "", as.character(x))
  digits <- gsub("[^0-9]", "", z)
  out <- suppressWarnings(sprintf("%04d", as.integer(digits)))
  out[is.na(out)] <- z[is.na(out)]
  out
}

d <- read.csv(input47, stringsAsFactors=FALSE)
d$CYCLE <- normalize_cycle(d$CYCLE)
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])

A <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")
PCS <- c(A,G)

sets <- list(
  X0=c("RIDAGEYR","RIAGENDR","RACE"),
  X1=c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3"),
  X2=c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI"),
  X3=c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
)

rows <- data.frame()

for (pop in c("2005-2008","2009-2018")) {
  dat <- d[d$PERIOD==pop & d$SURVEY_WT>0,]
  des_full <- svydesign(ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT,
                        nest=TRUE, data=dat)
  des <- subset(des_full, DOMAIN_SOMATIC_SCORE_X3==1)
  dom <- dat[dat$DOMAIN_SOMATIC_SCORE_X3==1,]

  for (nm in names(sets)) {
    covs <- c(sets[[nm]], "CYCLE")

    fit_scalar <- svyglm(
      make_formula("SOMATIC_SCORE", c("A","G_HBA1C",covs)),
      design=des, family=gaussian()
    )
    fit_pcs <- svyglm(
      make_formula("SOMATIC_SCORE", c(PCS,covs)),
      design=des, family=gaussian()
    )
    fit_union <- svyglm(
      make_formula("SOMATIC_SCORE", c("A","G_HBA1C",PCS,covs)),
      design=des, family=gaussian()
    )

    t_pc <- safe_test(fit_union, PCS)
    t_sc <- safe_test(fit_union, c("A","G_HBA1C"))

    r2s <- wr2(fit_scalar, dom, "SOMATIC_SCORE")
    r2p <- wr2(fit_pcs, dom, "SOMATIC_SCORE")
    r2u <- wr2(fit_union, dom, "SOMATIC_SCORE")

    rows <- rbind(rows, data.frame(
      population=pop, adjustment=nm, n=nrow(dom),
      design_df=degf(des),
      p_PCs_beyond_scalars=t_pc["p"],
      F_PCs_beyond_scalars=t_pc["F"],
      df_num_PCs=t_pc["df_num"], df_den_PCs=t_pc["df_den"],
      p_scalars_beyond_PCs=t_sc["p"],
      R2_scalar=r2s, R2_PCs=r2p, R2_union=r2u,
      delta_PCs_beyond_scalars=r2u-r2s,
      delta_scalars_beyond_PCs=r2u-r2p
    ))
  }
}

write.csv(rows, adj_out, row.names=FALSE)

# B) NONLINEAR / ARBITRARY CYCLE-SPECIFIC SLOPE HETEROGENEITY
t <- read.csv(input49, stringsAsFactors=FALSE)
t$CYCLE <- normalize_cycle(t$CYCLE)
t <- t[t$CYCLE %in% c("0506","0708","0910","1112","1314","1516","1718") &
       !is.na(t$WT_0518) & t$WT_0518>0,]
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) t[[v]] <- factor(t[[v]])

des_full_t <- svydesign(ids=~PSU, strata=~STRATUM, weights=~WT_0518,
                        nest=TRUE, data=t)
des_t <- subset(des_full_t, DOMAIN_SOMATIC_SCORE_X3==1)

X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")

fit_A <- svyglm(
  make_formula("SOMATIC_SCORE",
    c(PCS,X3,"CYCLE",
      "A_OI_PC1_FZ:CYCLE","A_OI_PC2_FZ:CYCLE","A_OI_PC3_FZ:CYCLE")),
  design=des_t, family=gaussian()
)

fit_G <- svyglm(
  make_formula("SOMATIC_SCORE",
    c(PCS,X3,"CYCLE",
      "G3_OI_PC1_FZ:CYCLE","G3_OI_PC2_FZ:CYCLE","G3_OI_PC3_FZ:CYCLE")),
  design=des_t, family=gaussian()
)

ta <- safe_test(fit_A, c("A_OI_PC1_FZ:CYCLE","A_OI_PC2_FZ:CYCLE","A_OI_PC3_FZ:CYCLE"))
tg <- safe_test(fit_G, c("G3_OI_PC1_FZ:CYCLE","G3_OI_PC2_FZ:CYCLE","G3_OI_PC3_FZ:CYCLE"))

out <- rbind(
  data.frame(block="A", F=ta["F"], df_num=ta["df_num"], df_den=ta["df_den"],
             p=ta["p"], design_df=degf(des_t)),
  data.frame(block="G3", F=tg["F"], df_num=tg["df_num"], df_den=tg["df_den"],
             p=tg["p"], design_df=degf(des_t))
)
write.csv(out, time_out, row.names=FALSE)
