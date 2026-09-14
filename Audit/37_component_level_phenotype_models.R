
args <- commandArgs(trailingOnly=TRUE)
primary_file <- args[1]
fasting_file <- args[2]
results_dir <- args[3]
user_lib <- args[4]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))
if (!requireNamespace("survey", quietly=TRUE)) {
    install.packages("survey", repos="https://cloud.r-project.org", lib=user_lib)
}
library(survey)
options(survey.lonely.psu="adjust")

primary <- read.csv(primary_file, stringsAsFactors=FALSE)
fasting <- read.csv(fasting_file, stringsAsFactors=FALSE)
primary$CYCLE <- as.character(primary$CYCLE)
fasting$CYCLE <- as.character(fasting$CYCLE)

X_rhs <- paste(
    "RIDAGEYR",
    "factor(RIAGENDR)",
    "factor(RIDRETH1)",
    "INDFMPIR",
    "factor(EDUC3)",
    "factor(SMOKING3)",
    "BMXBMI",
    "EGFR_2021",
    "factor(CYCLE)",
    sep=" + "
)

weighted_r2 <- function(fit, dat, outcome, weight_col) {
    pred <- as.numeric(predict(fit, newdata=dat, type="response"))
    y <- dat[[outcome]]
    w <- dat[[weight_col]]
    ybar <- sum(w * y) / sum(w)
    sse <- sum(w * (y - pred)^2)
    sst <- sum(w * (y - ybar)^2)
    1 - sse / sst
}

safe_term_test <- function(fit, terms, label, analysis_name, outcome, design_df) {
    form <- as.formula(paste("~", paste(terms, collapse=" + ")))
    tt <- tryCatch(
        regTermTest(fit, form, method="Wald"),
        error=function(e) NULL
    )
    if (is.null(tt)) {
        return(data.frame(
            analysis=analysis_name, outcome=outcome, test=label,
            F=NA, df_num=NA, df_den=NA, p=NA, design_df=design_df,
            stringsAsFactors=FALSE
        ))
    }
    data.frame(
        analysis=analysis_name,
        outcome=outcome,
        test=label,
        F=as.numeric(tt$Ftest),
        df_num=as.numeric(tt$df),
        df_den=as.numeric(tt$ddf),
        p=as.numeric(tt$p),
        design_df=design_df,
        stringsAsFactors=FALSE
    )
}

extract_terms <- function(fit, terms, analysis_name, outcome, model_name) {
    sm <- summary(fit)$coefficients
    ci <- confint(fit)
    rows <- list()
    for (term in terms) {
        if (!(term %in% rownames(sm))) next
        rows[[length(rows)+1]] <- data.frame(
            analysis=analysis_name,
            outcome=outcome,
            model=model_name,
            term=term,
            beta=sm[term, "Estimate"],
            se=sm[term, "Std. Error"],
            ci_low=ci[term, 1],
            ci_high=ci[term, 2],
            p=sm[term, ncol(sm)],
            stringsAsFactors=FALSE
        )
    }
    if (length(rows) == 0) return(NULL)
    do.call(rbind, rows)
}

model_row <- function(fit, dat, outcome, weight_col, analysis_name, model_name, baseline_r2) {
    r2 <- weighted_r2(fit, dat, outcome, weight_col)
    data.frame(
        analysis=analysis_name,
        outcome=outcome,
        model=model_name,
        weighted_R2=r2,
        delta_R2_over_X=r2-baseline_r2,
        n=nrow(dat),
        stringsAsFactors=FALSE
    )
}

coef_rows <- list()
block_rows <- list()
model_rows <- list()

# ---------------- PRIMARY: A PCs + scalar G ----------------
des_p <- svydesign(
    ids=~PSU, strata=~STRATUM, weights=~WTMEC4YR,
    nest=TRUE, data=primary
)

A_terms <- c("A_PC1_Z", "A_PC2_Z", "A_PC3_Z", "A_PC4_Z")

for (outcome in c("SOMATIC_SCORE", "PHQ9_TOTAL")) {
    f0 <- as.formula(paste(outcome, "~", X_rhs))
    f_scalar <- as.formula(paste(outcome, "~ A_SCALAR_Z + G_SCALAR_Z +", X_rhs))
    f_pc1 <- as.formula(paste(outcome, "~ A_PC1_Z + G_SCALAR_Z +", X_rhs))
    f_apc <- as.formula(paste(outcome, "~ A_PC1_Z + A_PC2_Z + A_PC3_Z + A_PC4_Z + G_SCALAR_Z +", X_rhs))
    f_union <- as.formula(paste(outcome, "~ A_SCALAR_Z + A_PC1_Z + A_PC2_Z + A_PC3_Z + A_PC4_Z + G_SCALAR_Z +", X_rhs))

    m0 <- svyglm(f0, design=des_p, family=gaussian())
    m_scalar <- svyglm(f_scalar, design=des_p, family=gaussian())
    m_pc1 <- svyglm(f_pc1, design=des_p, family=gaussian())
    m_apc <- svyglm(f_apc, design=des_p, family=gaussian())
    m_union <- svyglm(f_union, design=des_p, family=gaussian())

    r0 <- weighted_r2(m0, primary, outcome, "WTMEC4YR")
    model_rows[[length(model_rows)+1]] <- model_row(m0, primary, outcome, "WTMEC4YR", "Primary_MEC", "X", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_scalar, primary, outcome, "WTMEC4YR", "Primary_MEC", "X+scalar_A+scalar_G", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_pc1, primary, outcome, "WTMEC4YR", "Primary_MEC", "X+A_PC1+scalar_G", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_apc, primary, outcome, "WTMEC4YR", "Primary_MEC", "X+A_PC1-4+scalar_G", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_union, primary, outcome, "WTMEC4YR", "Primary_MEC", "X+scalar_A+A_PC1-4+scalar_G", r0)

    ddf <- degf(des_p)
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_apc, A_terms, "A_PC1-4_joint_given_G_X", "Primary_MEC", outcome, ddf
    )
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_apc, c("A_PC2_Z", "A_PC3_Z", "A_PC4_Z"),
        "A_PC2-4_extra_beyond_PC1_G_X", "Primary_MEC", outcome, ddf
    )
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_union, A_terms, "A_PCs_extra_beyond_scalar_A_G_X", "Primary_MEC", outcome, ddf
    )
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_union, c("A_SCALAR_Z"), "scalar_A_extra_beyond_A_PCs_G_X", "Primary_MEC", outcome, ddf
    )

    coef_rows[[length(coef_rows)+1]] <- extract_terms(
        m_apc, c(A_terms, "G_SCALAR_Z"), "Primary_MEC", outcome, "A_PC1-4+scalar_G+X"
    )
}

# ---------------- FASTING: A PCs + G PCs ----------------
des_f <- svydesign(
    ids=~PSU, strata=~STRATUM, weights=~WTFAST4YR,
    nest=TRUE, data=fasting
)

A_f_terms <- c("A_PC1_FZ", "A_PC2_FZ", "A_PC3_FZ", "A_PC4_FZ")
G_f_terms <- c("G_PC1_FZ", "G_PC2_FZ")

for (outcome in c("SOMATIC_SCORE", "PHQ9_TOTAL")) {
    f0 <- as.formula(paste(outcome, "~", X_rhs))
    f_scalar <- as.formula(paste(outcome, "~ A_SCALAR_FZ + G_SCALAR_FZ +", X_rhs))
    f_common <- as.formula(paste(outcome, "~ A_PC1_FZ + A_PC2_FZ + A_PC3_FZ + A_PC4_FZ + G_PC1_FZ +", X_rhs))
    f_full <- as.formula(paste(outcome, "~ A_PC1_FZ + A_PC2_FZ + A_PC3_FZ + A_PC4_FZ + G_PC1_FZ + G_PC2_FZ +", X_rhs))
    f_union <- as.formula(paste(outcome, "~ A_SCALAR_FZ + G_SCALAR_FZ + A_PC1_FZ + A_PC2_FZ + A_PC3_FZ + A_PC4_FZ + G_PC1_FZ + G_PC2_FZ +", X_rhs))

    m0 <- svyglm(f0, design=des_f, family=gaussian())
    m_scalar <- svyglm(f_scalar, design=des_f, family=gaussian())
    m_common <- svyglm(f_common, design=des_f, family=gaussian())
    m_full <- svyglm(f_full, design=des_f, family=gaussian())
    m_union <- svyglm(f_union, design=des_f, family=gaussian())

    r0 <- weighted_r2(m0, fasting, outcome, "WTFAST4YR")
    model_rows[[length(model_rows)+1]] <- model_row(m0, fasting, outcome, "WTFAST4YR", "Fasting_sensitivity", "X", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_scalar, fasting, outcome, "WTFAST4YR", "Fasting_sensitivity", "X+scalar_A+scalar_G", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_common, fasting, outcome, "WTFAST4YR", "Fasting_sensitivity", "X+A_PC1-4+G_PC1", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_full, fasting, outcome, "WTFAST4YR", "Fasting_sensitivity", "X+A_PC1-4+G_PC1-2", r0)
    model_rows[[length(model_rows)+1]] <- model_row(m_union, fasting, outcome, "WTFAST4YR", "Fasting_sensitivity", "X+scalars+A_PC1-4+G_PC1-2", r0)

    ddf <- degf(des_f)
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_full, A_f_terms, "A_PC1-4_joint_given_GPCs_X", "Fasting_sensitivity", outcome, ddf
    )
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_full, G_f_terms, "G_PC1-2_joint_given_APCs_X", "Fasting_sensitivity", outcome, ddf
    )
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_full, c("G_PC2_FZ"), "G_PC2_discordance_extra_beyond_G_PC1_A_X", "Fasting_sensitivity", outcome, ddf
    )
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_union, c(A_f_terms, G_f_terms), "all_PCs_extra_beyond_scalar_A_G_X", "Fasting_sensitivity", outcome, ddf
    )
    block_rows[[length(block_rows)+1]] <- safe_term_test(
        m_union, c("A_SCALAR_FZ", "G_SCALAR_FZ"), "scalars_extra_beyond_all_PCs_X", "Fasting_sensitivity", outcome, ddf
    )

    coef_rows[[length(coef_rows)+1]] <- extract_terms(
        m_full, c(A_f_terms, G_f_terms), "Fasting_sensitivity", outcome, "A_PC1-4+G_PC1-2+X"
    )
}

models <- do.call(rbind, model_rows)
blocks <- do.call(rbind, block_rows)
coefs <- do.call(rbind, coef_rows)

coefs$p_BH_within_block <- NA_real_
for (an in unique(coefs$analysis)) {
    for (out in unique(coefs$outcome)) {
        idxA <- which(coefs$analysis==an & coefs$outcome==out & grepl("^A_PC", coefs$term))
        idxG <- which(coefs$analysis==an & coefs$outcome==out & grepl("^G_PC", coefs$term))
        if (length(idxA) > 0) coefs$p_BH_within_block[idxA] <- p.adjust(coefs$p[idxA], method="BH")
        if (length(idxG) > 0) coefs$p_BH_within_block[idxG] <- p.adjust(coefs$p[idxG], method="BH")
    }
}

write.csv(models, file.path(results_dir, "37_component_model_comparison.csv"), row.names=FALSE)
write.csv(blocks, file.path(results_dir, "37_component_block_tests.csv"), row.names=FALSE)
write.csv(coefs, file.path(results_dir, "37_component_coefficients.csv"), row.names=FALSE)
