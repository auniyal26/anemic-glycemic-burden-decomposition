from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

# Resolve the project root from .../Scripts/38_*.py so folder spelling is irrelevant.
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

RESULTS.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RLIB.mkdir(parents=True, exist_ok=True)

INPUT = RESULTS / "37_component_primary_input.csv"

A_PCS = ["A_PC1_Z", "A_PC2_Z", "A_PC3_Z", "A_PC4_Z"]
G_TERM = "G_SCALAR_Z"

ADJUSTMENTS = {
    "X1_primary": [
        "RIDAGEYR", "RIAGENDR", "RIDRETH1",
        "INDFMPIR", "EDUC3", "SMOKING3"
    ],
    "X3_full": [
        "RIDAGEYR", "RIAGENDR", "RIDRETH1",
        "INDFMPIR", "EDUC3", "SMOKING3",
        "BMXBMI", "EGFR_2021"
    ],
}

print()
print("CROSS-CYCLE COMPONENT REPLICATION — FINAL")
print("=================================")

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing {INPUT}\nRun Scripts/37_component_level_phenotype_models.py first."
    )

df = pd.read_csv(INPUT, dtype={"CYCLE": str, "PSU": str, "STRATUM": str})
df["CYCLE"] = (
    df["CYCLE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .replace({"506": "0506", "708": "0708"})
)

needed = [
    "SEQN", "CYCLE", "PSU", "STRATUM", "WTMEC4YR",
    "SOMATIC_SCORE", "PHQ9_TOTAL", "A_SCALAR_Z", G_TERM
] + A_PCS + sorted({x for xs in ADJUSTMENTS.values() for x in xs})

missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=needed).copy()
df = df[df["WTMEC4YR"] > 0].copy()

counts = df["CYCLE"].value_counts().sort_index().to_dict()
if set(counts) != {"0506", "0708"}:
    raise ValueError(f"Expected cycles 0506 and 0708, found: {counts}")

input_csv = RESULTS / "38_cross_cycle_component_input.csv"
df.to_csv(input_csv, index=False)

rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])
if rscript is None:
    raise RuntimeError("Rscript not found.")

r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("survey", quietly=TRUE)) {
    install.packages("survey", repos="https://cloud.r-project.org", lib=user_lib)
}
if (!requireNamespace("MASS", quietly=TRUE)) {
    install.packages("MASS", repos="https://cloud.r-project.org", lib=user_lib)
}
library(survey)
library(MASS)
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)

# read.csv may coerce "0506"/"0708" to numeric 506/708.
# Normalize explicitly before any cycle-specific subsetting.
raw_cycle <- trimws(as.character(d$CYCLE))
raw_cycle <- sub("\\.0$", "", raw_cycle)
d$CYCLE <- ifelse(
    raw_cycle %in% c("506", "0506"),
    "0506",
    ifelse(raw_cycle %in% c("708", "0708"), "0708", raw_cycle)
)

cycle_counts <- table(d$CYCLE)
if (!all(c("0506", "0708") %in% names(cycle_counts))) {
    stop(
        paste(
            "Cycle normalization failed. Found:",
            paste(names(cycle_counts), cycle_counts, collapse=", ")
        )
    )
}

A_terms <- c("A_PC1_Z", "A_PC2_Z", "A_PC3_Z", "A_PC4_Z")
G_term <- "G_SCALAR_Z"

adjustments <- list(
    X1_primary = c(
        "RIDAGEYR", "RIAGENDR", "RIDRETH1",
        "INDFMPIR", "EDUC3", "SMOKING3"
    ),
    X3_full = c(
        "RIDAGEYR", "RIAGENDR", "RIDRETH1",
        "INDFMPIR", "EDUC3", "SMOKING3",
        "BMXBMI", "EGFR_2021"
    )
)

categorical_vars <- c("RIAGENDR", "RIDRETH1", "EDUC3", "SMOKING3")

build_rhs <- function(dat, adjustment_name, scope_label) {
    vars <- adjustments[[adjustment_name]]
    terms <- c()
    audit_rows <- list()

    for (v in vars) {
        vals <- dat[[v]][!is.na(dat[[v]])]
        n_unique <- length(unique(vals))
        keep <- TRUE
        reason <- ""

        if (v %in% categorical_vars) {
            if (n_unique >= 2) {
                terms <- c(terms, paste0("factor(", v, ")"))
            } else {
                keep <- FALSE
                reason <- "categorical_has_lt_2_levels"
            }
        } else {
            s <- if (length(vals) >= 2) sd(vals) else 0
            if (is.finite(s) && s > 0) {
                terms <- c(terms, v)
            } else {
                keep <- FALSE
                reason <- "numeric_has_zero_variance"
            }
        }

        audit_rows[[length(audit_rows)+1]] <- data.frame(
            scope=scope_label,
            adjustment=adjustment_name,
            variable=v,
            n_unique=n_unique,
            kept=keep,
            reason=reason,
            stringsAsFactors=FALSE
        )
    }

    if (length(terms) == 0) {
        rhs <- "1"
    } else {
        rhs <- paste(terms, collapse=" + ")
    }

    list(rhs=rhs, audit=do.call(rbind, audit_rows))
}

weighted_r2 <- function(fit, dat, outcome, weight_col="WTMEC4YR") {
    pred <- as.numeric(predict(fit, newdata=dat, type="response"))
    y <- dat[[outcome]]
    w <- dat[[weight_col]]
    ybar <- sum(w * y) / sum(w)
    sse <- sum(w * (y - pred)^2)
    sst <- sum(w * (y - ybar)^2)
    1 - sse / sst
}

wald_block <- function(fit, terms, design_df) {
    b <- coef(fit)
    V <- vcov(fit)
    keep <- terms[terms %in% names(b)]

    if (length(keep) == 0) {
        return(c(F=NA, df_num=NA, df_den=design_df, p=NA))
    }

    bb <- b[keep]
    VV <- V[keep, keep, drop=FALSE]

    invV <- tryCatch(
        solve(VV),
        error=function(e) MASS::ginv(VV)
    )

    q <- length(keep)
    Fval <- as.numeric(t(bb) %*% invV %*% bb / q)
    pval <- pf(Fval, df1=q, df2=design_df, lower.tail=FALSE)

    c(F=Fval, df_num=q, df_den=design_df, p=pval)
}

extract_terms <- function(fit, terms, cycle, adjustment, outcome, design_df) {
    b <- coef(fit)
    V <- vcov(fit)
    se <- sqrt(diag(V))
    crit <- qt(0.975, df=design_df)

    rows <- list()
    for (term in terms) {
        if (!(term %in% names(b))) next
        tval <- b[term] / se[term]
        p <- 2 * pt(abs(tval), df=design_df, lower.tail=FALSE)

        rows[[length(rows)+1]] <- data.frame(
            cycle=cycle,
            adjustment=adjustment,
            outcome=outcome,
            term=term,
            beta=unname(b[term]),
            se=unname(se[term]),
            ci_low=unname(b[term] - crit * se[term]),
            ci_high=unname(b[term] + crit * se[term]),
            p_design_t=unname(p),
            design_df=design_df,
            stringsAsFactors=FALSE
        )
    }
    if (length(rows) == 0) return(NULL)
    do.call(rbind, rows)
}

coef_rows <- list()
block_rows <- list()
r2_rows <- list()
heterogeneity_rows <- list()
covariate_audit_rows <- list()

for (adj_name in names(adjustments)) {

    for (cy in c("0506", "0708")) {
        dc <- d[d$CYCLE == cy, , drop=FALSE]

        if (nrow(dc) == 0) {
            stop(paste("No rows found for cycle", cy))
        }

        rhs_info <- build_rhs(dc, adj_name, paste0("cycle_", cy))
        X <- rhs_info$rhs
        covariate_audit_rows[[length(covariate_audit_rows)+1]] <- rhs_info$audit

        des <- svydesign(
            ids=~PSU,
            strata=~STRATUM,
            weights=~WTMEC4YR,
            nest=TRUE,
            data=dc
        )

        ddf <- degf(des)

        for (outcome in c("SOMATIC_SCORE", "PHQ9_TOTAL")) {
            f0 <- as.formula(paste(outcome, "~", X))
            f_scalar <- as.formula(
                paste(outcome, "~ A_SCALAR_Z +", G_term, "+", X)
            )
            f_pc1 <- as.formula(
                paste(outcome, "~ A_PC1_Z +", G_term, "+", X)
            )
            f_apc <- as.formula(
                paste(
                    outcome,
                    "~ A_PC1_Z + A_PC2_Z + A_PC3_Z + A_PC4_Z +",
                    G_term, "+", X
                )
            )
            f_union <- as.formula(
                paste(
                    outcome,
                    "~ A_SCALAR_Z + A_PC1_Z + A_PC2_Z + A_PC3_Z + A_PC4_Z +",
                    G_term, "+", X
                )
            )

            m0 <- svyglm(f0, design=des, family=gaussian())
            m_scalar <- svyglm(f_scalar, design=des, family=gaussian())
            m_pc1 <- svyglm(f_pc1, design=des, family=gaussian())
            m_apc <- svyglm(f_apc, design=des, family=gaussian())
            m_union <- svyglm(f_union, design=des, family=gaussian())

            coef_rows[[length(coef_rows)+1]] <- extract_terms(
                m_apc,
                c(A_terms, G_term),
                cy, adj_name, outcome, ddf
            )

            r0 <- weighted_r2(m0, dc, outcome)
            r_scalar <- weighted_r2(m_scalar, dc, outcome)
            r_pc1 <- weighted_r2(m_pc1, dc, outcome)
            r_apc <- weighted_r2(m_apc, dc, outcome)
            r_union <- weighted_r2(m_union, dc, outcome)

            r2_rows[[length(r2_rows)+1]] <- data.frame(
                cycle=cy,
                adjustment=adj_name,
                outcome=outcome,
                n=nrow(dc),
                design_df=ddf,
                R2_X=r0,
                R2_scalar_A_G=r_scalar,
                R2_PC1_G=r_pc1,
                R2_A_PC1_4_G=r_apc,
                R2_union_scalarA_APCs_G=r_union,
                delta_PC2_4_beyond_PC1=r_apc-r_pc1,
                delta_APCs_beyond_scalarA=r_union-r_scalar,
                delta_scalarA_beyond_APCs=r_union-r_apc,
                stringsAsFactors=FALSE
            )

            tests <- list(
                A_PC1_4_joint_given_G_X = list(
                    fit=m_apc, terms=A_terms
                ),
                A_PC2_4_extra_beyond_PC1_G_X = list(
                    fit=m_apc, terms=c("A_PC2_Z","A_PC3_Z","A_PC4_Z")
                ),
                A_PCs_extra_beyond_scalar_A_G_X = list(
                    fit=m_union, terms=A_terms
                ),
                scalar_A_extra_beyond_A_PCs_G_X = list(
                    fit=m_union, terms=c("A_SCALAR_Z")
                )
            )

            for (test_name in names(tests)) {
                wt <- wald_block(
                    tests[[test_name]]$fit,
                    tests[[test_name]]$terms,
                    ddf
                )
                block_rows[[length(block_rows)+1]] <- data.frame(
                    cycle=cy,
                    adjustment=adj_name,
                    outcome=outcome,
                    test=test_name,
                    F=unname(wt["F"]),
                    df_num=unname(wt["df_num"]),
                    df_den=unname(wt["df_den"]),
                    p=unname(wt["p"]),
                    stringsAsFactors=FALSE
                )
            }
        }
    }

    dp <- d
    dp$CYCLE_0708 <- as.numeric(dp$CYCLE == "0708")

    pooled_rhs_info <- build_rhs(dp, adj_name, "pooled_0506_0708")
    X <- pooled_rhs_info$rhs
    covariate_audit_rows[[length(covariate_audit_rows)+1]] <- pooled_rhs_info$audit

    for (pc in A_terms) {
        dp[[paste0(pc, "_x_CYCLE")]] <- dp[[pc]] * dp$CYCLE_0708
    }
    dp[[paste0(G_term, "_x_CYCLE")]] <- dp[[G_term]] * dp$CYCLE_0708

    desp <- svydesign(
        ids=~PSU,
        strata=~STRATUM,
        weights=~WTMEC4YR,
        nest=TRUE,
        data=dp
    )
    ddfp <- degf(desp)

    interaction_terms <- c(
        paste0(A_terms, "_x_CYCLE"),
        paste0(G_term, "_x_CYCLE")
    )

    for (outcome in c("SOMATIC_SCORE", "PHQ9_TOTAL")) {
        form <- as.formula(
            paste(
                outcome,
                "~",
                paste(c(A_terms, G_term, "CYCLE_0708", interaction_terms), collapse=" + "),
                "+",
                X
            )
        )

        mh <- svyglm(form, design=desp, family=gaussian())

        het_tests <- list(
            A_PC1_4_joint_cycle_heterogeneity = paste0(A_terms, "_x_CYCLE"),
            A_PC2_4_joint_cycle_heterogeneity = paste0(
                c("A_PC2_Z","A_PC3_Z","A_PC4_Z"), "_x_CYCLE"
            ),
            G_scalar_cycle_heterogeneity = paste0(G_term, "_x_CYCLE")
        )

        for (pc in A_terms) {
            het_tests[[paste0(pc, "_cycle_heterogeneity")]] <- paste0(pc, "_x_CYCLE")
        }

        for (test_name in names(het_tests)) {
            wt <- wald_block(mh, het_tests[[test_name]], ddfp)
            heterogeneity_rows[[length(heterogeneity_rows)+1]] <- data.frame(
                adjustment=adj_name,
                outcome=outcome,
                test=test_name,
                F=unname(wt["F"]),
                df_num=unname(wt["df_num"]),
                df_den=unname(wt["df_den"]),
                p=unname(wt["p"]),
                pooled_design_df=ddfp,
                stringsAsFactors=FALSE
            )
        }
    }
}

coefs <- do.call(rbind, coef_rows)
blocks <- do.call(rbind, block_rows)
r2 <- do.call(rbind, r2_rows)
heterogeneity <- do.call(rbind, heterogeneity_rows)
covariate_audit <- do.call(rbind, covariate_audit_rows)

write.csv(
    covariate_audit,
    file.path(results_dir, "38_covariate_variation_audit.csv"),
    row.names=FALSE
)

write.csv(
    coefs,
    file.path(results_dir, "38_cross_cycle_component_coefficients.csv"),
    row.names=FALSE
)
write.csv(
    blocks,
    file.path(results_dir, "38_cross_cycle_block_tests.csv"),
    row.names=FALSE
)
write.csv(
    r2,
    file.path(results_dir, "38_cross_cycle_information_replication.csv"),
    row.names=FALSE
)
write.csv(
    heterogeneity,
    file.path(results_dir, "38_component_cycle_heterogeneity.csv"),
    row.names=FALSE
)
'''

r_path = AUDIT / "38_cross_cycle_component_replication.R"
r_path.write_text(r_code, encoding="utf-8")

proc = subprocess.run(
    [str(rscript), str(r_path), str(input_csv), str(RESULTS), str(RLIB)],
    capture_output=True,
    text=True
)

if proc.returncode != 0:
    print("FAIL  Cross-cycle component replication did not complete.")
    print(proc.stdout[-2500:])
    print(proc.stderr[-3500:])
    raise SystemExit(proc.returncode)

coefs = pd.read_csv(RESULTS / "38_cross_cycle_component_coefficients.csv")
blocks = pd.read_csv(RESULTS / "38_cross_cycle_block_tests.csv")
r2 = pd.read_csv(RESULTS / "38_cross_cycle_information_replication.csv")
het = pd.read_csv(RESULTS / "38_component_cycle_heterogeneity.csv")

# pandas may again infer 0506/0708 as integers when reading R's CSV outputs.
# Normalize cycle labels before constructing replication verdicts.
def normalize_cycle_series(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .replace({"506": "0506", "708": "0708"})
    )

for _frame in (coefs, blocks, r2):
    if "cycle" in _frame.columns:
        _frame["cycle"] = normalize_cycle_series(_frame["cycle"])

coefs["p_BH_A_within_cycle"] = np.nan
for (adj, out, cy), idx in coefs.groupby(
    ["adjustment", "outcome", "cycle"]
).groups.items():
    idx = list(idx)
    a_idx = [i for i in idx if str(coefs.loc[i, "term"]).startswith("A_PC")]
    if not a_idx:
        continue
    p = coefs.loc[a_idx, "p_design_t"].to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    adjp = ranked * m / np.arange(1, m + 1)
    adjp = np.minimum.accumulate(adjp[::-1])[::-1]
    adjp = np.clip(adjp, 0, 1)
    restored = np.empty_like(adjp)
    restored[order] = adjp
    coefs.loc[a_idx, "p_BH_A_within_cycle"] = restored

coefs.to_csv(
    RESULTS / "38_cross_cycle_component_coefficients.csv",
    index=False
)

verdict_rows = []

for adjustment in ADJUSTMENTS:
    for outcome in ["SOMATIC_SCORE", "PHQ9_TOTAL"]:
        for term in A_PCS:
            dterm = coefs[
                (coefs["adjustment"] == adjustment)
                & (coefs["outcome"] == outcome)
                & (coefs["term"] == term)
            ].set_index("cycle")

            if not {"0506", "0708"}.issubset(dterm.index):
                continue

            b05 = float(dterm.loc["0506", "beta"])
            b07 = float(dterm.loc["0708", "beta"])
            same_direction = np.sign(b05) == np.sign(b07)

            hname = f"{term}_cycle_heterogeneity"
            h = het[
                (het["adjustment"] == adjustment)
                & (het["outcome"] == outcome)
                & (het["test"] == hname)
            ]
            hp = float(h["p"].iloc[0]) if len(h) else np.nan

            if not same_direction:
                verdict = "DIRECTION_FLIP"
            elif np.isfinite(hp) and hp < 0.05:
                verdict = "SAME_DIRECTION_BUT_HETEROGENEOUS"
            else:
                verdict = "DIRECTIONALLY_REPLICATED_NO_CLEAR_HETEROGENEITY"

            verdict_rows.append({
                "adjustment": adjustment,
                "outcome": outcome,
                "term": term,
                "beta_0506": b05,
                "beta_0708": b07,
                "same_direction": bool(same_direction),
                "heterogeneity_p": hp,
                "verdict": verdict
            })

if not verdict_rows:
    raise RuntimeError(
        "No replication verdict rows were created after cycle normalization. "
        f"Observed coefficient cycles: {sorted(coefs['cycle'].astype(str).unique())}"
    )

verdicts = pd.DataFrame(verdict_rows)
verdicts.to_csv(
    RESULTS / "38_component_replication_verdicts.csv",
    index=False
)

summary = {
    "analysis_sample_n": int(len(df)),
    "cycle_n": {str(k): int(v) for k, v in counts.items()},
    "primary_outcome": "SOMATIC_SCORE",
    "secondary_outcome": "PHQ9_TOTAL",
    "component_terms": A_PCS,
    "glycemic_term": G_TERM,
    "adjustments": ADJUSTMENTS,
    "formal_replication_rule": (
        "Do not require p<0.05 in each independent cycle. "
        "Use aligned effect direction plus pooled cycle-by-component "
        "heterogeneity tests; cycle-specific design-based estimates are reported."
    ),
    "interactions_tested": (
        "Only component-by-cycle interactions for heterogeneity. "
        "No A_i x G_j physiological interactions."
    )
}
with open(
    AUDIT / "38_cross_cycle_component_replication.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print(f"PASS  Replication input: n={len(df):,}")
print(
    "PASS  Cycle samples: "
    + ", ".join(f"{k} n={v:,}" for k, v in sorted(counts.items()))
)
print("PASS  Independent-cycle component coefficients and block tests saved.")
print("PASS  Scalar-vs-component information gain replicated separately by cycle.")
print("PASS  Pooled cycle-by-component heterogeneity tests saved.")
print("PASS  Covariate variation audit saved; one-level categorical covariates are dropped only within the affected cycle.")
print("NOTE  No physiological A_i x G_j interactions were tested.")
print()

print("SOMATIC — COMPONENT REPLICATION VERDICTS (X1 PRIMARY)")
print(
    verdicts[
        (verdicts["adjustment"] == "X1_primary")
        & (verdicts["outcome"] == "SOMATIC_SCORE")
    ].to_string(index=False)
)
print()

print("SOMATIC — BLOCK TESTS BY CYCLE (X1 PRIMARY)")
print(
    blocks[
        (blocks["adjustment"] == "X1_primary")
        & (blocks["outcome"] == "SOMATIC_SCORE")
    ].to_string(index=False)
)
print()

print("SOMATIC — INFORMATION GAIN BY CYCLE (X1 PRIMARY)")
print(
    r2[
        (r2["adjustment"] == "X1_primary")
        & (r2["outcome"] == "SOMATIC_SCORE")
    ][[
        "cycle", "n", "design_df",
        "delta_PC2_4_beyond_PC1",
        "delta_APCs_beyond_scalarA",
        "delta_scalarA_beyond_APCs"
    ]].to_string(index=False)
)
print()

print("SOMATIC — JOINT HETEROGENEITY TESTS")
print(
    het[
        (het["adjustment"] == "X1_primary")
        & (het["outcome"] == "SOMATIC_SCORE")
        & (het["test"].isin([
            "A_PC1_4_joint_cycle_heterogeneity",
            "A_PC2_4_joint_cycle_heterogeneity",
            "G_scalar_cycle_heterogeneity"
        ]))
    ].to_string(index=False)
)
