
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]

if (!requireNamespace("lavaan", quietly=TRUE)) {
    install.packages("lavaan", repos="https://cloud.r-project.org")
}

library(lavaan)

d <- read.csv(input_file)
items <- c("DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090")
d$CYCLE <- factor(d$CYCLE)

m1 <- "
Depression =~ DPQ010 + DPQ020 + DPQ030 + DPQ040 + DPQ050 + DPQ060 + DPQ070 + DPQ080 + DPQ090
"

m2 <- "
Somatic =~ DPQ030 + DPQ040 + DPQ050
CognitiveAffective =~ DPQ010 + DPQ020 + DPQ060 + DPQ070 + DPQ080 + DPQ090
"

fit_rows <- list()
loading_rows <- list()

for (cy in levels(d$CYCLE)) {
    dc <- d[d$CYCLE == cy, ]

    f1 <- cfa(
        m1,
        data=dc,
        ordered=items,
        estimator="WLSMV",
        std.lv=TRUE
    )

    f2 <- cfa(
        m2,
        data=dc,
        ordered=items,
        estimator="WLSMV",
        std.lv=TRUE
    )

    fits <- list(one_factor=f1, two_factor=f2)

    for (nm in names(fits)) {
        fit <- fits[[nm]]
        fm <- fitMeasures(fit, c("cfi","tli","rmsea","srmr"))

        fit_rows[[length(fit_rows)+1]] <- data.frame(
            cycle=cy,
            model=nm,
            cfi=unname(fm["cfi"]),
            tli=unname(fm["tli"]),
            rmsea=unname(fm["rmsea"]),
            srmr=unname(fm["srmr"])
        )
    }

    ss <- standardizedSolution(f2)
    ss <- ss[ss$op == "=~", c("lhs","rhs","est.std","se","pvalue")]
    ss$cycle <- cy
    loading_rows[[length(loading_rows)+1]] <- ss
}

fit_df <- do.call(rbind, fit_rows)
load_df <- do.call(rbind, loading_rows)

config <- cfa(
    m2,
    data=d,
    group="CYCLE",
    ordered=items,
    estimator="WLSMV",
    std.lv=TRUE
)

loading_equal <- cfa(
    m2,
    data=d,
    group="CYCLE",
    ordered=items,
    estimator="WLSMV",
    std.lv=TRUE,
    group.equal=c("loadings")
)

threshold_equal <- cfa(
    m2,
    data=d,
    group="CYCLE",
    ordered=items,
    estimator="WLSMV",
    std.lv=TRUE,
    group.equal=c("loadings","thresholds")
)

inv_rows <- list()
inv_fits <- list(
    configural=config,
    equal_loadings=loading_equal,
    equal_loadings_thresholds=threshold_equal
)

for (nm in names(inv_fits)) {
    fit <- inv_fits[[nm]]
    fm <- fitMeasures(fit, c("cfi","tli","rmsea","srmr"))

    inv_rows[[length(inv_rows)+1]] <- data.frame(
        model=nm,
        cfi=unname(fm["cfi"]),
        tli=unname(fm["tli"]),
        rmsea=unname(fm["rmsea"]),
        srmr=unname(fm["srmr"])
    )
}

inv_df <- do.call(rbind, inv_rows)

write.csv(fit_df, file.path(results_dir, "20_cfa_fit.csv"), row.names=FALSE)
write.csv(load_df, file.path(results_dir, "20_cfa_two_factor_loadings.csv"), row.names=FALSE)
write.csv(inv_df, file.path(results_dir, "20_cfa_invariance.csv"), row.names=FALSE)
