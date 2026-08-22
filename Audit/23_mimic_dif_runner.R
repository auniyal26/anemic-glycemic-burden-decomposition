
args <- commandArgs(
    trailingOnly=TRUE
)

input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(
    user_lib,
    recursive=TRUE,
    showWarnings=FALSE
)

.libPaths(
    c(
        user_lib,
        .libPaths()
    )
)

if (
    !requireNamespace(
        "lavaan",
        quietly=TRUE
    )
) {
    install.packages(
        "lavaan",
        repos="https://cloud.r-project.org",
        lib=user_lib
    )
}

library(lavaan)

d <- read.csv(
    input_file
)

d$CYCLE <- factor(
    d$CYCLE
)

items <- c(
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090"
)

exposures <- c(
    "A_Z",
    "G_HBA1C_Z",
    "AG_Z"
)

base_model <- "
Somatic =~ DPQ030 + DPQ040 + DPQ050

CognitiveAffective =~
DPQ010 +
DPQ020 +
DPQ060 +
DPQ070 +
DPQ080 +
DPQ090

Somatic ~
A_Z +
G_HBA1C_Z +
AG_Z +
RIDAGEYR_Z + INDFMPIR_Z + BMXBMI_Z + EGFR_2021_Z + RIAGENDR_2 + RIDRETH1_2 + RIDRETH1_3 + RIDRETH1_4 + RIDRETH1_5 + EDUC3_2 + EDUC3_3 + SMOKING3_1 + SMOKING3_2

CognitiveAffective ~
A_Z +
G_HBA1C_Z +
AG_Z +
RIDAGEYR_Z + INDFMPIR_Z + BMXBMI_Z + EGFR_2021_Z + RIAGENDR_2 + RIDRETH1_2 + RIDRETH1_3 + RIDRETH1_4 + RIDRETH1_5 + EDUC3_2 + EDUC3_3 + SMOKING3_1 + SMOKING3_2
"

message(
    "BASE_START"
)

base_fit <- sem(
    base_model,
    data=d,
    group="CYCLE",
    ordered=items,
    estimator="WLSMV",
    std.lv=TRUE,
    sampling_weights="WTMEC4YR",
    sampling_weights_type="design",
    group_equal=c(
        "loadings",
        "thresholds"
    )
)

message(
    "BASE_DONE"
)

rows <- list()

total_models <- (
    length(exposures)
    * length(items)
)

model_index <- 0

for (
    exposure in exposures
) {

    for (
        item in items
    ) {

        model_index <- (
            model_index + 1
        )

        message(
            sprintf(
                "START|%d|%d|%s|%s",
                model_index,
                total_models,
                exposure,
                item
            )
        )

        model <- paste0(
            base_model,
            "\n",
            item,
            " ~ ",
            exposure
        )

        fit <- try(
            sem(
                model,
                data=d,
                group="CYCLE",
                ordered=items,
                estimator="WLSMV",
                std.lv=TRUE,
                sampling_weights="WTMEC4YR",
                sampling_weights_type="design",
                group_equal=c(
                    "loadings",
                    "thresholds"
                )
            ),
            silent=TRUE
        )

        if (
            inherits(
                fit,
                "try-error"
            )
        ) {
            message(
                sprintf(
                    "DONE|%d|%d|%s|%s",
                    model_index,
                    total_models,
                    exposure,
                    item
                )
            )

            next
        }

        pe <- parameterEstimates(
            fit,
            standardized=TRUE,
            ci=TRUE
        )

        q <- pe[
            pe$op == "~" &
            pe$lhs == item &
            pe$rhs == exposure,
            c(
                "lhs",
                "rhs",
                "group",
                "est",
                "se",
                "z",
                "pvalue",
                "ci.lower",
                "ci.upper",
                "std.all"
            )
        ]

        if (
            nrow(q) == 0
        ) {
            message(
                sprintf(
                    "DONE|%d|%d|%s|%s",
                    model_index,
                    total_models,
                    exposure,
                    item
                )
            )

            next
        }

        q$item <- item
        q$exposure <- exposure

        rows[[length(rows) + 1]] <- q

        message(
            sprintf(
                "DONE|%d|%d|%s|%s",
                model_index,
                total_models,
                exposure,
                item
            )
        )
    }
}

out <- do.call(
    rbind,
    rows
)

levels_cycle <- levels(
    d$CYCLE
)

out$cycle <- levels_cycle[
    out$group
]

out$q_global <- p.adjust(
    out$pvalue,
    method="BH"
)

out$q_exposure <- ave(
    out$pvalue,
    out$exposure,
    FUN=function(x) {
        p.adjust(
            x,
            method="BH"
        )
    }
)

fm <- fitMeasures(
    base_fit,
    c(
        "cfi",
        "tli",
        "rmsea",
        "srmr"
    )
)

fitrow <- data.frame(
    cfi=unname(
        fm["cfi"]
    ),
    tli=unname(
        fm["tli"]
    ),
    rmsea=unname(
        fm["rmsea"]
    ),
    srmr=unname(
        fm["srmr"]
    )
)

write.csv(
    out,
    file.path(
        results_dir,
        "23_mimic_dif_paths.csv"
    ),
    row.names=FALSE
)

write.csv(
    fitrow,
    file.path(
        results_dir,
        "23_mimic_dif_base_fit.csv"
    ),
    row.names=FALSE
)
