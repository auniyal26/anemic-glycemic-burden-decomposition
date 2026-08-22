
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("lavaan", quietly=TRUE)) {
    install.packages("lavaan", repos="https://cloud.r-project.org", lib=user_lib)
}

library(lavaan)

d <- read.csv(input_file)
d$CYCLE <- factor(d$CYCLE)

items <- c("DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090")

model <- "
Somatic =~ DPQ030 + DPQ040 + DPQ050
CognitiveAffective =~ DPQ010 + DPQ020 + DPQ060 + DPQ070 + DPQ080 + DPQ090

Somatic ~ A_Z + G_HBA1C_Z + AG_Z + RIDAGEYR_Z + INDFMPIR_Z + BMXBMI_Z + EGFR_2021_Z + RIAGENDR_2 + RIDRETH1_2 + RIDRETH1_3 + RIDRETH1_4 + RIDRETH1_5 + EDUC3_2 + EDUC3_3 + SMOKING3_1 + SMOKING3_2
CognitiveAffective ~ A_Z + G_HBA1C_Z + AG_Z + RIDAGEYR_Z + INDFMPIR_Z + BMXBMI_Z + EGFR_2021_Z + RIAGENDR_2 + RIDRETH1_2 + RIDRETH1_3 + RIDRETH1_4 + RIDRETH1_5 + EDUC3_2 + EDUC3_3 + SMOKING3_1 + SMOKING3_2
"

fit <- sem(
    model,
    data=d,
    group="CYCLE",
    ordered=items,
    estimator="WLSMV",
    std.lv=TRUE,
    sampling_weights="WTMEC4YR",
    sampling_weights_type="design",
    group_equal=c("loadings","thresholds")
)

pe <- parameterEstimates(
    fit,
    standardized=TRUE,
    ci=TRUE
)

keep <- pe[
    pe$op == "~" &
    pe$lhs %in% c("Somatic","CognitiveAffective") &
    pe$rhs %in% c("A_Z","G_HBA1C_Z","AG_Z"),
    c("lhs","op","rhs","group","est","se","z","pvalue","ci.lower","ci.upper","std.all")
]

levels_cycle <- levels(d$CYCLE)
keep$cycle <- levels_cycle[keep$group]

fm <- fitMeasures(fit, c("cfi","tli","rmsea","srmr"))

fitrow <- data.frame(
    cfi=unname(fm["cfi"]),
    tli=unname(fm["tli"]),
    rmsea=unname(fm["rmsea"]),
    srmr=unname(fm["srmr"])
)

write.csv(keep, file.path(results_dir, "21_mimic_paths.csv"), row.names=FALSE)
write.csv(fitrow, file.path(results_dir, "21_mimic_fit.csv"), row.names=FALSE)
