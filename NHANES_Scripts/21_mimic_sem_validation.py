from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

FIG.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RLIB.mkdir(parents=True, exist_ok=True)

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

cohort = pd.read_parquet(PROCESSED / "nhanes_adjustment_cohort_20plus.parquet")
working = pd.read_parquet(PROCESSED / "nhanes_core_working.parquet")

phq = working[["SEQN", "CYCLE"] + ITEMS].copy()
df = cohort.merge(phq, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")

for c in ITEMS:
    df.loc[df[c].abs() < 1e-10, c] = 0
    df.loc[~df[c].isin([0, 1, 2, 3]), c] = np.nan

required = [
    "A", "G_HBA1C", "WTMEC4YR",
    "RIDAGEYR", "RIAGENDR", "RIDRETH1",
    "INDFMPIR", "EDUC3", "SMOKING3",
    "BMXBMI", "EGFR_2021"
] + ITEMS

df = df.dropna(subset=required).copy()
df["CYCLE"] = df["CYCLE"].astype(str)

def weighted_mean_sd(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.sum(w * x) / np.sum(w)
    v = np.sum(w * (x - m) ** 2) / np.sum(w)
    return m, np.sqrt(v)

for cycle in sorted(df["CYCLE"].unique()):
    idx = df["CYCLE"] == cycle
    w = df.loc[idx, "WTMEC4YR"]

    for col in ["A", "G_HBA1C", "RIDAGEYR", "INDFMPIR", "BMXBMI", "EGFR_2021"]:
        m, s = weighted_mean_sd(df.loc[idx, col], w)
        df.loc[idx, f"{col}_Z"] = (df.loc[idx, col] - m) / s

df["AG_Z"] = df["A_Z"] * df["G_HBA1C_Z"]

for col in ["RIAGENDR", "RIDRETH1", "EDUC3", "SMOKING3"]:
    dummies = pd.get_dummies(df[col].astype(int), prefix=col, drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1)

dummy_cols = [
    c for c in df.columns
    if c.startswith("RIAGENDR_")
    or c.startswith("RIDRETH1_")
    or c.startswith("EDUC3_")
    or c.startswith("SMOKING3_")
]

export_cols = [
    "SEQN", "CYCLE", "WTMEC4YR",
    "A_Z", "G_HBA1C_Z", "AG_Z",
    "RIDAGEYR_Z", "INDFMPIR_Z", "BMXBMI_Z", "EGFR_2021_Z"
] + dummy_cols + ITEMS

input_csv = RESULTS / "21_mimic_input.csv"
df[export_cols].to_csv(input_csv, index=False)

rscript = shutil.which("Rscript")

if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])

print()
print("LATENT MIMIC / SEM VALIDATION")
print("=============================")

if rscript is None:
    print("FAIL  Rscript not found.")
    raise SystemExit(1)

covariates = [
    "RIDAGEYR_Z", "INDFMPIR_Z", "BMXBMI_Z", "EGFR_2021_Z"
] + dummy_cols

covariate_rhs = " + ".join(covariates)

r_code = f'''
args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
results_dir <- args[2]
user_lib <- args[3]

dir.create(user_lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("lavaan", quietly=TRUE)) {{
    install.packages("lavaan", repos="https://cloud.r-project.org", lib=user_lib)
}}

library(lavaan)

d <- read.csv(input_file)
d$CYCLE <- factor(d$CYCLE)

items <- c("DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090")

model <- "
Somatic =~ DPQ030 + DPQ040 + DPQ050
CognitiveAffective =~ DPQ010 + DPQ020 + DPQ060 + DPQ070 + DPQ080 + DPQ090

Somatic ~ A_Z + G_HBA1C_Z + AG_Z + {covariate_rhs}
CognitiveAffective ~ A_Z + G_HBA1C_Z + AG_Z + {covariate_rhs}
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
'''

r_path = AUDIT / "21_mimic_runner.R"
r_path.write_text(r_code, encoding="utf-8")

proc = subprocess.run(
    [str(rscript), str(r_path), str(input_csv), str(RESULTS), str(RLIB)],
    capture_output=True,
    text=True
)

if proc.returncode != 0:
    print("FAIL  MIMIC model did not complete.")
    print(proc.stderr[-2500:])
    raise SystemExit(proc.returncode)

paths = pd.read_csv(RESULTS / "21_mimic_paths.csv")
fit = pd.read_csv(RESULTS / "21_mimic_fit.csv").iloc[0]

term_labels = {
    "A_Z": "A",
    "G_HBA1C_Z": "G",
    "AG_Z": "A×G"
}

for latent in ["Somatic", "CognitiveAffective"]:
    plot = paths[paths["lhs"] == latent].copy()

    plt.figure(figsize=(8, 5))
    positions = {"A_Z": 0, "G_HBA1C_Z": 1, "AG_Z": 2}
    offsets = {"0506": -0.08, "0708": 0.08}

    for cycle in ["0506", "0708"]:
        d = plot[plot["cycle"].astype(str) == cycle]
        x = np.array([positions[t] for t in d["rhs"]]) + offsets[cycle]
        plt.errorbar(
            x,
            d["est"],
            yerr=[
                d["est"] - d["ci.lower"],
                d["ci.upper"] - d["est"]
            ],
            fmt="o",
            capsize=4,
            label=cycle
        )

    plt.axhline(0, linewidth=0.8)
    plt.xticks([0, 1, 2], ["A", "G", "A×G"])
    plt.ylabel("Latent regression coefficient")
    plt.title(f"{latent} factor: MIMIC physiological paths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"21_{latent.lower()}_mimic_paths.png", dpi=300)
    plt.close()

def verdict(latent, term):
    d = paths[
        (paths["lhs"] == latent)
        & (paths["rhs"] == term)
    ].sort_values("cycle")

    if len(d) != 2:
        return "INCOMPLETE"

    same = np.sign(d["est"].values[0]) == np.sign(d["est"].values[1])
    nonzero = ((d["ci.lower"] > 0) | (d["ci.upper"] < 0)).values

    if same and nonzero.all():
        return "REPLICATES"
    if same and nonzero.any():
        return "PARTIAL"
    if same:
        return "SAME DIRECTION, UNCERTAIN"
    return "DOES NOT REPLICATE"

print(f"PASS  Weighted ordinal MIMIC model fit: CFI={fit['cfi']:.3f}, RMSEA={fit['rmsea']:.3f}, SRMR={fit['srmr']:.3f}.")
print("PASS  Measurement loadings and thresholds constrained invariant across cycles.")
print("NOTE  This latent model uses sampling weights; PSU/strata population inference remains anchored to script 16.")
print()
print("SOMATIC LATENT FACTOR")
print("A     ", verdict("Somatic", "A_Z"))
print("G     ", verdict("Somatic", "G_HBA1C_Z"))
print("A×G   ", verdict("Somatic", "AG_Z"))
print()
print("COGNITIVE-AFFECTIVE LATENT FACTOR")
print("A     ", verdict("CognitiveAffective", "A_Z"))
print("G     ", verdict("CognitiveAffective", "G_HBA1C_Z"))
print("A×G   ", verdict("CognitiveAffective", "AG_Z"))
print()
print("Full latent paths and figures saved.")
print("NEXT  Compare latent somatic paths with the survey-weighted observed-score results.")
