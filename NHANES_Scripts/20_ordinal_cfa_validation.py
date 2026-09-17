from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"D:\Hematological and Glycemic Decomposision of Depression")
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
FIG = RESULTS / "Figures"
AUDIT = ROOT / "Audit"

FIG.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)

ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]

cohort = pd.read_parquet(PROCESSED / "nhanes_adjustment_cohort_20plus.parquet")
working = pd.read_parquet(PROCESSED / "nhanes_core_working.parquet")

phq = working[["SEQN", "CYCLE"] + ITEMS].copy()
df = cohort[["SEQN", "CYCLE"]].merge(
    phq,
    on=["SEQN", "CYCLE"],
    how="left",
    validate="one_to_one"
)

for c in ITEMS:
    df.loc[df[c].abs() < 1e-10, c] = 0
    df.loc[~df[c].isin([0, 1, 2, 3]), c] = pd.NA

df = df.dropna(subset=ITEMS).copy()
df["CYCLE"] = df["CYCLE"].astype(str)

input_csv = RESULTS / "20_cfa_input.csv"
df.to_csv(input_csv, index=False)

rscript = shutil.which("Rscript")

if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])

print()
print("ORDINAL CFA VALIDATION")
print("======================")

if rscript is None:
    print("NEED R  Rscript was not found.")
    print("Install R, then rerun this same script. No other change is needed.")
    raise SystemExit(1)

r_code = r'''
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
'''

r_path = AUDIT / "20_cfa_runner.R"
r_path.write_text(r_code, encoding="utf-8")

proc = subprocess.run(
    [str(rscript), str(r_path), str(input_csv), str(RESULTS)],
    capture_output=True,
    text=True
)

if proc.returncode != 0:
    print("FAIL  lavaan CFA did not complete.")
    print(proc.stderr[-2000:])
    raise SystemExit(proc.returncode)

fit = pd.read_csv(RESULTS / "20_cfa_fit.csv")
loadings = pd.read_csv(RESULTS / "20_cfa_two_factor_loadings.csv")
inv = pd.read_csv(RESULTS / "20_cfa_invariance.csv")

comparison = []

for cycle in sorted(fit["cycle"].astype(str).unique()):
    d = fit[fit["cycle"].astype(str) == cycle].set_index("model")
    one = d.loc["one_factor"]
    two = d.loc["two_factor"]

    comparison.append({
        "cycle": cycle,
        "delta_cfi": two["cfi"] - one["cfi"],
        "delta_rmsea": two["rmsea"] - one["rmsea"],
        "two_cfi": two["cfi"],
        "two_rmsea": two["rmsea"],
        "two_srmr": two["srmr"]
    })

comparison = pd.DataFrame(comparison)
comparison.to_csv(RESULTS / "20_cfa_model_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
for cycle in sorted(fit["cycle"].astype(str).unique()):
    d = fit[fit["cycle"].astype(str) == cycle]
    x = np.arange(len(d))
    plt.plot(x, d["cfi"], marker="o", label=cycle)
plt.xticks([0, 1], ["One factor", "Somatic + cognitive-affective"])
plt.ylim(0, 1.02)
plt.ylabel("CFI")
plt.title("Ordinal CFA model fit across NHANES cycles")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "20_cfa_model_fit.png", dpi=300)
plt.close()

somatic = loadings[loadings["lhs"] == "Somatic"].copy()

plt.figure(figsize=(8, 5))
for cycle in sorted(somatic["cycle"].astype(str).unique()):
    d = somatic[somatic["cycle"].astype(str) == cycle]
    x = np.arange(len(d))
    plt.plot(x, d["est.std"], marker="o", label=cycle)
plt.xticks([0, 1, 2], ["Sleep", "Fatigue", "Appetite"])
plt.ylim(0, 1)
plt.ylabel("Standardized loading")
plt.title("Ordinal CFA somatic-factor loadings")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "20_somatic_factor_loadings.png", dpi=300)
plt.close()

two_better = (
    (comparison["delta_cfi"] > 0)
    & (comparison["delta_rmsea"] < 0)
).all()

somatic_stable = (
    somatic.groupby("cycle")["est.std"].min() >= 0.4
).all()

inv = inv.set_index("model")
delta_cfi_load = inv.loc["equal_loadings", "cfi"] - inv.loc["configural", "cfi"]
delta_cfi_thresh = inv.loc["equal_loadings_thresholds", "cfi"] - inv.loc["equal_loadings", "cfi"]

print("PASS  WLSMV used for ordinal PHQ-9 items.")

if two_better:
    print("PASS  Somatic + cognitive-affective structure fits better than one-factor depression in both cycles.")
else:
    print("MIXED  Two-factor model does not improve fit consistently across both cycles.")

if somatic_stable:
    print("PASS  Sleep, fatigue and appetite load consistently on the somatic factor.")
else:
    print("WARNING  At least one somatic item has a weak loading.")

if abs(delta_cfi_load) <= 0.01:
    print("PASS  Somatic/cognitive-affective loadings are approximately invariant across cycles.")
else:
    print("WARNING  Factor loadings shift meaningfully across cycles.")

if abs(delta_cfi_thresh) <= 0.01:
    print("PASS  Ordinal thresholds are approximately invariant across cycles.")
else:
    print("WARNING  Item thresholds shift meaningfully across cycles.")

print("PASS  Full fit indices, loadings and invariance results saved.")
print()
print("NEXT  If the two-factor structure holds, regress the latent somatic factor on A, G and A×G with a MIMIC/SEM model.")
