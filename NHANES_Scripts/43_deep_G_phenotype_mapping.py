from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
RLIB = ROOT / "R_library"

for p in [RESULTS, AUDIT, RLIB]:
    p.mkdir(parents=True, exist_ok=True)

PARITY = RESULTS / "25_survey_parity_input.csv"
G3_SCORES = PROCESSED / "42_G3_discovery_frozen_scores.parquet"
G4_SCORES = PROCESSED / "42_G4_discovery_frozen_scores.parquet"

for path in [PARITY, G3_SCORES, G4_SCORES]:
    if not path.exists():
        raise FileNotFoundError(path)

print()
print("DEEP G PHENOTYPE MAPPING")
print("========================")
print("Primary: frozen G3 = HbA1c + fasting glucose + log fasting insulin")
print("Secondary: frozen G4 = G3 + 2-hour OGTT")
print("Primary outcome: somatic PHQ symptom score")
print("Adjustment: frozen X3")
print("Formal inference: R survey::svyglm + survey::regTermTest")
print()

X3 = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "INDFMPIR",
    "EDUC3",
    "SMOKING3",
    "BMXBMI",
    "EGFR_2021",
]

parity = pd.read_csv(PARITY)
parity["CYCLE"] = (
    parity["CYCLE"].astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .replace({"506": "0506", "708": "0708"})
)

required_base = [
    "SEQN", "CYCLE", "STRATUM", "PSU",
    "SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C",
] + X3

missing = [c for c in required_base if c not in parity.columns]
if missing:
    raise ValueError(f"Parity input missing: {missing}")

def weighted_mean_sd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    sw = w.sum()
    m = np.sum(w * x) / sw
    v = np.sum(w * (x - m) ** 2) / sw
    return float(m), float(np.sqrt(v))

def make_input(score_path, block):
    s = pd.read_parquet(score_path).copy()
    s["CYCLE"] = (
        s["CYCLE"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .replace({"506": "0506", "708": "0708"})
    )

    if block == "G3":
        raw_pcs = [f"G3_FROZEN_PC{i}" for i in range(1, 4)]
        raw_weight = "WTSAF2YR"
        model_weight = "WTFAST4YR_PHENO"
    else:
        raw_pcs = [f"G4_FROZEN_PC{i}" for i in range(1, 5)]
        raw_weight = "WTSOG2YR"
        model_weight = "WTOGTT4YR_PHENO"

    need = ["SEQN", "CYCLE", raw_weight] + raw_pcs
    miss = [c for c in need if c not in s.columns]
    if miss:
        raise ValueError(f"{block} scores missing: {miss}")

    s = s[need].copy()
    s[model_weight] = pd.to_numeric(s[raw_weight], errors="coerce") / 2.0

    zpcs = []
    scale_rows = []
    for pc in raw_pcs:
        m, sd = weighted_mean_sd(s[pc], s[model_weight])
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"Invalid SD for {pc}")
        z = pc + "_Z"
        s[z] = (s[pc] - m) / sd
        zpcs.append(z)
        scale_rows.append({
            "block": block,
            "component": pc,
            "weighted_mean": m,
            "weighted_sd": sd,
            "z_column": z,
        })

    d = parity.merge(
        s[["SEQN", "CYCLE", model_weight] + zpcs],
        on=["SEQN", "CYCLE"],
        how="inner",
        validate="one_to_one",
    )

    complete = (
        ["SOMATIC_SCORE", "PHQ9_TOTAL", "A", "G_HBA1C",
         "STRATUM", "PSU", model_weight]
        + X3 + zpcs
    )
    d = d.dropna(subset=complete).copy()
    d = d[pd.to_numeric(d[model_weight], errors="coerce") > 0].copy()

    return d, zpcs, model_weight, pd.DataFrame(scale_rows)

g3, g3pcs, g3weight, scale3 = make_input(G3_SCORES, "G3")
g4, g4pcs, g4weight, scale4 = make_input(G4_SCORES, "G4")

pd.concat([scale3, scale4], ignore_index=True).to_csv(
    RESULTS / "43_G_component_score_scaling.csv", index=False
)

g3_path = RESULTS / "43_G3_phenotype_input.csv"
g4_path = RESULTS / "43_G4_phenotype_input.csv"
g3.to_csv(g3_path, index=False)
g4.to_csv(g4_path, index=False)

print(f"PASS  G3 X3 phenotype sample: n={len(g3):,}")
print(f"PASS  G4 X3 phenotype sample: n={len(g4):,}")
print()

rscript = shutil.which("Rscript")
if rscript is None:
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    if candidates:
        rscript = str(candidates[-1])
if rscript is None:
    raise FileNotFoundError("Rscript not found")

r_code = r'''
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
'''

r_path = AUDIT / "43_G_phenotype_mapping.R"
r_path.write_text(r_code, encoding="utf-8")

# Parse R before running it.
r_path_r = str(r_path).replace("\\", "/")
parse_expr = f'parse(file="{r_path_r}")'
parse_proc = subprocess.run(
    [str(rscript), "-e", parse_expr],
    capture_output=True,
    text=True,
)

if parse_proc.returncode != 0:
    print("FAIL  Generated R script did not parse.")
    print(parse_proc.stdout[-4000:])
    print(parse_proc.stderr[-4000:])
    raise SystemExit(parse_proc.returncode)

print("PASS  Generated R script syntax check.")

proc = subprocess.run(
    [str(rscript), str(r_path), str(g3_path), str(g4_path), str(RESULTS), str(RLIB)],
    capture_output=True,
    text=True,
)

if proc.returncode != 0:
    print("FAIL  R survey phenotype mapping failed.")
    print(proc.stdout[-4000:])
    print(proc.stderr[-6000:])
    raise SystemExit(proc.returncode)

coef = pd.read_csv(RESULTS / "43_G_component_coefficients.csv")
tests = pd.read_csv(RESULTS / "43_G_component_block_tests.csv")
r2 = pd.read_csv(RESULTS / "43_G_model_weighted_R2.csv")

def bh_adjust(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    restored = np.empty(n)
    restored[order] = adj
    out[np.where(ok)[0]] = restored
    return out

coef["p_BH_within_block_outcome"] = np.nan
mask = coef["model"].eq("PC_block")
for (_, _), idx in coef[mask].groupby(["block", "outcome"]).groups.items():
    coef.loc[idx, "p_BH_within_block_outcome"] = bh_adjust(
        coef.loc[idx, "p"].to_numpy(float)
    )
coef.to_csv(RESULTS / "43_G_component_coefficients.csv", index=False)

delta_rows = []
for (block, outcome), q in r2.groupby(["block", "outcome"]):
    v = dict(zip(q["model"], q["weighted_R2"]))
    delta_rows.append({
        "block": block,
        "outcome": outcome,
        "delta_scalar_G_beyond_XA":
            v["scalar_G"] - v["X_plus_A"],
        "delta_PC1_beyond_XA":
            v["PC1_only"] - v["X_plus_A"],
        "delta_PCblock_beyond_XA":
            v["PC_block"] - v["X_plus_A"],
        "delta_PCblock_beyond_scalarG":
            v["union_scalar_plus_PCs"] - v["scalar_G"],
        "delta_scalarG_beyond_PCblock":
            v["union_scalar_plus_PCs"] - v["PC_block"],
    })

delta = pd.DataFrame(delta_rows)
delta.to_csv(RESULTS / "43_G_incremental_information.csv", index=False)

audit = {
    "script": "43_deep_G_phenotype_mapping_v3.py",
    "primary_block": "G3_core",
    "secondary_block": "G4_extended",
    "primary_outcome": "SOMATIC_SCORE",
    "secondary_outcome": "PHQ9_TOTAL",
    "adjustment": "X3_kidney_direct_effect",
    "hematology_control": "scalar A",
    "G3_weight": "WTSAF2YR / 2",
    "G4_weight": "WTSOG2YR / 2",
    "formal_inference": "survey::svyglm + survey::regTermTest",
    "G3_n": int(len(g3)),
    "G4_n": int(len(g4)),
}
(AUDIT / "43_deep_G_phenotype_mapping.json").write_text(
    json.dumps(audit, indent=2), encoding="utf-8"
)

print()
print("FORMAL BLOCK TESTS — SOMATIC")
print(
    tests[tests["outcome"].eq("SOMATIC_SCORE")][
        ["block","test","F","df_num","df_den","p","design_df"]
    ].to_string(index=False)
)
print()

print("INDIVIDUAL COMPONENTS — SOMATIC")
print(
    coef[
        coef["outcome"].eq("SOMATIC_SCORE")
        & coef["model"].eq("PC_block")
    ][
        ["block","term","beta","se","ci_low","ci_high","p",
         "p_BH_within_block_outcome"]
    ].to_string(index=False)
)
print()

print("INCREMENTAL INFORMATION — SOMATIC")
print(
    delta[delta["outcome"].eq("SOMATIC_SCORE")].to_string(index=False)
)
print()

print("Saved:")
print("  Results/43_G_component_coefficients.csv")
print("  Results/43_G_component_block_tests.csv")
print("  Results/43_G_model_weighted_R2.csv")
print("  Results/43_G_incremental_information.csv")
print("  Audit/43_G_phenotype_mapping.R")
print("  Audit/43_deep_G_phenotype_mapping.json")
