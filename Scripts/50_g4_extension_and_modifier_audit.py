from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
for p in [RESULTS, AUDIT]:
    p.mkdir(parents=True, exist_ok=True)

BASE = RESULTS / "47_domain_corrected_AG_input.csv"
if not BASE.exists():
    raise FileNotFoundError(f"Missing {BASE}. Run Scripts 46-47 first.")

print()
print("G4 EXTENSION + A x G4 + X-MODIFIER AUDIT")
print("=========================================")
print("Goals:")
print("  1) Rebuild G4 outcome-independently: HbA1c + FPG + log insulin + 2h OGTT.")
print("  2) Compare G3 vs G4 on the exact same OGTT analytic sample.")
print("  3) Test whether 2h OGTT adds phenotype information beyond the full G3 state.")
print("  4) Stress-test A x G4 and prespecified age/sex/BMI/eGFR modification.")
print("  5) Keep all representation fitting independent of PHQ/depression.")
print("No existing result file is overwritten.")
print()

A_PCS = [f"A_OI_PC{i}_FZ" for i in range(1, 4)]
G3_PCS = [f"G3_OI_PC{i}_FZ" for i in range(1, 4)]
X3 = ["RIDAGEYR", "RIAGENDR", "RACE", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"]
G4_RAW = ["LBXGH", "LBXGLU", "LOG_IN", "LBXGLT"]

DISC = {"0506": "D", "0708": "E"}
REPL = {"0910": "F", "1112": "G", "1314": "H", "1516": "I"}
ALL = {**DISC, **REPL}
DATA_DISC = ROOT / "Data" / "NHANES"
DATA_TEMP = ROOT / "Data" / "NHANES_Transfer"


def norm_cycle_value(x):
    s = str(x).strip().replace(".0", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(4) if digits else s


def xpt_path(base_dir: Path, stem: str) -> Path:
    for name in [f"{stem}.XPT", f"{stem}.xpt"]:
        p = base_dir / name
        if p.exists():
            return p
    matches = list(base_dir.glob(f"{stem}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {stem} under {base_dir}")


def cycle_base(cycle: str) -> Path:
    return (DATA_DISC if cycle in DISC else DATA_TEMP) / cycle


def insulin_path(cycle: str) -> Path:
    s = ALL[cycle]
    stem = f"GLU_{s}" if cycle in {"0506", "0708", "0910", "1112"} else f"INS_{s}"
    return xpt_path(cycle_base(cycle), stem)


def read_ogtt(cycle: str) -> pd.DataFrame:
    s = ALL[cycle]
    b = cycle_base(cycle)
    og = pd.read_sas(xpt_path(b, f"OGTT_{s}"), format="xport")
    if "LBXGLT" not in og.columns:
        raise ValueError(f"{cycle} OGTT file has no LBXGLT")

    keep = ["SEQN", "LBXGLT"]
    if "WTSOG2YR" in og.columns:
        keep.append("WTSOG2YR")
        out = og[keep].copy()
    else:
        demo = pd.read_sas(xpt_path(b, f"DEMO_{s}"), format="xport")
        if "WTSOG2YR" not in demo.columns:
            raise ValueError(f"{cycle}: WTSOG2YR not found in OGTT or DEMO")
        out = og[["SEQN", "LBXGLT"]].merge(
            demo[["SEQN", "WTSOG2YR"]], on="SEQN", how="left", validate="one_to_one"
        )
    out["CYCLE"] = cycle
    return out


def read_g3_raw(cycle: str) -> pd.DataFrame:
    s = ALL[cycle]
    b = cycle_base(cycle)
    ghb = pd.read_sas(xpt_path(b, f"GHB_{s}"), format="xport")
    glu = pd.read_sas(xpt_path(b, f"GLU_{s}"), format="xport")
    ins = pd.read_sas(insulin_path(cycle), format="xport")
    for name, df, cols in [
        ("GHB", ghb, ["SEQN", "LBXGH"]),
        ("GLU", glu, ["SEQN", "LBXGLU"]),
        ("INS", ins, ["SEQN", "LBXIN"]),
    ]:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ValueError(f"{cycle} {name} missing {miss}")
    out = ghb[["SEQN", "LBXGH"]].merge(
        glu[["SEQN", "LBXGLU"]], on="SEQN", how="outer", validate="one_to_one"
    ).merge(
        ins[["SEQN", "LBXIN"]], on="SEQN", how="outer", validate="one_to_one"
    )
    out.loc[pd.to_numeric(out["LBXIN"], errors="coerce") <= 0, "LBXIN"] = np.nan
    out["LOG_IN"] = np.log(pd.to_numeric(out["LBXIN"], errors="coerce"))
    out["CYCLE"] = cycle
    return out


def weighted_pca_fit(df, vars_, wcol):
    X = df[vars_].to_numpy(float)
    w = df[wcol].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X, w = X[ok], w[ok]
    if len(X) < 3 or w.sum() <= 0:
        raise RuntimeError(f"No valid weighted rows for PCA: {vars_}")
    sw = w.sum()
    mu = np.sum(w[:, None] * X, axis=0) / sw
    sd = np.sqrt(np.sum(w[:, None] * (X - mu) ** 2, axis=0) / sw)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        raise RuntimeError(f"Invalid weighted scale for {vars_}")
    Z = (X - mu) / sd
    cov = (Z.T * w) @ Z / sw
    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, L = vals[order], vecs[:, order]
    for j in range(L.shape[1]):
        anchor = np.argmax(np.abs(L[:, j]))
        if L[anchor, j] < 0:
            L[:, j] *= -1
    if L[:, 0].sum() < 0:
        L[:, 0] *= -1
    return {"mean": mu, "sd": sd, "L": L, "evals": vals, "evr": vals / vals.sum()}


def project(df, vars_, model):
    X = df[vars_].to_numpy(float)
    Z = (X - model["mean"]) / model["sd"]
    return Z @ model["L"]


def weighted_mean_sd(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0 or w.sum() <= 0:
        raise RuntimeError("No valid rows for weighted scaling")
    mu = np.sum(w * x) / w.sum()
    sd = np.sqrt(np.sum(w * (x - mu) ** 2) / w.sum())
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Invalid weighted SD")
    return float(mu), float(sd)


def reconstruction_energy(df, vars_, model, k=3):
    X = df[vars_].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    X = X[ok]
    if len(X) == 0:
        return np.nan
    Z = (X - model["mean"]) / model["sd"]
    S = Z @ model["L"][:, :k]
    Zhat = S @ model["L"][:, :k].T
    den = np.sum(Z ** 2)
    return float(1 - np.sum((Z - Zhat) ** 2) / den) if den > 0 else np.nan


# ------------------------------------------------------------------
# Assemble full OGTT physiology frame without reading any PHQ variable.
# ------------------------------------------------------------------
base = pd.read_csv(BASE)
base["CYCLE"] = base["CYCLE"].map(norm_cycle_value)
base = base[base["CYCLE"].isin(ALL)].copy()

raw_rows = []
for cy in ALL:
    g3 = read_g3_raw(cy)
    og = read_ogtt(cy)
    x = g3.merge(og, on=["SEQN", "CYCLE"], how="outer", validate="one_to_one")
    raw_rows.append(x)
raw = pd.concat(raw_rows, ignore_index=True)
if raw.duplicated(["SEQN", "CYCLE"]).any():
    raise RuntimeError("Duplicate SEQN/CYCLE in raw G4 frame")

d = base.merge(raw, on=["SEQN", "CYCLE"], how="left", validate="one_to_one")
for c in G4_RAW + ["WTSOG2YR"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

# Correct pooled OGTT weights for each analysis period.
d["OGTT_WT_PERIOD"] = np.where(
    d["CYCLE"].isin(DISC), d["WTSOG2YR"] / 2.0,
    np.where(d["CYCLE"].isin(REPL), d["WTSOG2YR"] / 4.0, np.nan),
)

# Outcome-independent G4 fit: eligible adult/nonpreg + positive OGTT weight + G4 markers only.
fit = d[
    d["CYCLE"].isin(DISC) & d["ELIGIBLE_ADULT_NONPREG"].eq(1)
].dropna(subset=G4_RAW + ["OGTT_WT_PERIOD"]).copy()
fit = fit[fit["OGTT_WT_PERIOD"] > 0].copy()
if fit.empty:
    raise RuntimeError("Outcome-independent G4 discovery fit is empty")

g4_model = weighted_pca_fit(fit, G4_RAW, "OGTT_WT_PERIOD")
score = np.full((len(d), 4), np.nan)
ok = d[G4_RAW].notna().all(axis=1)
score[ok.to_numpy(), :] = project(d.loc[ok], G4_RAW, g4_model)
for j in range(4):
    d[f"G4_PC{j+1}"] = score[:, j]

# Freeze discovery score scales. Primary G4 keeps K=3 as in the earlier G4 analysis.
G4_PCS = []
scale_rows = []
for j in range(1, 4):
    col = f"G4_PC{j}"
    q = d.loc[fit.index, [col, "OGTT_WT_PERIOD"]].dropna()
    mu, sd = weighted_mean_sd(q[col], q["OGTT_WT_PERIOD"])
    z = f"G4_PC{j}_FZ"
    d[z] = (d[col] - mu) / sd
    G4_PCS.append(z)
    scale_rows.append({"component": col, "mean": mu, "sd": sd, "fit_n": len(fit)})

# Frozen discovery scaling for raw 2h OGTT itself, used for the clean nested incremental test.
ogtt_mu, ogtt_sd = weighted_mean_sd(fit["LBXGLT"], fit["OGTT_WT_PERIOD"])
d["OGTT2H_DISC_Z"] = (d["LBXGLT"] - ogtt_mu) / ogtt_sd

# Modifiers: centered/scaled using outcome-independent discovery physiology fit.
modifier_specs = {
    "AGE10_C": ("RIDAGEYR", 10.0),
    "BMI5_C": ("BMXBMI", 5.0),
    "EGFR10_C": ("EGFR_2021", 10.0),
}
for new, (raw_col, divisor) in modifier_specs.items():
    vals = pd.to_numeric(fit[raw_col], errors="coerce")
    w = fit["OGTT_WT_PERIOD"]
    mu, _ = weighted_mean_sd(vals, w)
    d[new] = (pd.to_numeric(d[raw_col], errors="coerce") - mu) / divisor

d["FEMALE"] = np.where(pd.to_numeric(d["RIAGENDR"], errors="coerce") == 2, 1.0,
                       np.where(pd.to_numeric(d["RIAGENDR"], errors="coerce") == 1, 0.0, np.nan))

# Explicit interaction columns preserve hierarchical main effects in R.
axg = []
for i, a in enumerate(A_PCS, 1):
    for j, g in enumerate(G4_PCS, 1):
        n = f"A{i}_X_G4_{j}"
        d[n] = pd.to_numeric(d[a], errors="coerce") * pd.to_numeric(d[g], errors="coerce")
        axg.append(n)

modifier_cols = {}
for mod in ["AGE10_C", "FEMALE", "BMI5_C", "EGFR10_C"]:
    Aints, Gints = [], []
    for i, a in enumerate(A_PCS, 1):
        n = f"A{i}_X_{mod}"
        d[n] = pd.to_numeric(d[a], errors="coerce") * pd.to_numeric(d[mod], errors="coerce")
        Aints.append(n)
    for j, g in enumerate(G4_PCS, 1):
        n = f"G4_{j}_X_{mod}"
        d[n] = pd.to_numeric(d[g], errors="coerce") * pd.to_numeric(d[mod], errors="coerce")
        Gints.append(n)
    modifier_cols[mod] = (Aints, Gints)

# Structural diagnostics.
struct_rows = []
for pop, cycles in [("2005-2008", list(DISC)), ("2009-2016", list(REPL))]:
    sub = d[d["CYCLE"].isin(cycles)].dropna(subset=G4_RAW).copy()
    struct_rows.append({
        "population": pop,
        "n_complete_G4": len(sub),
        "K3_reconstruction_energy": reconstruction_energy(sub, G4_RAW, g4_model, 3),
        "K4_reconstruction_energy": reconstruction_energy(sub, G4_RAW, g4_model, 4),
    })
pd.DataFrame(struct_rows).to_csv(RESULTS / "50_g4_structural_transfer.csv", index=False)

load_rows = []
for j in range(4):
    for v, loading in zip(G4_RAW, g4_model["L"][:, j]):
        load_rows.append({"component": f"PC{j+1}", "variable": v, "loading": float(loading), "evr": float(g4_model["evr"][j])})
pd.DataFrame(load_rows).to_csv(RESULTS / "50_g4_frozen_loadings.csv", index=False)
pd.DataFrame(scale_rows).to_csv(RESULTS / "50_g4_frozen_scaling.csv", index=False)

# Domain flags are created without filtering the full survey design frame.
common = A_PCS + G3_PCS + G4_PCS + X3 + ["SOMATIC_SCORE", "OGTT2H_DISC_Z", "OGTT_WT_PERIOD"]

# Explicit preflight audit so an empty complete-case domain can never fail opaquely in R.
audit_rows = []
for pop, cycles in [("2005-2008", list(DISC)), ("2009-2016", list(REPL))]:
    sub = d[d["CYCLE"].isin(cycles)].copy()
    pos = sub[sub["OGTT_WT_PERIOD"].fillna(0) > 0].copy()
    row = {"population": pop, "rows_total": len(sub), "positive_ogtt_weight": len(pos)}
    for c in common:
        row[f"nonmissing__{c}"] = int(pos[c].notna().sum())
    row["complete_common"] = int(pos[common].notna().all(axis=1).sum())
    audit_rows.append(row)

domain_audit = pd.DataFrame(audit_rows)
domain_audit.to_csv(RESULTS / "50_g4_domain_preflight.csv", index=False)

for _, rr in domain_audit.iterrows():
    print(f"PRECHECK {rr['population']}: positive OGTT-weight n={int(rr['positive_ogtt_weight'])}; complete common n={int(rr['complete_common'])}")
    if int(rr["complete_common"]) == 0:
        counts = []
        npos = max(int(rr["positive_ogtt_weight"]), 1)
        for c in common:
            counts.append((c, int(rr[f"nonmissing__{c}"])))
        counts.sort(key=lambda x: x[1])
        print("  Lowest non-missing variables:", ", ".join(f"{c}={n}" for c, n in counts[:8]))

d["DOMAIN_G4_X3"] = d[common].notna().all(axis=1).astype(int)
d.loc[~(d["OGTT_WT_PERIOD"] > 0), "DOMAIN_G4_X3"] = 0

# Hard fail here, before R, if a requested population has no valid exact-same-sample domain.
bad = domain_audit.loc[domain_audit["complete_common"] == 0, "population"].tolist()
if bad:
    raise RuntimeError(
        "Exact G3/G4/X3 analytic domain is empty for: " + ", ".join(bad) +
        ". See Results/50_g4_domain_preflight.csv; the terminal now prints the variables causing the collapse."
    )

OUT = RESULTS / "50_g4_modifier_input.csv"
d.to_csv(OUT, index=False)

# ------------------------------------------------------------------
# R survey inference: design first, domain second.
# ------------------------------------------------------------------
rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
if rscript is None:
    raise FileNotFoundError("Rscript not found on PATH. Activate the physiodecomp conda environment and run with `python`, not the base-Python path.")

r_code = r'''
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
'''

r_file = AUDIT / "50_g4_modifier_audit.R"
r_file.write_text(r_code, encoding="utf-8")
r_env = dict(__import__("os").environ)
# Do not allow stale project/user R libraries to override the active Conda R installation.
r_env.pop("R_LIBS_USER", None)
r_env.pop("R_LIBS_SITE", None)

proc = subprocess.run(
    [rscript, str(r_file), str(OUT), str(RESULTS)],
    capture_output=True,
    text=True,
    env=r_env,
)
if proc.stdout:
    print(proc.stdout)
if proc.returncode != 0:
    if proc.stderr:
        print(proc.stderr)
    raise RuntimeError(f"R audit failed with exit code {proc.returncode}")

struct = pd.read_csv(RESULTS / "50_g4_structural_transfer.csv")
print("PASS  G4 outcome-independent discovery fit n=", len(fit), sep="")
print(f"PASS  G4 K3 discovery reconstruction energy={struct.loc[struct.population.eq('2005-2008'),'K3_reconstruction_energy'].iloc[0]:.6f}")
print(f"PASS  G4 K3 2009-2016 frozen reconstruction energy={struct.loc[struct.population.eq('2009-2016'),'K3_reconstruction_energy'].iloc[0]:.6f}")
print("PASS  G3-vs-G4 same-sample, 2h-OGTT increment, A x G4, and X-modifier tests saved under prefix 50.")
print("PASS  Existing outputs remain untouched.")

summary = {
    "script": 50,
    "g4_raw": G4_RAW,
    "g4_k_primary": 3,
    "fit_cycles": list(DISC),
    "replication_cycles": list(REPL),
    "fit_n": int(len(fit)),
    "ogtt_discovery_mean": ogtt_mu,
    "ogtt_discovery_sd": ogtt_sd,
    "notes": [
        "G4 fitted without any PHQ/depression variable.",
        "OGTT survey design created before analytic-domain subsetting.",
        "G3 and G4 compared on the same OGTT-complete X3 sample.",
        "2h OGTT incremental test is nested on top of the full G3 state.",
        "Modifier family is limited to age, sex, BMI, and eGFR; BH q-values are reported.",
    ],
}
(AUDIT / "50_g4_modifier_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
