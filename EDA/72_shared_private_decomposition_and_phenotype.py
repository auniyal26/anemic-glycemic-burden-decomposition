from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PERM = 250
RHO_EFFECT_FLOOR = 0.10

ROOT = Path.cwd()
RESULTS = ROOT / "EDA" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    RESULTS / "63_deep_ddx_input.csv",
    ROOT / "63_deep_ddx_input.csv",
    Path("/mnt/data/63_deep_ddx_input.csv"),
]
INPUT = next((p for p in CANDIDATES if p.exists()), None)
if INPUT is None:
    raise FileNotFoundError("Could not find 63_deep_ddx_input.csv")

A_VARS = ["LBXHGB", "LBXRBCSI", "LBXMCVSI", "LBXRDW"]
G_VARS = ["LBXGH", "LBXGLU", "LOG_IN"]
OUTCOMES = {
    "SOMATIC": "SOMATIC_SCORE",
    "COGAFF": "COGAFF_SUM",
    "PHQ9_TOTAL": "PHQ9_TOTAL",
}
COVARS = ["RIDAGEYR", "RIAGENDR", "RACE", "INDFMPIR", "EDUC3", "SMOKING3", "BMXBMI", "EGFR_2021"]
DESIGN = ["SURVEY_WT", "STRATUM", "PSU", "CYCLE", "PERIOD"]


def wnorm(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    if np.any(~np.isfinite(w)) or np.sum(w) <= 0:
        raise ValueError("Invalid weights")
    return w / np.sum(w)


def weighted_mean_sd(X: np.ndarray, w: np.ndarray):
    wn = wnorm(w)
    mu = np.sum(X * wn[:, None], axis=0)
    Xc = X - mu
    sd = np.sqrt(np.sum((Xc ** 2) * wn[:, None], axis=0))
    if np.any(sd <= 0):
        raise ValueError("Zero weighted SD in physiology block")
    return mu, sd


def wcov(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    wn = wnorm(w)
    return (X * wn[:, None]).T @ Y


def invsqrt(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vals, vecs = np.linalg.eigh((S + S.T) / 2)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    wn = wnorm(w)
    mx, my = np.sum(wn * x), np.sum(wn * y)
    xc, yc = x - mx, y - my
    vx, vy = np.sum(wn * xc * xc), np.sum(wn * yc * yc)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(np.sum(wn * xc * yc) / np.sqrt(vx * vy))


def fit_weighted_cca(XA_raw: np.ndarray, XG_raw: np.ndarray, w: np.ndarray):
    ma, sa = weighted_mean_sd(XA_raw, w)
    mg, sg = weighted_mean_sd(XG_raw, w)
    A = (XA_raw - ma) / sa
    G = (XG_raw - mg) / sg

    Saa = wcov(A, A, w)
    Sgg = wcov(G, G, w)
    Sag = wcov(A, G, w)

    IA = invsqrt(Saa)
    IG = invsqrt(Sgg)
    P, rho, Qt = np.linalg.svd(IA @ Sag @ IG, full_matrices=False)
    Q = Qt.T

    WA_pair = IA @ P           # 4 x 3
    WG = IG @ Q                # 3 x 3

    # Complete A to a square invertible 4 x 4 canonical basis.
    _, _, vt = np.linalg.svd(P.T, full_matrices=True)
    p4 = vt[-1, :]
    WA = IA @ np.column_stack([P, p4])

    return {
        "A_mean": ma,
        "A_sd": sa,
        "G_mean": mg,
        "G_sd": sg,
        "WA": WA,
        "WG": WG,
        "rho": rho,
        "A_std": A,
        "G_std": G,
    }


def permutation_pvalues(A: np.ndarray, G: np.ndarray, w: np.ndarray, rho_obs: np.ndarray, n_perm=N_PERM):
    rng = np.random.default_rng(SEED)
    Saa = wcov(A, A, w)
    Sgg = wcov(G, G, w)
    IA, IG = invsqrt(Saa), invsqrt(Sgg)
    null = np.empty((n_perm, len(rho_obs)))
    for b in range(n_perm):
        idx = rng.permutation(len(G))
        Sag = wcov(A, G[idx], w)
        null[b] = np.linalg.svd(IA @ Sag @ IG, compute_uv=False)
    p = (1 + np.sum(null >= rho_obs[None, :], axis=0)) / (n_perm + 1)
    q95 = np.quantile(null, 0.95, axis=0)
    return p, q95


def z_names(shared_rank: int):
    names = []
    for j in range(3):
        k = j + 1
        if j < shared_rank:
            names += [f"SP_SHARED{k}", f"SP_DISCORD{k}"]
        else:
            names += [f"SP_A_PRIVATE{k}", f"SP_G_PRIVATE{k}"]
    names += ["SP_A_PRIVATE4"]
    return names


def transform_block(XA_raw: np.ndarray, XG_raw: np.ndarray, fit: dict, shared_rank: int):
    A = (XA_raw - fit["A_mean"]) / fit["A_sd"]
    G = (XG_raw - fit["G_mean"]) / fit["G_sd"]
    U = A @ fit["WA"]
    V = G @ fit["WG"]
    rho = fit["rho"]

    out = {}
    for j in range(3):
        k = j + 1
        if j < shared_rank:
            # Unit-variance common and discordance coordinates under discovery rho.
            out[f"SP_SHARED{k}"] = (U[:, j] + V[:, j]) / np.sqrt(2 * (1 + rho[j]))
            out[f"SP_DISCORD{k}"] = (U[:, j] - V[:, j]) / np.sqrt(2 * (1 - rho[j]))
        else:
            out[f"SP_A_PRIVATE{k}"] = U[:, j]
            out[f"SP_G_PRIVATE{k}"] = V[:, j]
    out["SP_A_PRIVATE4"] = U[:, 3]

    z = pd.DataFrame(out)
    # Keep the canonical coordinates too: useful for explicit multiplicative A x G tests.
    for j in range(4):
        z[f"CCA_A{j+1}"] = U[:, j]
    for j in range(3):
        z[f"CCA_G{j+1}"] = V[:, j]
    return z, U, V


def inverse_block(z: pd.DataFrame, fit: dict, shared_rank: int):
    n = len(z)
    U = np.zeros((n, 4))
    V = np.zeros((n, 3))
    rho = fit["rho"]

    for j in range(3):
        k = j + 1
        if j < shared_rank:
            s = z[f"SP_SHARED{k}"].to_numpy(float)
            d = z[f"SP_DISCORD{k}"].to_numpy(float)
            ap = np.sqrt(2 * (1 + rho[j])) * s
            am = np.sqrt(2 * (1 - rho[j])) * d
            U[:, j] = (ap + am) / 2
            V[:, j] = (ap - am) / 2
        else:
            U[:, j] = z[f"SP_A_PRIVATE{k}"].to_numpy(float)
            V[:, j] = z[f"SP_G_PRIVATE{k}"].to_numpy(float)
    U[:, 3] = z["SP_A_PRIVATE4"].to_numpy(float)

    A_std_hat = U @ np.linalg.inv(fit["WA"])
    G_std_hat = V @ np.linalg.inv(fit["WG"])
    A_hat = A_std_hat * fit["A_sd"] + fit["A_mean"]
    G_hat = G_std_hat * fit["G_sd"] + fit["G_mean"]
    return A_hat, G_hat


def reconstruction_stats(X: np.ndarray, Xhat: np.ndarray, w: np.ndarray, block: str, period: str):
    wn = wnorm(w)
    rows = []
    for j in range(X.shape[1]):
        mu = np.sum(wn * X[:, j])
        sse = np.sum(wn * (X[:, j] - Xhat[:, j]) ** 2)
        sst = np.sum(wn * (X[:, j] - mu) ** 2)
        rows.append({
            "period": period,
            "block": block,
            "variable_index": j + 1,
            "rmse": float(np.sqrt(np.sum(wn * (X[:, j] - Xhat[:, j]) ** 2))),
            "r2": float(1 - sse / sst) if sst > 0 else np.nan,
            "max_abs_error": float(np.max(np.abs(X[:, j] - Xhat[:, j]))),
        })
    return rows


def r_script_text(shared_terms, discord_terms, a_private_terms, g_private_terms):
    # Terms are injected from our own deterministic names only.
    return f'''args <- commandArgs(trailingOnly=TRUE)
input_file <- args[1]
out_dir <- args[2]
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu="adjust")

d <- read.csv(input_file, stringsAsFactors=FALSE)
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])

SHARED <- c({','.join(json.dumps(x) for x in shared_terms)})
DISCORD <- c({','.join(json.dumps(x) for x in discord_terms)})
A_PRIVATE <- c({','.join(json.dumps(x) for x in a_private_terms)})
G_PRIVATE <- c({','.join(json.dumps(x) for x in g_private_terms)})
PHYS <- c(SHARED, DISCORD, A_PRIVATE, G_PRIVATE)
A_CANON <- c("CCA_A1","CCA_A2","CCA_A3","CCA_A4")
G_CANON <- c("CCA_G1","CCA_G2","CCA_G3")
INTERACTIONS <- as.vector(outer(A_CANON, G_CANON, function(a,g) paste0(a,":",g)))
X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")

make_formula <- function(outcome, terms) as.formula(paste(outcome, "~", paste(terms, collapse=" + ")))

safe_test <- function(fit, terms, period, outcome, label) {{
  if (length(terms)==0) return(NULL)
  z <- tryCatch(regTermTest(fit, as.formula(paste("~", paste(terms, collapse=" + "))), method="Wald"), error=function(e) NULL)
  if (is.null(z)) return(data.frame(period=period,outcome=outcome,test=label,F=NA,df_num=NA,df_den=NA,p=NA,inference_status="unavailable"))
  ddf <- as.numeric(z$ddf)
  status <- ifelse(is.na(ddf),"unavailable",ifelse(ddf<=3,"design_limited","available"))
  data.frame(period=period,outcome=outcome,test=label,F=as.numeric(z$Ftest),df_num=as.numeric(z$df),df_den=ddf,p=as.numeric(z$p),inference_status=status)
}}

extract_coef <- function(fit, period, outcome, model, terms) {{
  sm <- summary(fit)$coefficients
  ci <- suppressMessages(confint(fit))
  pcol <- grep("^Pr", colnames(sm), value=TRUE)[1]
  out <- data.frame()
  for (term in terms) {{
    if (!(term %in% rownames(sm))) next
    out <- rbind(out, data.frame(period=period,outcome=outcome,model=model,term=term,beta=sm[term,"Estimate"],se=sm[term,"Std. Error"],ci_low=ci[term,1],ci_high=ci[term,2],p=sm[term,pcol]))
  }}
  out
}}

weighted_r2 <- function(fit, dat, outcome) {{
  pred <- as.numeric(predict(fit, newdata=dat, type="response"))
  y <- dat[[outcome]]; w <- dat$SURVEY_WT
  ok <- is.finite(y) & is.finite(w) & is.finite(pred) & w>0
  y <- y[ok]; w <- w[ok]; pred <- pred[ok]
  mu <- sum(w*y)/sum(w); sst <- sum(w*(y-mu)^2); sse <- sum(w*(y-pred)^2)
  ifelse(sst>0, 1-sse/sst, NA)
}}

all_tests <- data.frame(); all_coef <- data.frame(); all_r2 <- data.frame(); audit <- data.frame()
periods <- unique(d$PERIOD)
outcomes <- c(SOMATIC="SOMATIC_SCORE", COGAFF="COGAFF_SUM", PHQ9_TOTAL="PHQ9_TOTAL")

for (period in periods) {{
  dp <- d[d$PERIOD==period,]
  if (nrow(dp)==0) next
  covars <- X3
  if (length(unique(dp$CYCLE))>1) covars <- c(covars,"CYCLE")

  for (oname in names(outcomes)) {{
    outcome <- outcomes[[oname]]
    need <- unique(c(outcome, PHYS, A_CANON, G_CANON, covars, "SURVEY_WT","STRATUM","PSU"))
    ok <- dp$ELIGIBLE_ADULT_NONPREG==1 &
      complete.cases(dp[,need]) &
      is.finite(dp$SURVEY_WT) &
      dp$SURVEY_WT>0
    dp$MODEL_OK <- ok
    des_full <- svydesign(ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT, nest=TRUE, data=dp)
    des <- subset(des_full, MODEL_OK)
    dm <- dp[ok,]
    if (nrow(dm)<100) next

    fit0 <- svyglm(make_formula(outcome,covars), design=des, family=gaussian())
    fitmain <- svyglm(make_formula(outcome,c(covars,PHYS)), design=des, family=gaussian())
    fitint <- svyglm(make_formula(outcome,c(covars,PHYS,INTERACTIONS)), design=des, family=gaussian())

    all_tests <- rbind(all_tests,
      safe_test(fitmain, SHARED, period,oname,"shared_joint_physiology_given_private_discord_covars"),
      safe_test(fitmain, DISCORD, period,oname,"AG_discordance_given_shared_private_covars"),
      safe_test(fitmain, A_PRIVATE, period,oname,"A_private_given_shared_Gprivate_discord_covars"),
      safe_test(fitmain, G_PRIVATE, period,oname,"G_private_given_shared_Aprivate_discord_covars"),
      safe_test(fitmain, PHYS, period,oname,"all_shared_private_physiology_given_covars"),
      safe_test(fitint, INTERACTIONS, period,oname,"multiplicative_AxG_beyond_shared_private_main_effects")
    )

    all_coef <- rbind(all_coef, extract_coef(fitmain,period,oname,"shared_private_main",PHYS))
    all_r2 <- rbind(all_r2,
      data.frame(period=period,outcome=oname,model="covariates_only",weighted_R2=weighted_r2(fit0,dm,outcome)),
      data.frame(period=period,outcome=oname,model="shared_private_main",weighted_R2=weighted_r2(fitmain,dm,outcome)),
      data.frame(period=period,outcome=oname,model="shared_private_plus_multiplicative",weighted_R2=weighted_r2(fitint,dm,outcome))
    )
    audit <- rbind(audit,data.frame(period=period,outcome=oname,n=nrow(dm),full_design_df=degf(des_full),domain_df=degf(des)))
  }}
}}

write.csv(all_tests,file.path(out_dir,"72_shared_private_block_tests.csv"),row.names=FALSE)
write.csv(all_coef,file.path(out_dir,"72_shared_private_coefficients.csv"),row.names=FALSE)
write.csv(all_r2,file.path(out_dir,"72_shared_private_model_R2.csv"),row.names=FALSE)
write.csv(audit,file.path(out_dir,"72_shared_private_design_audit.csv"),row.names=FALSE)
'''


df = pd.read_csv(INPUT)
missing = [c for c in A_VARS + G_VARS + DESIGN + list(OUTCOMES.values()) + COVARS if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# Fit only on discovery physiology. No phenotype or covariate enters the decomposition.
disc = df[df["PERIOD"].astype(str) == "2005-2008"].copy()
fitmask = (
    (disc["ELIGIBLE_ADULT_NONPREG"] == 1)
    & disc[A_VARS + G_VARS + ["SURVEY_WT"]].notna().all(axis=1)
    & (disc["SURVEY_WT"] > 1e-12)
)
fitd = disc.loc[fitmask].copy()
if len(fitd) < 500:
    raise RuntimeError(f"Too few discovery physiology-complete rows: {len(fitd)}")

fit = fit_weighted_cca(fitd[A_VARS].to_numpy(float), fitd[G_VARS].to_numpy(float), fitd["SURVEY_WT"].to_numpy(float))
p_perm, q95 = permutation_pvalues(fit["A_std"], fit["G_std"], fitd["SURVEY_WT"].to_numpy(float), fit["rho"])

# Primary rank requires statistical evidence AND a nontrivial correlation magnitude.
keep = (p_perm < 0.05) & (fit["rho"] >= RHO_EFFECT_FLOOR)
shared_rank = int(np.sum(keep))
# Canonical correlations are ordered; enforce a leading block.
if shared_rank:
    shared_rank = int(np.max(np.where(keep)[0]) + 1)

shared_terms = [f"SP_SHARED{k}" for k in range(1, shared_rank + 1)]
discord_terms = [f"SP_DISCORD{k}" for k in range(1, shared_rank + 1)]
a_private_terms = [f"SP_A_PRIVATE{k}" for k in range(shared_rank + 1, 4)] + ["SP_A_PRIVATE4"]
g_private_terms = [f"SP_G_PRIVATE{k}" for k in range(shared_rank + 1, 4)]

# Transform every period using frozen discovery parameters.
frames = []
recon_rows = []
corr_rows = []
for period in ["2005-2008", "2009-2018", "2021-2023"]:
    dp = df[df["PERIOD"].astype(str) == period].copy()
    ok = (
    (dp["ELIGIBLE_ADULT_NONPREG"] == 1)
    & dp[A_VARS + G_VARS + ["SURVEY_WT"]].notna().all(axis=1)
    & (dp["SURVEY_WT"] > 1e-12)
    )
    idx = dp.index[ok]
    if len(idx) == 0:
        continue
    XA = dp.loc[idx, A_VARS].to_numpy(float)
    XG = dp.loc[idx, G_VARS].to_numpy(float)
    w = dp.loc[idx, "SURVEY_WT"].to_numpy(float)
    z, U, V = transform_block(XA, XG, fit, shared_rank)
    z.index = idx
    for c in z.columns:
        dp.loc[idx, c] = z[c]
    frames.append(dp)

    Ahat, Ghat = inverse_block(z, fit, shared_rank)
    recon_rows += reconstruction_stats(XA, Ahat, w, "A", period)
    recon_rows += reconstruction_stats(XG, Ghat, w, "G", period)
    for j in range(3):
        corr_rows.append({"period": period, "pair": j+1, "frozen_canonical_corr": weighted_corr(U[:,j], V[:,j], w)})

model_df = pd.concat(frames, axis=0).sort_index()
model_path = RESULTS / "72_shared_private_model_input.csv"
model_df.to_csv(model_path, index=False)

cca_summary = pd.DataFrame({
    "pair": [1,2,3],
    "discovery_canonical_corr": fit["rho"],
    "permutation_p": p_perm,
    "permutation_null_95": q95,
    "selected_as_shared_primary": [i < shared_rank for i in range(3)],
})
cca_summary.to_csv(RESULTS / "72_shared_private_cca_summary.csv", index=False)
pd.DataFrame(recon_rows).to_csv(RESULTS / "72_shared_private_roundtrip.csv", index=False)
pd.DataFrame(corr_rows).to_csv(RESULTS / "72_shared_private_temporal_coupling.csv", index=False)

# Save frozen transformation parameters for exact reproducibility.
np.savez(
    RESULTS / "72_shared_private_frozen_transform.npz",
    A_mean=fit["A_mean"], A_sd=fit["A_sd"], G_mean=fit["G_mean"], G_sd=fit["G_sd"],
    WA=fit["WA"], WG=fit["WG"], rho=fit["rho"], shared_rank=np.array([shared_rank], dtype=int),
)

rpath = RESULTS / "72_shared_private_survey_models.R"
rpath.write_text(r_script_text(shared_terms, discord_terms, a_private_terms, g_private_terms), encoding="utf-8")

r_status = "not_run"
r_error = ""
rscript = shutil.which("Rscript")
if rscript:
    proc = subprocess.run([rscript, str(rpath), str(model_path), str(RESULTS)], capture_output=True, text=True)
    r_status = "pass" if proc.returncode == 0 else "fail"
    r_error = proc.stderr[-3000:] if proc.returncode else ""
else:
    r_error = "Rscript not found; Python decomposition completed, survey regression not executed."

manifest = {
    "script": "72_shared_private_decomposition_and_phenotype.py",
    "scientific_question": "Is apparent A-G interaction better represented as outcome-independent shared physiology, private physiology, discordance, or a residual multiplicative phenotype interaction?",
    "decomposition": "weighted CCA canonical basis learned on 2005-08 physiology only; selected correlated pairs reparameterized into common(shared) and difference(discordance) coordinates; weak/unpaired directions retained as private coordinates",
    "primary_shared_rank_rule": f"permutation p<0.05 and discovery canonical correlation >= {RHO_EFFECT_FLOOR}",
    "shared_rank": shared_rank,
    "discovery_n": int(len(fitd)),
    "A_variables": A_VARS,
    "G_variables": G_VARS,
    "outcome_used_to_fit_decomposition": False,
    "future_refit": False,
    "transform_is_full_rank": True,
    "interpretation_boundary": "shared coordinates represent statistically shared A-G physiological variation, not a proven biological mechanism or causal source",
    "phenotype_model": "survey-weighted Y ~ shared + discordance + private A + private G + covariates; separate block test adds all CCA_A x CCA_G multiplicative interactions",
    "r_survey_status": r_status,
    "r_error": r_error,
    "locked_outputs_modified": False,
}
(RESULTS / "72_shared_private_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("PASS Script 72 shared/private physiology decomposition complete.")
print(f"PASS Discovery physiology-only n={len(fitd)}; future refit=False.")
print("Canonical correlations:", ", ".join(f"{x:.4f}" for x in fit["rho"]))
print("Permutation p-values:", ", ".join(f"{x:.4g}" for x in p_perm))
print(f"Primary shared rank = {shared_rank} (effect floor rho >= {RHO_EFFECT_FLOOR}).")
print("Shared terms:", shared_terms if shared_terms else "none")
print("Discordance terms:", discord_terms if discord_terms else "none")
print("A-private terms:", a_private_terms if a_private_terms else "none")
print("G-private terms:", g_private_terms if g_private_terms else "none")
print(f"Survey regression status: {r_status}")
if r_status != "pass":
    print("NOTE", r_error)
print("\nKey interpretation:")
print("  SHARED block = candidate statistical A-G mixture before phenotype modeling.")
print("  DISCORD block = A-vs-G mismatch along those same coupled axes.")
print("  multiplicative block = asks whether classic A×G interactions add anything AFTER shared/private structure is represented.")
