#!/usr/bin/env python
"""
51_lock_in_falsification_suite.py
================================

Purpose
-------
A pre-lock "try to break it" audit for the NHANES physiological-decomposition
pipeline.

Recommended final run:
    python NHANES_Scripts/51_lock_in_falsification_suite.py --rerun-core

What this DOES:
- optionally reruns the frozen core pipeline (46-50);
- audits outcome leakage, frozen transforms, structural stability/reconstruction;
- verifies the final domain-corrected component-vs-scalar result;
- checks temporal replication;
- checks raw-marker, quasi-Poisson, no-insulin, no-HbA1c, OGTT and interaction sensitivities;
- runs a NEW same-sample X0->X3 adjustment-ladder stress test;
- runs a NEW arbitrary cycle-specific temporal-heterogeneity stress test;
- audits complete-case selection imbalance on observed variables;
- checks multicollinearity among frozen components;
- records multiplicity caveats;
- writes a cryptographic lock manifest.

What this CANNOT prove:
No finite script can prove absence of every possible bias. It cannot eliminate
unmeasured confounding, MNAR missingness, reverse causation, unknown measurement
error, or transportability problems outside the observed data.

Outputs:
    Results/51_lock_in_findings.csv
    Results/51_lock_in_adjustment_ladder.csv
    Results/51_lock_in_nonlinear_time.csv
    Results/51_lock_in_selection_smd.csv
    Results/51_lock_in_report.md
    Audit/51_lock_manifest.json

Exit codes:
    0 = core lock passed (warnings may remain)
    2 = at least one critical lock criterion failed
    3 = preflight/execution failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ALPHA = 0.05


def find_repo_root() -> Path:
    candidates = []
    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if (p / "NHANES_Scripts").is_dir() and (p / "Results").is_dir():
            return p
    raise RuntimeError(
        "Could not locate repository root. Place this script under NHANES_Scripts/ "
        "or run it from the repository root."
    )


ROOT = find_repo_root()
SCRIPTS = ROOT / "NHANES_Scripts"
RESULTS = ROOT / "Results"
AUDIT = ROOT / "Audit"
PARAMETERS = ROOT / "Parameters"
AUDIT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class Finding:
    category: str
    test: str
    status: str
    critical: bool
    evidence: str
    interpretation: str


findings: list[Finding] = []


def add(category: str, test: str, status: str, evidence: str,
        interpretation: str, critical: bool = False):
    findings.append(Finding(
        category=category,
        test=test,
        status=status,
        critical=critical,
        evidence=evidence,
        interpretation=interpretation,
    ))


def require_file(path: Path):
    if not path.exists():
        add("preflight", str(path.relative_to(ROOT)), "FAIL",
            "Required file missing.", "Run prerequisite scripts first.", True)
        return False
    return True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def one(df: pd.DataFrame, **conds) -> pd.Series:
    q = df.copy()
    for k, v in conds.items():
        if k not in q.columns:
            raise KeyError(f"Column {k!r} missing")
        if isinstance(v, (list, tuple, set)):
            q = q[q[k].isin(v)]
        else:
            q = q[q[k].astype(str) == str(v)]
    if len(q) != 1:
        raise RuntimeError(f"Expected 1 row for {conds}, found {len(q)}")
    return q.iloc[0]


def finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def pcheck(p, expected: str, label: str, category: str,
           critical: bool = True, alpha: float = ALPHA):
    p = float(p)
    if expected == "lt":
        ok = p < alpha
        meaning = f"p={p:.6g} < {alpha}"
    elif expected == "ge":
        ok = p >= alpha
        meaning = f"p={p:.6g} >= {alpha}"
    else:
        raise ValueError(expected)
    add(category, label, "PASS" if ok else ("FAIL" if critical else "WARN"),
        meaning,
        "Criterion met." if ok else "Criterion not met; do not lock this claim without review.",
        critical)


CORE_SCRIPTS = [
    "46_refreeze_outcome_independent_representations.py",
    "47_domain_corrected_combined_AG_inference.py",
    "48_comparator_and_sensitivity_audit.py",
    "49_interaction_and_temporal_drift_audit.py",
    "50_g4_extension_and_modifier_audit.py",
]


def rerun_core():
    print("\n=== RERUNNING CORE 46 -> 50 ===")
    for name in CORE_SCRIPTS:
        path = SCRIPTS / name
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n>>> {name}")
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        if proc.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
        return commit, branch, dirty
    except Exception:
        return None, None, None


def find_rscript() -> Optional[str]:
    r = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if r:
        return r
    candidates = sorted(Path(r"C:\Program Files\R").glob(r"R-*\bin\Rscript.exe"))
    return str(candidates[-1]) if candidates else None


def run_new_r_stress_tests() -> tuple[Optional[Path], Optional[Path]]:
    rscript = find_rscript()
    if not rscript:
        add("preflight", "Rscript", "FAIL", "Rscript not found.",
            "The new lock-in stress tests require R + survey.", True)
        return None, None

    input47 = RESULTS / "47_domain_corrected_AG_input.csv"
    input49 = RESULTS / "49_interaction_temporal_input.csv"
    if not (require_file(input47) and require_file(input49)):
        return None, None

    adj_out = RESULTS / "51_lock_in_adjustment_ladder.csv"
    time_out = RESULTS / "51_lock_in_nonlinear_time.csv"
    rfile = AUDIT / "51_lock_in_stress_tests.R"

    r_code = r"""
args <- commandArgs(trailingOnly=TRUE)
input47 <- args[1]
input49 <- args[2]
adj_out <- args[3]
time_out <- args[4]

if (!requireNamespace("survey", quietly=TRUE)) {
  stop("R package 'survey' is required")
}
library(survey)
options(survey.lonely.psu="adjust")

safe_test <- function(fit, terms) {
  z <- tryCatch(
    regTermTest(fit, as.formula(paste("~", paste(terms, collapse=" + "))), method="Wald"),
    error=function(e) NULL
  )
  if (is.null(z)) {
    return(c(F=NA_real_, df_num=NA_real_, df_den=NA_real_, p=NA_real_))
  }
  c(F=as.numeric(z$Ftest), df_num=as.numeric(z$df),
    df_den=as.numeric(z$ddf), p=as.numeric(z$p))
}

make_formula <- function(outcome, terms) {
  as.formula(paste(outcome, "~", paste(unique(terms), collapse=" + ")))
}

wr2 <- function(fit, d, outcome) {
  y <- d[[outcome]]
  pred <- as.numeric(predict(fit, newdata=d, type="response"))
  w <- d$SURVEY_WT
  mu <- sum(w*y)/sum(w)
  sse <- sum(w*(y-pred)^2)
  sst <- sum(w*(y-mu)^2)
  if (sst <= 0) return(NA_real_)
  1 - sse/sst
}

# A) SAME-SAMPLE ADJUSTMENT LADDER
normalize_cycle <- function(x) {
  z <- gsub("\\.0$", "", as.character(x))
  digits <- gsub("[^0-9]", "", z)
  out <- suppressWarnings(sprintf("%04d", as.integer(digits)))
  out[is.na(out)] <- z[is.na(out)]
  out
}

d <- read.csv(input47, stringsAsFactors=FALSE)
d$CYCLE <- normalize_cycle(d$CYCLE)
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) d[[v]] <- factor(d[[v]])

A <- c("A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ")
G <- c("G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ")
PCS <- c(A,G)

sets <- list(
  X0=c("RIDAGEYR","RIAGENDR","RACE"),
  X1=c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3"),
  X2=c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI"),
  X3=c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")
)

rows <- data.frame()

for (pop in c("2005-2008","2009-2018")) {
  dat <- d[d$PERIOD==pop & d$SURVEY_WT>0,]
  des_full <- svydesign(ids=~PSU, strata=~STRATUM, weights=~SURVEY_WT,
                        nest=TRUE, data=dat)
  des <- subset(des_full, DOMAIN_SOMATIC_SCORE_X3==1)
  dom <- dat[dat$DOMAIN_SOMATIC_SCORE_X3==1,]

  for (nm in names(sets)) {
    covs <- c(sets[[nm]], "CYCLE")

    fit_scalar <- svyglm(
      make_formula("SOMATIC_SCORE", c("A","G_HBA1C",covs)),
      design=des, family=gaussian()
    )
    fit_pcs <- svyglm(
      make_formula("SOMATIC_SCORE", c(PCS,covs)),
      design=des, family=gaussian()
    )
    fit_union <- svyglm(
      make_formula("SOMATIC_SCORE", c("A","G_HBA1C",PCS,covs)),
      design=des, family=gaussian()
    )

    t_pc <- safe_test(fit_union, PCS)
    t_sc <- safe_test(fit_union, c("A","G_HBA1C"))

    r2s <- wr2(fit_scalar, dom, "SOMATIC_SCORE")
    r2p <- wr2(fit_pcs, dom, "SOMATIC_SCORE")
    r2u <- wr2(fit_union, dom, "SOMATIC_SCORE")

    rows <- rbind(rows, data.frame(
      population=pop, adjustment=nm, n=nrow(dom),
      design_df=degf(des),
      p_PCs_beyond_scalars=t_pc["p"],
      F_PCs_beyond_scalars=t_pc["F"],
      df_num_PCs=t_pc["df_num"], df_den_PCs=t_pc["df_den"],
      p_scalars_beyond_PCs=t_sc["p"],
      R2_scalar=r2s, R2_PCs=r2p, R2_union=r2u,
      delta_PCs_beyond_scalars=r2u-r2s,
      delta_scalars_beyond_PCs=r2u-r2p
    ))
  }
}

write.csv(rows, adj_out, row.names=FALSE)

# B) NONLINEAR / ARBITRARY CYCLE-SPECIFIC SLOPE HETEROGENEITY
t <- read.csv(input49, stringsAsFactors=FALSE)
t$CYCLE <- normalize_cycle(t$CYCLE)
t <- t[t$CYCLE %in% c("0506","0708","0910","1112","1314","1516","1718") &
       !is.na(t$WT_0518) & t$WT_0518>0,]
for (v in c("RIAGENDR","RACE","EDUC3","SMOKING3","CYCLE")) t[[v]] <- factor(t[[v]])

des_full_t <- svydesign(ids=~PSU, strata=~STRATUM, weights=~WT_0518,
                        nest=TRUE, data=t)
des_t <- subset(des_full_t, DOMAIN_SOMATIC_SCORE_X3==1)

X3 <- c("RIDAGEYR","RIAGENDR","RACE","INDFMPIR","EDUC3","SMOKING3","BMXBMI","EGFR_2021")

fit_A <- svyglm(
  make_formula("SOMATIC_SCORE",
    c(PCS,X3,"CYCLE",
      "A_OI_PC1_FZ:CYCLE","A_OI_PC2_FZ:CYCLE","A_OI_PC3_FZ:CYCLE")),
  design=des_t, family=gaussian()
)

fit_G <- svyglm(
  make_formula("SOMATIC_SCORE",
    c(PCS,X3,"CYCLE",
      "G3_OI_PC1_FZ:CYCLE","G3_OI_PC2_FZ:CYCLE","G3_OI_PC3_FZ:CYCLE")),
  design=des_t, family=gaussian()
)

ta <- safe_test(fit_A, c("A_OI_PC1_FZ:CYCLE","A_OI_PC2_FZ:CYCLE","A_OI_PC3_FZ:CYCLE"))
tg <- safe_test(fit_G, c("G3_OI_PC1_FZ:CYCLE","G3_OI_PC2_FZ:CYCLE","G3_OI_PC3_FZ:CYCLE"))

out <- rbind(
  data.frame(block="A", F=ta["F"], df_num=ta["df_num"], df_den=ta["df_den"],
             p=ta["p"], design_df=degf(des_t)),
  data.frame(block="G3", F=tg["F"], df_num=tg["df_num"], df_den=tg["df_den"],
             p=tg["p"], design_df=degf(des_t))
)
write.csv(out, time_out, row.names=FALSE)
"""

    rfile.write_text(r_code, encoding="utf-8")
    proc = subprocess.run(
        [str(rscript), str(rfile), str(input47), str(input49),
         str(adj_out), str(time_out)],
        cwd=str(ROOT), capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-8000:])
        add("new_stress_tests", "R stress-test execution", "FAIL",
            f"R returned exit code {proc.returncode}.",
            "Inspect Audit/51_lock_in_stress_tests.R and R output.", True)
        return None, None

    add("new_stress_tests", "R stress-test execution", "PASS",
        "Same-sample adjustment ladder and arbitrary cycle-heterogeneity tests completed.",
        "New falsification outputs were generated.", True)
    return adj_out, time_out


def weighted_mean_var(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) < 2 or w.sum() <= 0:
        return np.nan, np.nan, int(len(x))
    m = np.sum(w * x) / np.sum(w)
    v = np.sum(w * (x - m) ** 2) / np.sum(w)
    return float(m), float(v), int(len(x))


def selection_audit(input_path: Path) -> Optional[pd.DataFrame]:
    usecols = [
        "PERIOD","SURVEY_WT","ELIGIBLE_ADULT_NONPREG","DOMAIN_SOMATIC_SCORE_X3",
        "RIDAGEYR","INDFMPIR","BMXBMI","EGFR_2021","A","G_HBA1C"
    ]
    try:
        d = pd.read_csv(input_path, usecols=usecols)
    except Exception as e:
        add("selection", "Observed-variable selection audit", "WARN",
            f"Could not compute SMD audit: {e}",
            "Selection remains an explicit limitation.", False)
        return None

    rows = []
    max_smd = 0.0
    any_smd = False

    for pop in ["2005-2008","2009-2018","2021-2023"]:
        q = d[(d["PERIOD"].astype(str) == pop) &
              (d["ELIGIBLE_ADULT_NONPREG"] == 1)].copy()
        inc = q[q["DOMAIN_SOMATIC_SCORE_X3"] == 1]
        exc = q[q["DOMAIN_SOMATIC_SCORE_X3"] == 0]

        sw_all = q["SURVEY_WT"].where(q["SURVEY_WT"] > 0).sum()
        sw_inc = inc["SURVEY_WT"].where(inc["SURVEY_WT"] > 0).sum()
        weighted_retention = sw_inc / sw_all if sw_all > 0 else np.nan
        unweighted_retention = len(inc) / len(q) if len(q) else np.nan

        add("selection", f"{pop} analytic retention",
            "PASS" if (finite(weighted_retention) and weighted_retention >= 0.70) else "WARN",
            f"eligible n={len(q):,}; included n={len(inc):,}; "
            f"unweighted retention={unweighted_retention:.3f}; weighted retention={weighted_retention:.3f}",
            "Lower retention increases sensitivity to complete-case selection. "
            "This is diagnostic, not proof of MAR.", False)

        for var in ["RIDAGEYR","INDFMPIR","BMXBMI","EGFR_2021","A","G_HBA1C"]:
            m1, v1, n1 = weighted_mean_var(inc[var], inc["SURVEY_WT"])
            m0, v0, n0 = weighted_mean_var(exc[var], exc["SURVEY_WT"])
            den = math.sqrt((v1 + v0) / 2) if finite(v1) and finite(v0) and (v1 + v0) > 0 else np.nan
            smd = (m1 - m0) / den if finite(den) and den > 0 else np.nan
            if finite(smd):
                max_smd = max(max_smd, abs(float(smd)))
                any_smd = True
            rows.append({
                "population": pop, "variable": var,
                "included_n_observed": n1, "excluded_n_observed": n0,
                "included_weighted_mean": m1, "excluded_weighted_mean": m0,
                "weighted_SMD": smd,
            })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "51_lock_in_selection_smd.csv", index=False)

    if any_smd:
        add("selection", "Maximum observed-variable selection SMD",
            "PASS" if max_smd < 0.10 else "WARN",
            f"max |weighted SMD| = {max_smd:.3f}",
            "SMD <0.10 is reassuring. Larger imbalance does not prove bias, "
            "but means complete-case selection must remain a stated limitation.", False)

    add("selection", "MNAR / unobserved selection", "UNTESTABLE",
        "Observed data cannot establish that missingness is independent of unobserved physiology/phenotype.",
        "Berkson/collider-type selection from unobserved determinants cannot be ruled out by this dataset alone.", False)
    return out


def component_vif_audit(input_path: Path):
    cols = [
        "PERIOD","DOMAIN_SOMATIC_SCORE_X3",
        "A_OI_PC1_FZ","A_OI_PC2_FZ","A_OI_PC3_FZ",
        "G3_OI_PC1_FZ","G3_OI_PC2_FZ","G3_OI_PC3_FZ",
    ]
    try:
        d = pd.read_csv(input_path, usecols=cols)
    except Exception as e:
        add("model_specification", "Component VIF", "WARN", str(e),
            "Could not compute component-only VIF.", False)
        return

    pc = cols[2:]
    for pop in ["2005-2008","2009-2018"]:
        q = d[(d["PERIOD"].astype(str) == pop) &
              (d["DOMAIN_SOMATIC_SCORE_X3"] == 1)][pc].dropna()
        X = q.to_numpy(float)
        if len(X) < 20:
            continue
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        vifs = []
        for j in range(X.shape[1]):
            y = X[:, j]
            others = np.delete(X, j, axis=1)
            A = np.column_stack([np.ones(len(others)), others])
            b, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ b
            sst = np.sum((y - y.mean())**2)
            sse = np.sum((y - pred)**2)
            r2 = 1 - sse/sst if sst > 0 else np.nan
            vif = 1/(1-r2) if finite(r2) and r2 < 1 else np.inf
            vifs.append(vif)
        mv = float(np.nanmax(vifs))
        add("model_specification", f"{pop} component-only VIF",
            "PASS" if mv < 5 else "WARN",
            f"max VIF among six frozen PCs = {mv:.3f}",
            "VIF <5 argues against severe redundancy among component predictors.", False)


def audit_existing(alpha: float):
    required = [
        SCRIPTS / "46_refreeze_outcome_independent_representations.py",
        RESULTS / "46_outcome_independent_structure_audit.csv",
        RESULTS / "46_outcome_independent_A_reconstruction_transfer.csv",
        RESULTS / "47_domain_corrected_block_tests.csv",
        RESULTS / "47_domain_corrected_incremental_information.csv",
        RESULTS / "47_domain_corrected_design_audit.csv",
        RESULTS / "48_comparator_sensitivity_tests.csv",
        RESULTS / "48_quasipoisson_somatic_tests.csv",
        RESULTS / "48_representation_comparator_R2.csv",
        RESULTS / "49_interaction_block_tests.csv",
        RESULTS / "49_temporal_drift_tests.csv",
        RESULTS / "49_temporal_drift_coefficients.csv",
        RESULTS / "50_g4_modifier_tests.csv",
        RESULTS / "50_g4_same_sample_R2.csv",
    ]
    if not all(require_file(p) for p in required):
        return

    s46 = (SCRIPTS / "46_refreeze_outcome_independent_representations.py").read_text(
        encoding="utf-8", errors="replace"
    )
    forbidden_operational_tokens = ["DPQ_", "PHQ9_TOTAL", "SOMATIC_SCORE"]
    leaked = [x for x in forbidden_operational_tokens if x in s46]
    add("representation", "Outcome leakage static audit",
        "PASS" if not leaked else "FAIL",
        "No operational depression/outcome tokens in Script 46."
        if not leaked else f"Found tokens: {leaked}",
        "The final physiological representation must be learned without phenotype conditioning.",
        True)

    audit46 = AUDIT / "46_outcome_independent_refreeze.json"
    if audit46.exists():
        a = json.loads(audit46.read_text(encoding="utf-8"))
        ok = a.get("reads_PHQ_or_depression") is False and a.get("A_retained_components") == 3
        add("representation", "Script 46 audit manifest",
            "PASS" if ok else "FAIL",
            f"reads_PHQ_or_depression={a.get('reads_PHQ_or_depression')}; "
            f"A_retained_components={a.get('A_retained_components')}",
            "Outcome independence and retained dimensionality are explicit.",
            True)

    st = pd.read_csv(RESULTS / "46_outcome_independent_structure_audit.csv")
    min_sim = float(st["cross_cycle_loading_similarity"].min())
    add("representation", "Cross-cycle loading similarity",
        "PASS" if min_sim >= 0.95 else "FAIL",
        f"minimum aligned loading similarity = {min_sim:.6f}",
        "Discovery-cycle PCA geometry should reproduce across 2005-06 and 2007-08.",
        True)

    rec = pd.read_csv(RESULTS / "46_outcome_independent_A_reconstruction_transfer.csv")
    min_rec = float(rec["reconstruction_energy_fraction"].min())
    add("representation", "Frozen A reconstruction",
        "PASS" if min_rec >= 0.98 else "FAIL",
        f"minimum reconstruction-energy fraction = {min_rec:.6f}",
        "Retained A representation must preserve >=98% of standardized hematology energy.",
        True)

    b47 = pd.read_csv(RESULTS / "47_domain_corrected_block_tests.csv")
    inc47 = pd.read_csv(RESULTS / "47_domain_corrected_incremental_information.csv")
    da47 = pd.read_csv(RESULTS / "47_domain_corrected_design_audit.csv")

    for pop in ["2005-2008","2009-2018"]:
        r = one(
            b47, population=pop, adjustment="X3", outcome="SOMATIC_SCORE",
            test="all_PCs_extra_beyond_scalar_A_scalar_G_X"
        )
        pcheck(r["p"], "lt", f"{pop}: components beyond scalars",
               "primary_replication", True, alpha)

        r2 = one(
            b47, population=pop, adjustment="X3", outcome="SOMATIC_SCORE",
            test="both_scalars_extra_beyond_all_PCs_X"
        )
        pcheck(r2["p"], "ge", f"{pop}: scalars beyond components",
               "primary_replication", False, alpha)

        di = one(inc47, population=pop, adjustment="X3", outcome="SOMATIC_SCORE")
        dpc = float(di["delta_all_PCs_beyond_scalar_model"])
        dsc = float(di["delta_scalars_beyond_all_PCs"])
        add("primary_replication", f"{pop}: incremental weighted R2 asymmetry",
            "PASS" if (dpc > 0 and dsc < dpc) else "FAIL",
            f"PCs beyond scalars ΔR²={dpc:.6f}; scalars beyond PCs ΔR²={dsc:.6f}",
            "Richer representation should add more information than the reverse direction.",
            True)

        aud = one(da47, population=pop, adjustment="X3", outcome="SOMATIC_SCORE")
        okdf = int(aud["full_design_df"]) == int(aud["domain_design_df"])
        add("survey_design", f"{pop}: design-before-domain df preservation",
            "PASS" if okdf else "WARN",
            f"full design df={int(aud['full_design_df'])}; domain df={int(aud['domain_design_df'])}",
            "Full survey design is created before analytic-domain subsetting.", False)

    for pop in ["2005-2008","2009-2018"]:
        r = one(
            b47, population=pop, adjustment="X3", outcome="PHQ9_TOTAL",
            test="all_PCs_extra_beyond_scalar_A_scalar_G_X"
        )
        pcheck(r["p"], "lt", f"{pop}: PHQ9 total components beyond scalars",
               "outcome_sensitivity", False, alpha)

    c = one(
        b47, population="2009-2018", adjustment="X3", outcome="COGAFF_SUM",
        test="all_PCs_extra_beyond_scalar_A_scalar_G_X"
    )
    add("outcome_sensitivity", "Cognitive-affective replication",
        "INFO", f"2009-2018 p={float(c['p']):.6g}",
        "Do not claim uniform phenotype effects; cognitive-affective-only evidence is weaker.", False)

    c48 = pd.read_csv(RESULTS / "48_comparator_sensitivity_tests.csv")
    for pop in ["2005-2008","2009-2018"]:
        r = one(c48, population=pop, analysis="comparator",
                test="raw7_extra_beyond_threshold_scalars")
        pcheck(r["p"], "lt", f"{pop}: raw7 beyond threshold scalars",
               "representation_sensitivity", True, alpha)
        rr = one(c48, population=pop, analysis="comparator",
                 test="threshold_scalars_extra_beyond_raw7")
        pcheck(rr["p"], "ge", f"{pop}: threshold scalars beyond raw7",
               "representation_sensitivity", False, alpha)

    comp_r2 = pd.read_csv(RESULTS / "48_representation_comparator_R2.csv")
    for pop in ["2005-2008","2009-2018"]:
        raw = one(comp_r2, population=pop, model="raw7_multivariate")["weighted_R2"]
        pca = one(comp_r2, population=pop, model="frozen_components")["weighted_R2"]
        diff = abs(float(raw) - float(pca))
        add("representation_sensitivity", f"{pop}: raw7 vs PCA R2 proximity",
            "PASS" if diff < 0.002 else "WARN",
            f"|R²_raw7 - R²_PCA| = {diff:.6f}",
            "Similar performance argues against a PCA-specific artifact.", False)

    qp = pd.read_csv(RESULTS / "48_quasipoisson_somatic_tests.csv")
    for pop in ["2005-2008","2009-2018"]:
        r = one(qp, population=pop, analysis="quasipoisson",
                test="all_PCs_extra_beyond_threshold_scalars")
        pcheck(r["p"], "lt", f"{pop}: quasi-Poisson components beyond scalars",
               "model_specification", True, alpha)

    for pop in ["2005-2008","2009-2018"]:
        r = one(c48, population=pop, analysis="G2_no_insulin",
                test="all_components_extra_beyond_threshold_scalars")
        pcheck(r["p"], "lt", f"{pop}: no-insulin sensitivity",
               "measurement_sensitivity", True, alpha)
        r = one(c48, population=pop, analysis="G2_no_HbA1c",
                test="all_components_extra_beyond_A_FPG_scalars")
        pcheck(r["p"], "lt", f"{pop}: no-HbA1c sensitivity",
               "measurement_sensitivity", True, alpha)

    i49 = pd.read_csv(RESULTS / "49_interaction_block_tests.csv")
    r = one(i49, population="2009-2018", family="A_x_G",
            test="all_9_AxG_interactions")
    pcheck(r["p"], "ge", "Replication: broad A×G interaction not supported",
           "interaction_nonclaim", False, alpha)

    td = pd.read_csv(RESULTS / "49_temporal_drift_tests.csv")
    arow = one(td, population="2005-2018", family="temporal_drift",
               test="A_components_linear_cycle_drift")
    grow = one(td, population="2005-2018", family="temporal_drift",
               test="G_components_linear_cycle_drift")
    allrow = one(td, population="2005-2018", family="temporal_drift",
                 test="all_components_linear_cycle_drift")
    pcheck(arow["p"], "ge", "A linear temporal drift", "temporal_mapping", False, alpha)
    pcheck(grow["p"], "lt", "G block linear temporal drift", "temporal_mapping", False, alpha)
    add("temporal_mapping", "All-component linear drift",
        "INFO", f"p={float(allrow['p']):.6g}",
        "Do not overstate a borderline joint result.", False)

    tcoef = pd.read_csv(RESULTS / "49_temporal_drift_coefficients.csv")
    qmin = float(tcoef["p_BH_within_model"].min())
    add("multiplicity", "Individual temporal interactions after BH",
        "PASS" if qmin >= alpha else "WARN",
        f"minimum BH-adjusted p = {qmin:.6g}",
        "Do not localize temporal drift to an individual PC unless adjusted evidence survives.", False)

    g4 = pd.read_csv(RESULTS / "50_g4_modifier_tests.csv")
    for pop in ["2005-2008","2009-2016"]:
        r = one(g4, population=pop, family="G4_extension",
                test="2h_OGTT_extra_beyond_full_G3")
        pcheck(r["p"], "ge", f"{pop}: 2h OGTT beyond G3",
               "complexity_control", False, alpha)

    g4r2 = pd.read_csv(RESULTS / "50_g4_same_sample_R2.csv")
    for pop in ["2005-2008","2009-2016"]:
        a = one(g4r2, population=pop, model="A_plus_G3")["weighted_R2"]
        b = one(g4r2, population=pop, model="A_plus_G4")["weighted_R2"]
        gain = float(b) - float(a)
        add("complexity_control", f"{pop}: G4 same-sample R2 gain",
            "PASS" if gain < 0.002 else "WARN",
            f"R²(G4)-R²(G3)={gain:.6f}",
            "Tiny gain supports keeping G3 as the primary parsimonious representation.", False)

    qvals = pd.to_numeric(g4["q_BH_modifier_family"], errors="coerce").dropna()
    if len(qvals):
        qmin = float(qvals.min())
        add("multiplicity", "G4 modifier family after BH",
            "PASS" if qmin >= alpha else "WARN",
            f"minimum modifier-family BH q={qmin:.6g}",
            "No modifier becomes a claim from one nominal p-value.", False)

    modern = b47[
        (b47["population"].astype(str) == "2021-2023") &
        (b47["adjustment"].astype(str) == "X3") &
        (b47["outcome"].astype(str) == "SOMATIC_SCORE")
    ]
    limited = len(modern) > 0 and (modern["inference_status"].astype(str) == "design_limited").all()
    add("survey_design", "2021-23 full X3 inference",
        "PASS" if limited else "WARN",
        "All primary 2021-23 X3 block tests are design-limited." if limited
        else "Unexpected modern-cycle inference status.",
        "Do not use 2021-23 as confirmatory significance evidence.", False)

    for name, evidence, interp in [
        ("Unmeasured confounding",
         "No observational adjustment set can prove that all common causes were measured.",
         "Association estimates remain vulnerable to residual/unmeasured confounding."),
        ("Reverse causation",
         "NHANES is cross-sectional within each cycle.",
         "Depressive phenotype may affect physiology/behavior as well as the reverse."),
        ("Collider/overadjustment",
         "BMI/eGFR may have different causal roles under different biological DAGs.",
         "Adjustment-ladder stability helps, but causal status cannot be identified from associations alone."),
        ("MNAR missingness",
         "Missingness mechanisms involving unobserved values are not identified.",
         "Complete-case selection can never be fully proven safe here."),
        ("External transportability",
         "2009-18 is temporal transfer within NHANES.",
         "Independent external-population validation remains required."),
        ("True longitudinal change",
         "NHANES cycles are repeated cross-sections.",
         "Do not call current analysis within-person longitudinal modelling."),
    ]:
        add("irreducible_limitations", name, "UNTESTABLE", evidence, interp, False)


def audit_new_stress_outputs(adj_path: Optional[Path], time_path: Optional[Path], alpha: float):
    if adj_path and adj_path.exists():
        a = pd.read_csv(adj_path)
        for _, r in a.iterrows():
            p = float(r["p_PCs_beyond_scalars"])
            dr = float(r["delta_PCs_beyond_scalars"])
            ok = p < alpha and dr > 0
            add("adjustment_ladder",
                f"{r['population']} {r['adjustment']}: PCs beyond scalars",
                "PASS" if ok else "FAIL",
                f"p={p:.6g}; ΔR²={dr:.6f}; same X3-complete sample n={int(r['n'])}",
                "Core component advantage should not depend on one adjustment set.",
                True)

        reverse_bad = a[
            (pd.to_numeric(a["p_scalars_beyond_PCs"], errors="coerce") < alpha) &
            (pd.to_numeric(a["delta_scalars_beyond_PCs"], errors="coerce") >
             pd.to_numeric(a["delta_PCs_beyond_scalars"], errors="coerce"))
        ]
        add("adjustment_ladder", "Reverse scalar advantage across adjustment ladder",
            "PASS" if reverse_bad.empty else "WARN",
            "No adjustment set shows a statistically supported reverse scalar advantage."
            if reverse_bad.empty else f"{len(reverse_bad)} adjustment rows favor the reverse direction.",
            "Guards against the headline result being created by one covariate specification.", False)

    if time_path and time_path.exists():
        t = pd.read_csv(time_path)
        for block in ["A","G3"]:
            r = one(t, block=block)
            p = float(r["p"]) if finite(r["p"]) else np.nan
            if not finite(p):
                add("temporal_mapping", f"{block}: arbitrary cycle-specific slope heterogeneity",
                    "WARN", "Test unavailable/design-limited.",
                    "Linear drift remains only one temporal specification.", False)
            elif block == "A":
                add("temporal_mapping", "A: arbitrary cycle-specific slope heterogeneity",
                    "PASS" if p >= alpha else "WARN",
                    f"p={p:.6g}",
                    "If significant, weaken the earlier 'stable A mapping' statement.", False)
            else:
                add("temporal_mapping", "G3: arbitrary cycle-specific slope heterogeneity",
                    "INFO", f"p={p:.6g}",
                    "Interpret alongside Script 49; do not promote this stress test into a new claim.", False)


def make_manifest(core_locked: bool, commit, branch, dirty):
    files_to_hash = []
    for name in CORE_SCRIPTS + ["51_lock_in_falsification_suite.py"]:
        p = SCRIPTS / name
        if p.exists():
            files_to_hash.append(p)

    result_names = [
        "46_outcome_independent_structure_audit.csv",
        "46_outcome_independent_A_reconstruction_transfer.csv",
        "47_domain_corrected_block_tests.csv",
        "47_domain_corrected_incremental_information.csv",
        "47_domain_corrected_design_audit.csv",
        "48_comparator_sensitivity_tests.csv",
        "48_quasipoisson_somatic_tests.csv",
        "48_representation_comparator_R2.csv",
        "49_interaction_block_tests.csv",
        "49_temporal_drift_tests.csv",
        "49_temporal_drift_coefficients.csv",
        "50_g4_modifier_tests.csv",
        "50_g4_same_sample_R2.csv",
        "51_lock_in_adjustment_ladder.csv",
        "51_lock_in_nonlinear_time.csv",
        "51_lock_in_selection_smd.csv",
        "51_lock_in_findings.csv",
    ]
    for name in result_names:
        p = RESULTS / name
        if p.exists():
            files_to_hash.append(p)

    manifest = {
        "core_locked": bool(core_locked),
        "git_commit": commit,
        "git_branch": branch,
        "working_tree_dirty": None if dirty is None else bool(dirty),
        "working_tree_status": dirty,
        "alpha": ALPHA,
        "files": {
            str(p.relative_to(ROOT)): {
                "sha256": sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in files_to_hash
        },
        "explicit_nonclaims": [
            "causality",
            "absence of unmeasured confounding",
            "absence of MNAR/selection bias",
            "biological independence of hematology and glycemia",
            "stable A×G interaction",
            "individual G2/G3 temporal drift after multiplicity correction",
            "confirmatory 2021-23 significance",
            "within-person longitudinal change",
            "external-population generalizability",
        ],
    }
    out = AUDIT / "51_lock_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def write_report(core_locked: bool):
    df = pd.DataFrame([asdict(x) for x in findings])
    out_csv = RESULTS / "51_lock_in_findings.csv"
    df.to_csv(out_csv, index=False)

    counts = df["status"].value_counts().to_dict() if len(df) else {}
    report = []
    report.append("# Chapter 2 lock-in falsification report")
    report.append("")
    report.append(f"**CORE LOCK STATUS: {'PASS' if core_locked else 'FAIL'}**")
    report.append("")
    report.append(
        "This is a falsification/stability audit, not mathematical proof that all bias is absent. "
        "A PASS means the prespecified core claim survived the checks implemented here."
    )
    report.append("")
    report.append("## Summary")
    for k in ["PASS","WARN","FAIL","INFO","UNTESTABLE"]:
        report.append(f"- {k}: {counts.get(k,0)}")
    report.append("")
    report.append("## Findings")
    report.append("")
    report.append("| Category | Test | Status | Critical | Evidence |")
    report.append("|---|---|---:|---:|---|")
    for f in findings:
        ev = f.evidence.replace("|","\\|").replace("\n"," ")
        report.append(
            f"| {f.category} | {f.test} | **{f.status}** | "
            f"{'yes' if f.critical else 'no'} | {ev} |"
        )
    report.append("")
    report.append("## Locked claim if CORE LOCK STATUS = PASS")
    report.append("")
    report.append(
        "> In NHANES, a frozen outcome-independent multidimensional physiological "
        "representation retains phenotype-associated information beyond simple threshold "
        "scalars, and that advantage reproduces under frozen 2009-18 temporal transfer. "
        "The result is associative, not causal."
    )
    report.append("")
    report.append("## Irreducible limitations")
    report.append("")
    report.append(
        "Unmeasured confounding, MNAR missingness/selection, reverse causation, causal "
        "roles of BMI/eGFR, and transportability outside NHANES cannot be eliminated by "
        "this script. They remain explicit limitations even when the lock passes."
    )
    report.append("")

    out_md = RESULTS / "51_lock_in_report.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    return out_csv, out_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerun-core", action="store_true",
        help="Rerun scripts 46-50 before auditing. Recommended for the final lock."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for prespecified lock checks (default 0.05)."
    )
    args = parser.parse_args()

    global ALPHA
    ALPHA = float(args.alpha)

    print("CHAPTER 2 LOCK-IN FALSIFICATION SUITE")
    print("====================================")
    print(f"Repository: {ROOT}")
    print(f"alpha: {ALPHA}")
    print()
    print("PASS != proof that every possible bias is absent.")
    print("This suite tries to falsify the locked claim and records what cannot be tested.")
    print()

    if args.rerun_core:
        try:
            rerun_core()
            add("reproducibility", "Core rerun 46-50", "PASS",
                "All five core scripts completed successfully.",
                "Current results were regenerated from current code/data.", True)
        except Exception as e:
            add("reproducibility", "Core rerun 46-50", "FAIL",
                repr(e), "Do not lock until the pipeline reruns cleanly.", True)

    try:
        audit_existing(ALPHA)
        adj, time = run_new_r_stress_tests()
        audit_new_stress_outputs(adj, time, ALPHA)

        input47 = RESULTS / "47_domain_corrected_AG_input.csv"
        if input47.exists():
            selection_audit(input47)
            component_vif_audit(input47)

    except Exception as e:
        add("execution", "Unhandled audit exception", "FAIL",
            repr(e), "Fix the audit error before locking.", True)

    commit, branch, dirty = git_info()
    if commit:
        add("reproducibility", "Git commit identified", "PASS",
            f"branch={branch}; commit={commit}",
            "Lock manifest can be tied to an exact repository state.", False)
        if dirty:
            add("reproducibility", "Working tree cleanliness", "WARN",
                "Working tree has uncommitted changes.",
                "Commit final scripts/results after reviewing the lock report.", False)
        else:
            add("reproducibility", "Working tree cleanliness", "PASS",
                "Working tree clean.",
                "Repository state is already reproducible by commit.", False)
    else:
        add("reproducibility", "Git state", "WARN",
            "Could not read git state.",
            "File hashes are still recorded, but no commit SHA is available.", False)

    critical_failures = [f for f in findings if f.critical and f.status == "FAIL"]
    core_locked = len(critical_failures) == 0

    out_csv, out_md = write_report(core_locked)
    manifest = make_manifest(core_locked, commit, branch, dirty)

    print()
    print("=" * 60)
    print(f"CORE LOCK STATUS: {'PASS' if core_locked else 'FAIL'}")
    print(f"Findings: {out_csv}")
    print(f"Report:   {out_md}")
    print(f"Manifest: {manifest}")
    print("=" * 60)

    if critical_failures:
        print("\nCRITICAL FAILURES:")
        for f in critical_failures:
            print(f"- [{f.category}] {f.test}: {f.evidence}")
        sys.exit(2)

    warnings = [f for f in findings if f.status in {"WARN","UNTESTABLE"}]
    if warnings:
        print("\nLOCK PASSED WITH EXPLICIT LIMITATIONS/WARNINGS:")
        for f in warnings:
            print(f"- [{f.status}] {f.test}: {f.interpretation}")

    sys.exit(0)


if __name__ == "__main__":
    main()
