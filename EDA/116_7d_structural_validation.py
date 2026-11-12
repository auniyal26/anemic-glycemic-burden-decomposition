#!/usr/bin/env python3
"""116 | Seven-dimensional structural validation with negative controls.

Place in EDA/ beside 72, 103, 112 and 114. Runs on SYNTHETIC observations;
reads no NHANES participant rows. Reuses Script114's official JIVE/AJIVE
adapters, synthetic generator and model-fitting protocol, not surrogates.

Predeclared questions:
  (1) Does the rank-selection gate avoid declaring shared H/G structure when
      the population contains none? (An independent synthetic null.)
  (2) Do two independently fitted, seven-score representations recover all
      seven observed marker coordinates equally well? (They should.)
  (3) Given a prelabelled synthetic truth, how selectively/stably do specific
      score groups recover their assigned target at MATCHED GROUP DIMENSIONS?

All-score recovery is an information-equivalence check, not evidence of novel
information. Groupwise R² uses a TRAIN-ONLY SUPERVISED truth decoder, so it
is diagnostic and NOT evidence of clinical/biological identification.
PCA leading-two is a variance baseline, NOT PCA's claimed shared component.
AJIVE can reduce its output rank; runs with <7 actual scores are reported but
EXCLUDED from fixed-7D comparisons (never padded or mislabelled as 7D).
No-sharing gate evaluates only the H/G CCA recipe. Fixed-rank JIVE/AJIVE
settings do not provide a corresponding adaptive rank-selection control.

Usage:
  python EDA/116_7d_structural_validation.py --self-test
  python EDA/116_7d_structural_validation.py --reps 50 --permutations 199
  python EDA/116_7d_structural_validation.py --methods HG7 PCA7 --reps 5 --permutations 49 --run-name 116_7d_pilot

Outputs: protocol, per-rep metrics, rank-selection, paired comparisons,
Monte-Carlo summaries, failures, manuscript-safe readout, optional SVG figures.
No PDF output. Requires numpy/pandas/scipy/scikit-learn; native JIVE/AJIVE need
exactly the R/Python environments already used by Script114.
"""
from __future__ import annotations
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler

EDA = Path(__file__).resolve().parent
ROOT = EDA.parent
SCENARIOS = ("shared_low_noise", "private_dominant", "discordant", "one_shared", "no_sharing")
METHODS = ("HG7", "PCA7", "JIVE7", "AJIVE7_adaptive")
GROUPS = {"shared": ("shared", 2), "discordant": ("discordant", 2),
          "H_private": ("H_private", 2), "G_private": ("G_private", 1)}
EXPECTED = {"shared": "shared", "discordant": "discordant",
            "H_private": "H_private", "G_private": "G_private"}


def invsqrt(a: np.ndarray) -> np.ndarray:
    e, v = np.linalg.eigh(a)
    if float(e.min()) < 1e-10:
        raise ValueError("Near-singular covariance in synthetic CCA")
    return (v * (e**(-0.5))) @ v.T


def canonical_rhos(raw: np.ndarray) -> np.ndarray:
    """Three H/G canonical correlations; equal weights, simulated data only."""
    x = StandardScaler().fit_transform(raw)
    a, g = x[:, :4], x[:, 4:]
    n = len(x)
    h = a.T @ a / n
    k = g.T @ g / n
    c = a.T @ g / n
    return np.clip(np.linalg.svd(invsqrt(h) @ c @ invsqrt(k),
                                 compute_uv=False), 0, 1)


def gate(raw: np.ndarray, permutations: int, rng: np.random.Generator,
         alpha: float, min_rho: float) -> dict:
    """Conservative max-correlation familywise permutation gate over 3 ranks.

    Every observed canonical correlation is compared with the MAXIMUM over
    all canonical correlations under each independent row permutation of G.
    This controls the familywise event of spurious *any coupling* under the
    independence null, but is conservative for later ranks. It does not
    identify physiological or causal shared sources.
    """
    observed = canonical_rhos(raw)
    max_null = np.empty(permutations)
    for i in range(permutations):
        shuffled = raw.copy()
        shuffled[:, 4:] = shuffled[rng.permutation(len(raw)), 4:]
        max_null[i] = canonical_rhos(shuffled)[0]
    p = np.array([(1 + np.count_nonzero(max_null >= value)) / (permutations + 1)
                  for value in observed], dtype=float)
    supported = (observed >= min_rho) & (p <= alpha)
    # Canonical correlations are ordered; require contiguous rank acceptance.
    selected = 0
    for yes in supported:
        if not bool(yes):
            break
        selected += 1
    return {"rho1": float(observed[0]), "rho2": float(observed[1]),
            "rho3": float(observed[2]), "p1_fwer": float(p[0]),
            "p2_fwer": float(p[1]), "p3_fwer": float(p[2]),
            "detected_rank": int(selected), "any_detected": bool(selected > 0)}


def data_generator(rng: np.random.Generator, n: int, scenario: str,
                   rotations: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Exact simulation structure from Script114; kept here for self-test."""
    s = rng.normal(size=(n, 2))
    d = rng.normal(size=(n, 2))
    ah = rng.normal(size=(n, 2))
    ag = rng.normal(size=(n, 1))
    if scenario == "no_sharing":
        hpair = rng.normal(size=(n, 2))
        gpair = rng.normal(size=(n, 2))
    elif scenario == "one_shared":
        hpair = np.column_stack((s[:, 0], rng.normal(size=n)))
        gpair = np.column_stack((s[:, 0], rng.normal(size=n)))
    else:
        delta = 0.6 if scenario == "discordant" else 0.0
        hpair = s + delta * d
        gpair = s - delta * d
    amp = 4.0 if scenario == "private_dominant" else 1.0
    h = np.column_stack((hpair, amp * ah)) @ rotations[0]
    g = np.column_stack((gpair, amp * ag)) @ rotations[1]
    x = np.column_stack((h, g)) + rng.normal(scale=0.2, size=(n, 7))
    truth = {"shared": s[:, :1] if scenario == "one_shared" else s,
             "H_private": ah, "G_private": ag}
    if scenario == "discordant":
        truth["discordant"] = d
    return x, truth


def affine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y,
                           rcond=1e-12)[0]


def predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(x)), x)) @ coef


def predictive_r2(train_scores: np.ndarray, test_scores: np.ndarray,
                  train_y: np.ndarray, test_y: np.ndarray) -> float:
    """Out-of-sample synthetic truth R²; supervised truth decoder fit on train."""
    if test_scores.shape[1] == 0:
        return float("nan")
    yhat = predict(test_scores, affine(train_scores, train_y))
    num = float(np.sum((test_y - yhat)**2))
    den = float(np.sum((test_y - np.mean(test_y, axis=0))**2))
    return 1 - num / den if den > 0 else float("nan")


def reconstruction_r2(train_x: np.ndarray, test_x: np.ndarray,
                      train_scores: np.ndarray, test_scores: np.ndarray) -> float:
    xhat = predict(test_scores, affine(train_scores, train_x))
    num = float(np.sum((test_x-xhat)**2))
    den = float(np.sum((test_x-train_x.mean(0))**2))
    return 1-num/den if den>0 else float("nan")


def source_hashes() -> dict:
    out={}
    for name in ("72_shared_private_decomposition_and_phenotype.py",
                 "103_jive_ajive_benchmark.py", "112_finish_benchmarks.py",
                 "114_structure_recovery.py", Path(__file__).name):
        p=EDA/name
        if p.is_file():
            out[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def load_module(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    if s is None or s.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def ci_paired_mean(x: np.ndarray, rng: np.random.Generator,
                   n_boot: int=1999) -> tuple[float,float]:
    """MC 95% interval for mean paired difference across independent replications."""
    n=len(x)
    if n<5:
        return float("nan"),float("nan")
    ix=rng.integers(0,n,size=(n_boot,n))
    means=x[ix].mean(axis=1)
    lo,hi=np.percentile(means,[2.5,97.5])
    return float(lo),float(hi)


def make_summaries(out: Path, records: list[dict], gates: list[dict],
                   failures: list[dict], seed: int, expected_reps: int,
                   methods: tuple[str,...]) -> dict:
    result={}
    met=pd.DataFrame(records)
    ng=pd.DataFrame(gates)
    nf=pd.DataFrame(failures,columns=["scenario","replicate","method","error"])
    ng.to_csv(out/"116_rank_gate.csv",index=False)
    nf.to_csv(out/"116_failures.csv",index=False)
    if met.empty:
        pd.DataFrame(columns=["scenario","replicate","method","scope","target","heldout_R2"]).to_csv(
            out/"116_metrics.csv",index=False)
        return {"fitted_records":0,"failures":len(failures)}
    met.to_csv(out/"116_metrics.csv",index=False)
    summary=met.groupby(["scenario","method","scope","target","actual_dim","group_dim","eligible_7D"],dropna=False).agg(
        successful_reps=("heldout_R2","count"),
        mean_R2=("heldout_R2","mean"),
        sd_R2=("heldout_R2","std"),
        p2p5_replicate_R2=("heldout_R2",lambda z:np.nanpercentile(z,2.5)),
        p97p5_replicate_R2=("heldout_R2",lambda z:np.nanpercentile(z,97.5)),
        mean_refit_similarity=("refit_similarity","mean"),
    ).reset_index()
    summary.to_csv(out/"116_summary.csv",index=False)
    # Pair method results by scenario and identical synthetic replicate. A comparison
    # requires actual 7D on BOTH sides and matching SUBSPACE score counts.
    comparators=[m for m in methods if m!="HG7"]
    rows=[]
    for scenario in SCENARIOS:
        for scope,target in [("shared","shared"),("H_private","H_private"),
                             ("G_private","G_private")]:
            base=met.loc[(met.scenario==scenario)&(met.method=="HG7")&
                         (met.scope==scope)&(met.target==target)&
                         (met.eligible_7D)]
            if base.empty:
                continue
            for method in comparators:
                candidate=met.loc[(met.scenario==scenario)&(met.method==method)&
                                  (met.scope==scope)&(met.target==target)&
                                  (met.eligible_7D)]
                if candidate.empty:
                    continue
                a=base[["replicate","heldout_R2","group_dim"]].rename(
                    columns={"heldout_R2":"z_R2","group_dim":"z_dim"})
                b=candidate[["replicate","heldout_R2","group_dim"]].rename(
                    columns={"heldout_R2":"ref_R2","group_dim":"ref_dim"})
                both=a.merge(b,on="replicate",validate="one_to_one")
                both=both.loc[both.z_dim.eq(both.ref_dim)].dropna()
                if both.empty:
                    continue
                diff=(both.z_R2-both.ref_R2).to_numpy(float)
                rng=np.random.default_rng(np.random.SeedSequence([
                    seed,SCENARIOS.index(scenario),METHODS.index(method),
                    list(GROUPS).index(scope)]))
                low,high=ci_paired_mean(diff,rng)
                rows.append({"scenario":scenario,"target":target,"scope":scope,
                             "comparison":"HG7 minus "+method,
                             "matched_group_dim":int(both.z_dim.iloc[0]),
                             "paired_reps":len(diff),"mean_delta_R2":float(diff.mean()),
                             "paired_bootstrap_mean_low95":low,"paired_bootstrap_mean_high95":high,
                             "remark":"Synthetic truth decoder; exploratory; no biological claim"})
    pd.DataFrame(rows,columns=["scenario","target","scope","comparison","matched_group_dim",
                               "paired_reps","mean_delta_R2","paired_bootstrap_mean_low95",
                               "paired_bootstrap_mean_high95","remark"]).to_csv(out/"116_paired_group_differences.csv",index=False)
    # Null gate acceptance diagnostics include exact binomial interval and
    # a one-sided test against nominal 5% familywise error, NOT a pass claim.
    null=ng[ng.scenario.eq("no_sharing")]
    k=int(null.any_detected.sum()); n=len(null)
    if n:
        b=binomtest(k,n)
        c=b.proportion_ci(confidence_level=.95,method="exact")
        p=float(binomtest(k,n,.05,alternative="greater").pvalue)
        gate_summary={"n_no_sharing":n,"spurious_any_shared":k,
                      "observed_false_positive_rate":k/n,
                      "exact_binomial_interval_95":[float(c.low),float(c.high)],
                      "one_sided_p_vs_nominal_0p05":p}
    else:
        gate_summary={"n_no_sharing":0}
    gate_summary["mean_selected_rank_by_scenario"]={
        scenario:float(ng[ng.scenario.eq(scenario)].detected_rank.mean())
        for scenario in SCENARIOS}
    # The one-shared scenario detects potential overcounting of a second rank.
    # This is descriptive; the global-independence permutation reference is
    # not a formal sequential test under the rank-one partial null.
    one=ng[ng.scenario.eq("one_shared")]
    if len(one):
        gate_summary["one_shared_overselected_rank2_or_more"]={
            "count":int(one.detected_rank.ge(2).sum()),
            "n":int(len(one)),
            "fraction":float(one.detected_rank.ge(2).mean()),
            "qualification":"descriptive only; not a valid partial-null rank test"}
    (out/"116_gate_summary.json").write_text(json.dumps(gate_summary,indent=2),encoding="utf-8")
    result=dict(fitted_records=len(met),failures=len(failures),
                complete_fit_fraction=(len(met.loc[met.scope.eq("all")])/max(1,expected_reps*len(SCENARIOS)*len(methods))),
                gate=gate_summary,
                strict_7d_fit_counts={str(k):int(v) for k,v in
                   met.loc[met.scope.eq("all")&met.eligible_7D].groupby("method").size().to_dict().items()})
    text=["# Script 116: seven-dimensional structure validation", "",
          "**Synthetic experiment only.** Fixed 7D budget for HG7, PCA7 and official JIVE7;",
          "AJIVE's actual rank is reported and <7D runs are excluded from strict comparisons.","",
          "## Negative control", "",
          f"No-sharing: {k}/{n} replicates falsely declared any coupling via max-statistic permutation gate.",
          f"Observed rate = {k/n if n else float('nan'):.4f}; exact binomial 95% interval "
          f"{gate_summary.get('exact_binomial_interval_95','N/A')}; one-sided p vs 0.05 "
          f"{gate_summary.get('one_sided_p_vs_nominal_0p05','N/A')}.","",
          "A non-significant calibration test is NOT proof of nominal error control.","",
          "## Evidence boundaries", "",
          "- A full-rank 7D affine map retains all seven measured coordinates; identical all-score decoding is expected.",
          "- Global-independence max-permutation calibration is NOT a formal sequential test of rank 2 vs rank 1.",
          "- Group tests use train-only synthetic-truth decoders. They do not validate physiology in humans.",
          "- Groupwise comparisons match dimensions but not the underlying semantic objective of each method.",
          "- PCA first two scores are used only as a leading-variance comparator.",
          "- An explicit discordant group is absent in the comparator native structures; do not substitute a labelled proxy.",
          "- H/G group dimensions of JIVE/AJIVE are truncated using native score order, not fitted to test truth.",
          "- Cross-sectional NHANES outcomes and patient-level validation are not evaluated here.",
          "- These Monte Carlo intervals quantify sampling across simulations, not clinical/survey confidence coverage.",
          "- If simulated results favor HG7, they establish only advantage under the stipulated data generator.","",
          "## Output inventory", "",
          "- 116_metrics.csv: all-score and native-group held-out target R² plus group refit stability",
          "- 116_summary.csv: replicate distribution (not CI) of metric values",
          "- 116_paired_group_differences.csv: paired-bootstrap intervals of mean differences",
          "- 116_rank_gate.csv / 116_gate_summary.json: conservative multiplicity-aware null gate",
          "- 116_failures.csv: full tracebacks; no failures silently removed",
          ""]
    (out/"116_READOUT.md").write_text("\n".join(text),encoding="utf-8")
    return result


def save_figures(out:Path):
    """Optional static SVGs only; no PDF or GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pair=out/"116_paired_group_differences.csv"
    if pair.exists():
        df=pd.read_csv(pair)
        df=df.loc[df.scenario.eq("discordant") & df.paired_reps.ge(5)].copy()
        if not df.empty:
            df=df.sort_values(["scope","comparison"])
            fig,ax=plt.subplots(figsize=(8.5,max(3.2,len(df)*.38+1.4)))
            y=np.arange(len(df)); lo=df.paired_bootstrap_mean_low95.to_numpy(float)
            hi=df.paired_bootstrap_mean_high95.to_numpy(float);v=df.mean_delta_R2.to_numpy(float)
            ax.errorbar(v,y,xerr=np.vstack((v-lo,hi-v)),fmt='o',color='#185c58',capsize=3)
            ax.axvline(0,color='#555',lw=.8)
            ax.set_yticks(y,[f"{r.scope} ({r.matched_group_dim}D) vs {r.comparison.split('minus ')[-1]}" for r in df.itertuples()])
            ax.invert_yaxis();ax.set_xlabel("Paired synthetic truth-recovery R² difference (HG7 minus comparator)")
            ax.set_title("7D models: matched-group diagnostic (discordance scenario)")
            ax.text(.02,-.15,"Only actual 7D runs; train-only supervised truth decoders; not clinical identification",
                    transform=ax.transAxes,fontsize=8)
            fig.tight_layout();fig.savefig(out/"116_group_recovery_7d.svg",bbox_inches="tight");plt.close(fig)
    gates=pd.read_csv(out/"116_rank_gate.csv")
    rates=gates.groupby("scenario",sort=False).any_detected.agg(['mean','sum','count']).reindex(SCENARIOS)
    fig,ax=plt.subplots(figsize=(7.3,3.8))
    ax.bar(np.arange(len(SCENARIOS)),rates['mean'].fillna(0).to_numpy(),color="#185c58")
    ax.axhline(.05,ls='--',lw=.9,color='#a04b33',label="Nominal familywise alpha = 0.05")
    ax.set_ylim(0,1.05);ax.set_ylabel("Fraction detecting any H/G coupling")
    ax.set_xticks(np.arange(len(SCENARIOS)),SCENARIOS,rotation=15,ha='right')
    ax.legend(frameon=False,loc='upper right')
    ax.set_title("Unsupervised coupling detection with a no-sharing negative control")
    fig.tight_layout();fig.savefig(out/"116_null_control.svg",bbox_inches='tight');plt.close(fig)


def self_test():
    rng=np.random.default_rng(456)
    rotations=[np.linalg.qr(rng.normal(size=(k,k)))[0] for k in (4,3)]
    for sc in SCENARIOS:
        x,y=data_generator(rng,500,sc,rotations)
        assert x.shape==(500,7) and set(y)>=set(("shared","H_private","G_private"))
        z=gate(x,39,rng,.05,.1)
        assert all(0<=z[k]<=1 for k in ("rho1","rho2","rho3","p1_fwer","p2_fwer","p3_fwer"))
        assert 0<=z['detected_rank']<=3
        assert z['rho1']>=z['rho2']>=z['rho3']
        if sc=="shared_low_noise":
            assert z['detected_rank']>=1, "Strong synthetic coupling should be detected"
    x=rng.normal(size=(200,7)); a=rng.normal(size=(7,7)); v=x@a
    assert np.linalg.matrix_rank(a)==7
    assert abs(reconstruction_r2(x,x,v,v)-1)<1e-9
    assert abs(predictive_r2(v,v,x,x)-1)<1e-9
    low,high=ci_paired_mean(np.linspace(-1,1,15),rng)
    assert low<0<high
    print("PASS: CCA gate, simulated no-sharing generator, affine full-rank reconstruction, paired-mean intervals")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reps',type=int,default=50)
    p.add_argument('--n-train',type=int,default=500)
    p.add_argument('--n-test',type=int,default=1500)
    p.add_argument('--permutations',type=int,default=199)
    p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--min-rho',type=float,default=.10)
    p.add_argument('--seed',type=int,default=116)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--run-name',default="116_7d_structural_validation")
    p.add_argument('--timeout',type=int,default=1200)
    p.add_argument('--rscript',type=Path,default=None)
    p.add_argument('--plot',action='store_true',help="Save manuscript candidate SVGs after simulation")
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        return self_test()
    if args.reps<5 or args.n_train<50 or args.n_test<50 or args.permutations<19:
        p.error("Require reps>=5, both sample sizes>=50, permutations>=19")
    if not (0<args.alpha<1 and 0<=args.min_rho<=1):
        p.error("Invalid alpha or min-rho")
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        p.error("run-name may contain only A-Z a-z 0-9 underscore or hyphen")
    out=EDA/'Results'/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    protocol_path=out/'116_protocol.json'
    if protocol_path.exists():
        raise FileExistsError(f"Already contains a 116 protocol: {out}. Change --run-name to protect prior results.")
    needed=[EDA/n for n in ("72_shared_private_decomposition_and_phenotype.py",
                           "103_jive_ajive_benchmark.py", "112_finish_benchmarks.py",
                           "114_structure_recovery.py")]
    missing=[str(x) for x in needed if not x.is_file()]
    if missing:
        raise FileNotFoundError("Required scripts not found next to Script116: "+", ".join(missing))
    elligible=tuple(dict.fromkeys(args.methods))
    rscript=args.rscript or (Path(sys.prefix)/'Lib/R/bin/x64/Rscript.exe')
    if not rscript.is_file():
        candidate=shutil.which('Rscript')
        if candidate:rscript=Path(candidate)
    if 'JIVE7' in elligible and not rscript.is_file():
        raise FileNotFoundError("Official JIVE7 requested but Rscript not found. Pass --rscript PATH.")
    protocol={"source_hashes":source_hashes(),"argv":sys.argv[1:],
              "settings":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
              "design":"Simulation: five prespecified conditions including rank-one and rank-zero negative controls; independent repeats; 7D-only fit policy",
              "group_rules":"HG S2/D2/H2/G1. JIVE/AJIVE H_individual first 2 of 3, G_individual first 1 of 2 in native order; PCA leading2 only.",
              "decision_rules":["No claim of novel information from any invertible 7D map",
                 "AJIVE adaptive ranks <7 flagged and excluded from equal-7D comparisons",
                 "No causal or clinical/physiological identification from synthetic target recovery",
                 "Use max-correlation permutation familywise gate on H/G independent no-sharing data",
                 "One-shared scenario is a descriptive check of rank over-selection, not a calibrated partial-null test",
                 "No direct discordance comparator because JIVE/AJIVE/PCA do not prelabel it",
                 "Preserve method failures, rank losses, and both unfavorable and favorable intervals"]}
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    source114=load_module('s114_for_116', needed[3]);helpers=source114.cca_functions(needed[0])
    benchmark=load_module('s103_for_116',needed[1]);native=load_module('s112_for_116',needed[2])
    rows=[]; gates=[]; failures=[];started=time.monotonic()
    total=len(SCENARIOS)*args.reps*len(elligible); n_done=0
    for scenario_id,scenario in enumerate(SCENARIOS):
        for rep in range(args.reps):
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,scenario_id,rep]))
            rotations=[np.linalg.qr(rng.normal(size=(k,k)))[0] for k in (4,3)]
            train,truth=data_generator(rng,args.n_train,scenario,rotations)
            test,truth_test=data_generator(rng,args.n_test,scenario,rotations)
            null_rng=np.random.default_rng(np.random.SeedSequence([args.seed,scenario_id,rep,888]))
            gate_result=gate(train,args.permutations,null_rng,args.alpha,args.min_rho)
            gates.append({"scenario":scenario,"replicate":rep,**gate_result})
            boot_idx=rng.integers(0,len(train),len(train))
            for method in elligible:
                n_done+=1
                print(f"[{n_done}/{total}] {scenario} rep={rep+1} {method} ({time.monotonic()-started:.0f}s)",flush=True)
                try:
                    first=source114.fit_method(method,train,helpers,benchmark,native,rscript,args.timeout)
                    second=source114.fit_method(method,train[boot_idx],helpers,benchmark,native,rscript,args.timeout)
                    ztr=first(train);zte=first(test);zbo=second(test)
                    actual=int(zte['all'].shape[1]);is7=(actual==7)
                    if method in ('HG7','PCA7','JIVE7') and not is7:
                        raise ValueError(f"{method} did not return rank 7: {actual}")
                    recon=reconstruction_r2(train,test,ztr['all'],zte['all'])
                    if is7 and abs(recon-1)>1e-8:
                        raise ValueError(f"Full-rank reconstruction failed: R²={recon}")
                    # All-score truth readout is information-equivalence control.
                    for target,ttr in truth.items():
                        r2=predictive_r2(ztr['all'],zte['all'],ttr,truth_test[target])
                        rows.append({"scenario":scenario,"replicate":rep,"method":method,
                            "scope":"all","target":target,"actual_dim":actual,"group_dim":actual,
                            "eligible_7D":is7,"reconstruction_R2":recon,"heldout_R2":r2,
                            "refit_similarity":source114.stability(zte['all'],zbo['all']),
                            "declared_shared_rank":gate_result['detected_rank']})
                    # Group scope: group dimensions are matched by deterministic
                    # first-k native-score truncation, with no truth-guided selection.
                    for scope,(label,dimension) in GROUPS.items():
                        if label not in ztr or label not in zte or label not in zbo:
                            continue
                        if min(ztr[label].shape[1],zte[label].shape[1],zbo[label].shape[1])<dimension:
                            continue
                        at=ztr[label][:,:dimension]; bt=zte[label][:,:dimension];ct=zbo[label][:,:dimension]
                        st=source114.stability(bt,ct)
                        for target,ttr in truth.items():
                            r2=predictive_r2(at,bt,ttr,truth_test[target])
                            rows.append({"scenario":scenario,"replicate":rep,"method":method,
                                "scope":scope,"target":target,"actual_dim":actual,"group_dim":dimension,
                                "eligible_7D":is7,"reconstruction_R2":recon,"heldout_R2":r2,
                                "refit_similarity":st,"declared_shared_rank":gate_result['detected_rank']})
                except Exception:
                    failures.append({"scenario":scenario,"replicate":rep,"method":method,
                                     "error":traceback.format_exc()})
                    print(f"METHOD FAILED: {scenario} rep={rep} method={method}",file=sys.stderr,flush=True)
                pd.DataFrame(rows).to_csv(out/"116_metrics.csv",index=False)
                pd.DataFrame(gates).to_csv(out/"116_rank_gate.csv",index=False)
                pd.DataFrame(failures,columns=["scenario","replicate","method","error"]).to_csv(
                    out/"116_failures.csv",index=False)
    completed=make_summaries(out,rows,gates,failures,args.seed,args.reps,elligible)
    (out/"116_completion.json").write_text(json.dumps(completed,indent=2),encoding='utf-8')
    if args.plot:save_figures(out)
    print(f"DONE: read {out/'116_READOUT.md'} and inspect 116_failures.csv / 116_rank_gate.csv",flush=True)


if __name__=='__main__':
    main()
