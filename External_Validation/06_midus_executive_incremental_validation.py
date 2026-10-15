#!/usr/bin/env python3
"""SCRIPT 06 | Validate whether the frozen two-PC glycemic representation adds
meaningful external executive-function information beyond HbA1c.

Run from the repository root:
    python External_Validation/06_midus_executive_incremental_validation.py

Required neighbors:
    External_Validation/05_midus_frozen_glycemic_transfer.py
    External_Validation/Results/05_manifest.json
    External_Validation/Results/05_cognitive_performance.csv

Reuses Script 05's local NHANES (2005-08) G3 reference, MIDUS reads,
participant eligibility, and unchanged Ridge alpha. It DOES NOT optimize
models using Refresher outcomes, change frozen NHANES artifacts, or output
participant-level records. Only aggregate statistics are saved.

Primary sensitivity (post-hoc, hypothesis-generating):
    Refresher locked Core-trained NHANES_PC2 vs Core-trained HbA1c.
    MSE, paired participant bootstrap CIs; error-variance vs squared-bias
    contribution to MSE, with within-cohort baseline-score reference checks.

Score-reference sensitivity is NOT psychometric measurement invariance.
The baseline-anchored analysis uses Refresher baseline cognitive score
moments (not future outcomes) to create a cohort-relative scale. It is
therefore a test-cohort baseline-referenced adaptation and must not be
represented as a completely frozen real-world individual deployment.

Dependencies: numpy, pandas, scikit-learn, pyreadstat (as Script 05).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "Results"
PREVIOUS = SCRIPT_DIR / "05_midus_frozen_glycemic_transfer.py"
SEED = 20261015
ALPHA = 10.0
FOLDS = 5
BOOT = 3000
PRIMARY_DOMAIN = "executive"
DOMAINS = ("comp", "memory", "executive")
MODELS = ("covariates", "HbA1c", "NHANES_PC2")
SCENARIOS = ("native_frozen", "baseline_anchored", "within_person_change")
SUPPRESS = "SUPPRESSED_0_TO_10"


def number(n):
    return str(int(n)) if int(n) > 10 else SUPPRESS


def fmt(x, places=5):
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{places}f}"


def load_script05():
    if not PREVIOUS.is_file():
        raise FileNotFoundError(f"Script 05 is required unmodified next to Script 06: {PREVIOUS}")
    spec = importlib.util.spec_from_file_location("midus_script05_reference", PREVIOUS)
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load existing Script 05")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if (mod.SEED, mod.FOLDS, mod.ALPHA) != (SEED, FOLDS, ALPHA):
        raise RuntimeError("Script 05 model setup changed: stop and review before comparing")
    if tuple(mod.DOMAIN) != DOMAINS or not set(MODELS).issubset(mod.SPECS):
        raise RuntimeError("Script 05 model/domain names changed: stop and review")
    return mod


def read_prior_artifacts(s05):
    manifest_path = RESULTS / "05_manifest.json"
    metric_path = RESULTS / "05_cognitive_performance.csv"
    for p in (manifest_path, metric_path):
        if not p.is_file():
            raise FileNotFoundError(f"Script 05 must have been run successfully: missing {p}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("nhanes_sha256") != hashlib.sha256(s05.SOURCE.read_bytes()).hexdigest():
        raise RuntimeError("NHANES reference differs from Script 05 manifest; refusing silent comparison")
    if (manifest.get("seed"), manifest.get("folds"), float(manifest.get("ridge_alpha", -1))) != (SEED, FOLDS, ALPHA):
        raise RuntimeError("Stored Script 05 hyperparameters differ; refusing silent comparison")
    if manifest.get("original_7D_map_modified") is not False:
        raise RuntimeError("Previous manifest does not confirm untouched 7D map")
    prior = pd.read_csv(metric_path)
    required = {"domain", "model", "cohort", "evaluation", "MSE", "R2", "bias", "n"}
    if not required.issubset(prior.columns):
        raise RuntimeError(f"Missing Script 05 performance columns: {required-set(prior.columns)}")
    return manifest, prior


def build_inputs(s05):
    f = s05.frozen_nhanes()
    data = {}
    observed = {}
    for key in s05.FILES:
        data[key], src = s05.load_midus(key)
        observed[key] = {"path": str(src.relative_to(ROOT)), "n": len(data[key])}
        print(f"  {key:11s} | {len(data[key]):5d} records")
    cohorts = {}
    for name in s05.COHORTS:
        cohort, counts = s05.prepare_cohort(name, data)
        cohorts[name] = s05.add_pcs(cohort, f)
        print(f"  {name:10s} | eligible chronology/physiology/covariates: {number(counts['lab_ordered_cov'])}")
    return f, cohorts, observed


def assembled(s05, cohorts, domain, scenario):
    core = s05.cohort_domain(cohorts["CORE"], domain)
    ref = s05.cohort_domain(cohorts["REFRESHER"], domain)
    if min(len(core), len(ref)) < 100:
        raise RuntimeError(f"Too few participants for {domain}/{scenario}")
    core, ref = core.copy(), ref.copy()
    base0 = core["base_"+domain].to_numpy(dtype=float)
    base1 = ref["base_"+domain].to_numpy(dtype=float)
    next0 = core["follow_"+domain].to_numpy(dtype=float)
    next1 = ref["follow_"+domain].to_numpy(dtype=float)
    if scenario == "native_frozen":
        core["base_score"] = base0
        ref["base_score"] = base1
        y0, y1 = next0, next1
        note = "Unmodified Script 05 score construction and frozen Core-to-Refresher transport"
    elif scenario == "baseline_anchored":
        m0, sd0 = float(base0.mean()), float(base0.std(ddof=0))
        m1, sd1 = float(base1.mean()), float(base1.std(ddof=0))
        if min(sd0, sd1) <= 1e-8:
            raise RuntimeError("Baseline BTACT score SD degenerate")
        core["base_score"] = (base0-m0)/sd0
        ref["base_score"] = (base1-m1)/sd1
        y0, y1 = (next0-m0)/sd0, (next1-m1)/sd1
        note = ("Cohort-specific baseline-only centering/scaling: uses Refresher earlier BTACT moments, "
                "never later outcomes; test-cohort normalization, not strictly frozen deployment")
    elif scenario == "within_person_change":
        core["base_score"] = base0
        ref["base_score"] = base1
        y0, y1 = next0-base0, next1-base1
        note = "Predict follow-up minus earlier BTACT on each native within-cohort reference scale"
    else:
        raise ValueError(scenario)
    c0,c1,levels,unseen = s05.encode_covariates(core, ref)
    if any(unseen.values()):
        raise RuntimeError("Unseen Refresher categories in Script 05 encoding")
    xs = {}
    for model in MODELS:
        fields = s05.model_features(model)
        a = pd.concat([c0,core[fields]],axis=1).to_numpy(float)
        b = pd.concat([c1,ref[fields]],axis=1).to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError("Invalid feature matrix")
        xs[model] = (a,b)
    return np.asarray(y0,float),np.asarray(y1,float),xs,len(core),len(ref),note


def predict_models(s05,y_train,train_test):
    pred_cv = {}
    pred_external = {}
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for model, (x_core,x_ref) in train_test.items():
        cv_preds = np.empty(len(y_train), dtype=float)
        for train_idx, val_idx in splitter.split(x_core):
            scaler, estimator = s05.fit_model(x_core[train_idx],y_train[train_idx])
            cv_preds[val_idx] = s05.pred(scaler,estimator,x_core[val_idx])
        pred_cv[model]=cv_preds
        scaler, estimator = s05.fit_model(x_core,y_train)
        pred_external[model] = s05.pred(scaler,estimator,x_ref)
    return pred_cv,pred_external


def verify_exact_reproduction(s05,prior,domain,cohort,y,pm):
    metric_cohort = "CORE" if cohort == "CORE" else "REFRESHER"
    evaluation = "five_fold_CV" if cohort == "CORE" else "locked_external"
    for model in MODELS:
        row = prior.loc[(prior.domain==domain)&(prior.model==model)&
                        (prior.cohort==metric_cohort)&(prior.evaluation==evaluation)]
        if len(row)!=1:
            raise RuntimeError(f"Could not find unique Script 05 metric for {domain}/{cohort}/{model}")
        st=s05.stats(y,pm[model]); saved=row.iloc[0]
        for metric in ("MSE", "R2", "bias"):
            if not np.isclose(st[metric],float(saved[metric]),atol=1e-10,rtol=1e-9):
                raise RuntimeError(f"Script 05 prediction mismatch: {domain}/{cohort}/{model}/{metric}: "
                                   f"rebuilt={st[metric]} saved={saved[metric]}")
        if int(saved["n"]) != len(y):
            raise RuntimeError(f"Script 05 sample changed: {domain}/{cohort}/{model}")


def components(y, p_a, p_b):
    """Improvements in b relative to a: + is favorable to b."""
    e_a = np.asarray(p_a,float)-np.asarray(y,float)
    e_b = np.asarray(p_b,float)-np.asarray(y,float)
    ma,mb=float(e_a.mean()),float(e_b.mean())
    va,vb=float(np.mean((e_a-ma)**2)),float(np.mean((e_b-mb)**2))
    total = float(np.mean(e_a**2)-np.mean(e_b**2))
    variance_part = va-vb
    bias_part = ma**2-mb**2
    assert np.isclose(total,variance_part+bias_part,atol=1e-12)
    return dict(delta_MSE=total, delta_centered_error_MSE=variance_part,
                delta_squared_bias=bias_part,
                MSE_reference=float(np.mean(e_a**2)),MSE_comparison=float(np.mean(e_b**2)),
                bias_reference=ma,bias_comparison=mb,
                centered_MSE_reference=va,centered_MSE_comparison=vb)


def boot_ci(y,p_a,p_b,reps,seed):
    if len(y) <= 10:
        raise RuntimeError("Participant count cannot be disclosed")
    a=np.asarray(p_a,float)-np.asarray(y,float)
    b=np.asarray(p_b,float)-np.asarray(y,float)
    rng=np.random.default_rng(seed)
    samples=np.empty((reps,3),dtype=float)
    n=len(a)
    for j in range(reps):
        ids=rng.integers(0,n,n)
        ea,eb=a[ids],b[ids]
        ea_mu,eb_mu=ea.mean(),eb.mean()
        bias_part=ea_mu**2-eb_mu**2
        var_part=np.mean((ea-ea_mu)**2)-np.mean((eb-eb_mu)**2)
        samples[j]=[var_part+bias_part,var_part,bias_part]
    return np.quantile(samples,[.025,.975],axis=0)


def audit(s05,y, preds, cohort, scenario, domain, reps, seed):
    rows=[]; errors=[]
    for compare, base in (("NHANES_PC2","HbA1c"),
                          ("NHANES_PC2","covariates"),
                          ("HbA1c","covariates")):
        a,b=preds[base],preds[compare]
        point=components(y,a,b)
        bounds=boot_ci(y,a,b,reps,seed+100*DOMAINS.index(domain)+\
                       10*SCENARIOS.index(scenario)+len(rows))
        row=dict(domain=domain,scenario=scenario,cohort=cohort,
                 reference_model=base,comparison_model=compare,n=number(len(y)),
                 delta_MSE=point["delta_MSE"],
                 delta_MSE_ci95_low=float(bounds[0,0]),delta_MSE_ci95_high=float(bounds[1,0]),
                 delta_centered_error_MSE=point["delta_centered_error_MSE"],
                 centered_ci95_low=float(bounds[0,1]),centered_ci95_high=float(bounds[1,1]),
                 delta_squared_bias=point["delta_squared_bias"],
                 bias_ci95_low=float(bounds[0,2]),bias_ci95_high=float(bounds[1,2]),
                 MSE_reference=point["MSE_reference"],
                 MSE_comparison=point["MSE_comparison"])
        rows.append(row)
        errors.append({k:row[k] for k in ("domain","scenario","cohort","reference_model",
                    "comparison_model","n")}
                      |{k:point[k] for k in ("bias_reference","bias_comparison",
                                                 "centered_MSE_reference","centered_MSE_comparison")}
                      |{k:point[k] for k in ("delta_MSE","delta_centered_error_MSE",
                                                 "delta_squared_bias")})
    return rows,errors


def report(rows,metadata):
    data=pd.DataFrame(rows)
    report=["# Script 06 — Executive-function transfer: does frozen PC2 add beyond HbA1c?","",
            "**Primary comparison:** the already-tested Core-trained `NHANES_PC2` model versus the Core-trained `HbA1c` model, evaluated in the same Refresher participants.",
            "**Status:** post-hoc validation/sensitivity analysis, not an independent confirmation of a prespecified hypothesis.",
            "**Data privacy:** no subject IDs, individual rows, fitted values or residuals exported; N <= 10 suppressed.",
            "", "## Locked executive outcome: paired comparisons", "",
            "Positive ΔMSE favors the **comparison** model. Intervals are descriptive participant-paired bootstrap percentiles; they do not account for cohort sampling, repeated model fitting, confounder bias, or multiple comparisons.", "",
            "| Comparison (external Refresher) | N | ΔMSE | 95% bootstrap CI | Centered-error Δ | Squared-bias Δ |",
            "|---|---:|---:|---|---:|---:|"]
    for comp,base in (("NHANES_PC2","HbA1c"),("NHANES_PC2","covariates"),("HbA1c","covariates")):
        r=data.loc[(data.domain=="executive")&(data.scenario=="native_frozen") &
                   (data.cohort=="REFRESHER")&(data.comparison_model==comp)&
                   (data.reference_model==base)].iloc[0]
        report.append(f"| {base} → {comp} | {r.n} | {fmt(r.delta_MSE)} | [{fmt(r.delta_MSE_ci95_low)}, {fmt(r.delta_MSE_ci95_high)}] | {fmt(r.delta_centered_error_MSE)} | {fmt(r.delta_squared_bias)} |")
    report += ["", "**Exact identity:** ΔMSE = Δ(centered prediction-error variance) + Δ(squared mean prediction bias).",
               "Centered-error MSE is an **outcome-informed, hypothetical perfect mean-bias correction** and is a diagnostic, not prospective deployable performance.",
               "", "## Score-reference / outcome-definition sensitivity (executive)","",
               "| Score treatment | ΔMSE PC2 versus HbA1c | 95% paired CI | Centered-error Δ |",
               "|---|---:|---|---:|"]
    for sc in SCENARIOS:
        r=data.loc[(data.domain=="executive")&(data.scenario==sc)&(data.cohort=="REFRESHER")&
                   (data.comparison_model=="NHANES_PC2")&
                   (data.reference_model=="HbA1c")].iloc[0]
        report.append(f"| {sc} | {fmt(r.delta_MSE)} | [{fmt(r.delta_MSE_ci95_low)}, {fmt(r.delta_MSE_ci95_high)}] | {fmt(r.delta_centered_error_MSE)} |")
    report += ["", "**Interpretation of scenarios:**",
               "- `native_frozen`: exactly reproduces Script 05's cohort-native BTACT source scoring and entirely frozen Core→Refresher model.",
               "- `baseline_anchored`: scores in each cohort are expressed relative to that cohort's *earlier* BTACT mean/SD; Refresher predictor-baseline distribution is used (without later outcomes). This is cohort-baseline adaptation, not a strictly frozen single-person deployment.",
               "- `within_person_change`: outcome is follow-up minus earlier BTACT, retaining each study's native standardization; baseline BTACT stays a covariate.",
               "- Different BTACT national reference populations, practice/retest effects and unknown measurement invariance remain unresolved by any of these checks. Do NOT claim proven cross-cohort phenotype equivalence.",
               "", "## Domain-specificity check (native locked transfer)","",
               "| Domain | Refresher N | ΔMSE PC2 versus HbA1c | 95% paired CI |",
               "|---|---:|---:|---|"]
    for d in DOMAINS:
        r=data.loc[(data.domain==d)&(data.scenario=="native_frozen")&(data.cohort=="REFRESHER")&
                   (data.comparison_model=="NHANES_PC2")&
                   (data.reference_model=="HbA1c")].iloc[0]
        report.append(f"| {d} | {r.n} | {fmt(r.delta_MSE)} | [{fmt(r.delta_MSE_ci95_low)}, {fmt(r.delta_MSE_ci95_high)}] |")
    report += ["", "## What the experiment cannot establish", "",
               "1. Does not establish causal physiology→cognition effects or clinical usefulness; the first glycemic assessment followed the earlier BTACT assessment.",
               "2. No MIDUS lab-assay cross-calibration, sampling/attrition weighting, medication/fasting validation, or complete psychometric invariance test.",
               "3. The two-PC model was selected for this comparison **after inspecting Script 05's executive result**. A positive paired bootstrap does not become independent external confirmation.",
               "4. Uses the NEW NHANES-frozen G3 PCA benchmark, not the original full seven-marker A/G shared–discordant map.",
               "5. Does not use Refresher follow-up outcomes to fit predictor coefficients, select a model, or choose bootstrap settings; within-person-change and baseline-anchored scores are sensitivity definitions.",
               "6. Refresher and Core remain different population and scoring references; re-anchoring is diagnostic, not a formal crosswalk.",
               "", "## Reproducibility", "",
               f"- Script 05 outputs exactly reproduced on native-scenario Core CV and locked Refresher, for all three domains/models checked.",
               f"- Bootstrap replicates: {metadata['bootstrap_reps']}; seed: {metadata['seed']}; no participant-level output.",
               f"- NHANES frozen reference SHA256: `{metadata['nhanes_sha256']}`.",
               "", "## Files", "", "- `06_readout.md` (this report)",
               "- `06_paired_model_comparisons.csv` (all three external and Core CV models, domain/score sensitivity and paired intervals)",
               "- `06_error_decomposition.csv` (model-specific bias and centered errors; no participant rows)",
               "- `06_manifest.json` (reproducibility details)","",
               "**Decision gate:** Judge added information over HbA1c only after the paired difference, centered-error contribution and score-reference sensitivities are read together. No result is interpreted as a novel external confirmation solely because an interval excludes zero."]
    (RESULTS / "06_readout.md").write_text("\n".join(report)+"\n",encoding="utf-8")


def smoke_test():
    rng=np.random.default_rng(SEED)
    y=rng.normal(size=250)
    a=y+rng.normal(loc=.16,scale=.4,size=len(y))
    b=y+rng.normal(loc=.08,scale=.42,size=len(y))
    p=components(y,a,b)
    assert np.isclose(p["delta_MSE"],p["delta_centered_error_MSE"]+p["delta_squared_bias"])
    ci=boot_ci(y,a,b,150,SEED)
    assert ci.shape==(2,3) and np.isfinite(ci).all()
    s05=load_script05()
    X=rng.normal(size=(180,6)); Y=.5*X[:,0]+rng.normal(size=180)
    sc,mod=s05.fit_model(X,Y)
    assert np.isfinite(s05.pred(sc,mod,X)).all()
    print("SCRIPT 06 SMOKE TEST PASS | error identity, paired bootstrap, existing Script 05 model")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--smoke-test",action="store_true",help="Run numerical tests only, no cohort files")
    cli.add_argument("--bootstrap-reps",type=int,default=BOOT,help="Paired descriptive bootstrap replicates (default 3000)")
    args=cli.parse_args()
    if args.smoke_test:
        smoke_test(); return
    if not (500 <= args.bootstrap_reps <= 10000):
        raise ValueError("--bootstrap-reps must be within 500..10000")
    s05=load_script05()
    prior_manifest,prior_metrics=read_prior_artifacts(s05)
    frozen,cohorts,observed=build_inputs(s05)
    if frozen["sha256"] != prior_manifest["nhanes_sha256"]:
        raise RuntimeError("NHANES data/reference not identical to Script 05")
    for key, rec in observed.items():
        prev=prior_manifest.get("midus_files",{}).get(key)
        if not prev or rec != prev:
            raise RuntimeError(f"MIDUS input provenance differs from Script 05 for {key}")
    comparisons=[]; errors=[]; analyses=[]
    # Reproduce primary native-score analysis across all three outcomes, including
    # the two domains that did not show improvement in Script 05.
    jobs=[(d,"native_frozen") for d in DOMAINS] + [
        (PRIMARY_DOMAIN,"baseline_anchored"),
        (PRIMARY_DOMAIN,"within_person_change")]
    for domain,scenario in jobs:
        y_core,y_ref,feature_mats,n_core,n_ref,note=assembled(s05,cohorts,domain,scenario)
        core_preds,ref_preds=predict_models(s05,y_core,feature_mats)
        if scenario=="native_frozen":
            verify_exact_reproduction(s05,prior_metrics,domain,"CORE",y_core,core_preds)
            verify_exact_reproduction(s05,prior_metrics,domain,"REFRESHER",y_ref,ref_preds)
        for cohort_name,yy,prs in (("CORE_CV",y_core,core_preds),("REFRESHER",y_ref,ref_preds)):
            result_a,result_b=audit(s05,yy,prs,cohort_name,scenario,domain,args.bootstrap_reps,SEED)
            comparisons.extend(result_a); errors.extend(result_b)
        analyses.append(dict(domain=domain,scenario=scenario,n_core=n_core,n_refresher=n_ref,note=note))
        headline=next(r for r in comparisons if r["domain"]==domain and r["scenario"]==scenario and
                      r["cohort"]=="REFRESHER" and r["reference_model"]=="HbA1c" and
                      r["comparison_model"]=="NHANES_PC2")
        print(f"  {domain:9s} | {scenario:20s} | Core {number(n_core)} / Refresher {number(n_ref)} | "
              f"PC2−HbA1c external ΔMSE {headline['delta_MSE']:+.6f} "
              f"[{headline['delta_MSE_ci95_low']:+.6f}, {headline['delta_MSE_ci95_high']:+.6f}]")
    RESULTS.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(comparisons).to_csv(RESULTS/"06_paired_model_comparisons.csv",index=False)
    pd.DataFrame(errors).to_csv(RESULTS/"06_error_decomposition.csv",index=False)
    meta=dict(script=Path(__file__).name,source_script="05_midus_frozen_glycemic_transfer.py",
              script05_sha256=hashlib.sha256(PREVIOUS.read_bytes()).hexdigest(),
              manifest05_sha256=hashlib.sha256((RESULTS/"05_manifest.json").read_bytes()).hexdigest(),
              nhanes_sha256=frozen["sha256"],seed=SEED,folds=FOLDS,ridge_alpha=ALPHA,
              bootstrap_reps=args.bootstrap_reps, models=list(MODELS),scenarios=list(SCENARIOS),
              analysis_domains=list(DOMAINS),analyses=analyses,
              prior_performance_reproduction="PASS: MSE, R2, bias, participant counts",
              no_participant_data_exported=True,original_7D_map_modified=False,
              notes="Outcome-informed centered-error decomposition diagnostic only; not individual calibration")
    (RESULTS/"06_manifest.json").write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
    report(comparisons,meta)
    print("SCRIPT 06 | Exact Script 05 reproduction PASS; locked pairwise validation complete")
    print("OUTPUT: "+str(RESULTS/"06_readout.md"))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("SCRIPT 06 FAILED: "+str(exc),file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
