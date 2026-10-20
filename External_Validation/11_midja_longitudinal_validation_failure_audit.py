#!/usr/bin/env python3
"""MIDJA longitudinal CES-D validation and prespecified failure audits."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "External_Validation" / "Results"
SCRIPT10 = Path(__file__).resolve().with_name("10_midja_independent_decomposition_cesd.py")
SEED = 20261019
FOLDS = 5
REPEATS = 20
ALPHA = 10.0
BOOT = 2000
PANELS = ("raw", "adjusted")
SCOPES = ("minimal", "expanded")
SPECS = ("prior_CESD", "HbA1c", "Hb", "hematology_raw", "raw_X4",
         "PC1_2", "PC1_2_3", "PC1_2_3_4", "A_PC1_2_plus_G")
EXTRAS = {
    "BMI": "J2CBMI",
    "Waist_cm": "J2CWST",
    "Systolic_BP": "J2CBPS1",
    "Medication_count": "J2MTM",
    "Chronic_condition_count": "J2QSYMN2",
}
RANGES = {
    "BMI": (12, 85),
    "Waist_cm": (40, 200),
    "Systolic_BP": (70, 260),
    "Medication_count": (0, 60),
    "Chronic_condition_count": (0, 80),
}


def load_script10():
    if not SCRIPT10.is_file():
        raise FileNotFoundError(
            f"Script 10 required alongside Script 11: {SCRIPT10.name}"
        )
    spec = importlib.util.spec_from_file_location("midja_script10", SCRIPT10)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_extra_covariates(module):
    import pyreadstat

    source = module.find_file("MIDJA1")
    _, meta = pyreadstat.read_sav(str(source), metadataonly=True)
    actual = {c.upper(): c for c in meta.column_names}
    requested = ["MIDJA_IDS", *EXTRAS.values()]
    missing = [c for c in requested if c.upper() not in actual]
    if missing:
        raise RuntimeError(f"MIDJA1: required audited covariates unavailable: {missing}")
    columns = [actual[c.upper()] for c in requested]
    d, _ = pyreadstat.read_sav(
        str(source), usecols=columns, user_missing=False, apply_value_formats=False
    )
    d.index = pd.Index(
        [module.canonical_id(v) for v in d[actual["MIDJA_IDS"]]],
        name="participant",
    )
    if d.index.has_duplicates:
        raise RuntimeError("Duplicate MIDJA1 participant identifiers in covariates")
    d = d.rename(columns={actual[v.upper()]: k for k, v in EXTRAS.items()})
    for k, (lo, hi) in RANGES.items():
        v = pd.to_numeric(d[k], errors="coerce").replace([np.inf, -np.inf], np.nan)
        d[k] = v.where(v.between(lo, hi))
    return d[list(EXTRAS)]


def assemble(first, second, extras, module):
    baseline, followup, matched, chronological = module.paired_rows(first, second)
    if chronological < 70:
        raise RuntimeError(f"Insufficient dated pairs: {chronological}")
    ids = baseline.index
    baseline = baseline.join(extras, how="left", validate="one_to_one")
    first = first.join(extras, how="left", validate="one_to_one")
    assert baseline.index.equals(followup.index)
    months = followup["MonthIndex"].to_numpy(float) - baseline["MonthIndex"].to_numpy(float)
    if not np.all(np.isfinite(months) & (months > 0)):
        raise RuntimeError("Chronological baseline/follow-up audit failed")
    baseline["Followup_months"] = months
    sel = first.index.isin(ids).astype(int)
    return first, baseline, followup, sel, matched, chronological


def impute_fit(train, test):
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    t = np.where(np.isfinite(train), train, med)
    v = np.where(np.isfinite(test), test, med)
    return t, v


def fit_predict(x_train, x_test, y_train, train_weight=None):
    a, b = impute_fit(x_train, x_test)
    sc = StandardScaler().fit(a)
    model = Ridge(alpha=ALPHA)
    model.fit(sc.transform(a), y_train, sample_weight=train_weight)
    return model.predict(sc.transform(b))


def covariates(d, scope, include_followup=False):
    result = [d["Age"].to_numpy(float), (d["Sex"] == 2).to_numpy(float),
              d["CESD_total"].to_numpy(float)]
    if scope == "expanded":
        result += [d[c].to_numpy(float) for c in EXTRAS]
    return np.column_stack(result)


def features(d, b, a, scope, spec, module):
    cv = covariates(d, scope)
    if spec == "prior_CESD":
        return cv
    if spec == "HbA1c":
        return np.column_stack((cv, d["HbA1c"].to_numpy(float)))
    if spec == "Hb":
        return np.column_stack((cv, d["Hb"].to_numpy(float)))
    if spec == "hematology_raw":
        return np.column_stack((cv, d.loc[:, list(module.H)].to_numpy(float)))
    if spec == "raw_X4":
        return np.column_stack((cv, module.x4(d)))
    if spec == "A_PC1_2_plus_G":
        z = module.project(d.loc[:, list(module.H)].to_numpy(float), a)
        return np.column_stack((cv, z[:, :2], d["HbA1c"].to_numpy(float)))
    k = {"PC1_2": 2, "PC1_2_3": 3, "PC1_2_3_4": 4}[spec]
    return np.column_stack((cv, module.project(module.x4(d), b)[:, :k]))


def cv_run(d, y, scope, seed, module, weights=None, train_weighted=False):
    pred = {spec: np.full(len(d), np.nan) for spec in SPECS}
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for it, iv in splitter.split(d):
        tr, te = d.iloc[it], d.iloc[iv]
        b, a = module.fit_bases(tr)
        for spec in SPECS:
            xtr = features(tr, b, a, scope, spec, module)
            xte = features(te, b, a, scope, spec, module)
            w = weights[it] if weights is not None and train_weighted else None
            pred[spec][iv] = fit_predict(xtr, xte, y[it], train_weight=w)
    if not all(np.all(np.isfinite(p)) for p in pred.values()):
        raise RuntimeError("Nonfinite held-out prediction encountered")
    return pred


def wmse(y, p, weights=None):
    z = (y-p)**2
    return float(np.mean(z)) if weights is None else float(np.average(z, weights=weights))


def paired_gain(y, p0, p1, weights=None):
    diff = (y-p0)**2 - (y-p1)**2
    return float(np.mean(diff)) if weights is None else float(np.average(diff, weights=weights))


def boot_ci(y, p0, p1, seed, weights=None):
    error_diff = (y-p0)**2 - (y-p1)**2
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = np.empty(BOOT)
    for j in range(BOOT):
        ix = rng.integers(0, n, size=n)
        boot[j] = np.mean(error_diff[ix]) if weights is None else np.average(error_diff[ix], weights=weights[ix])
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def attrition_weights(first, selection, module):
    base = np.column_stack((
        covariates(first, "expanded", include_followup=False),
        module.x4(first),
    ))
    propensity = np.full(len(first), np.nan)
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED+107)
    for it, iv in folds.split(base, selection):
        tr, te = impute_fit(base[it], base[iv])
        sc = StandardScaler().fit(tr)
        fit = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        fit.fit(sc.transform(tr), selection[it])
        propensity[iv] = fit.predict_proba(sc.transform(te))[:, 1]
    if not np.all(np.isfinite(propensity)):
        raise RuntimeError("Attrition propensity cross-fitting failed")
    p = np.clip(propensity, .05, .95)
    selected_w = float(np.mean(selection)) / p[selection.astype(bool)]
    if not np.all(np.isfinite(selected_w)):
        raise RuntimeError("Nonfinite stabilized attrition weights")
    ess = float(selected_w.sum()**2 / np.sum(selected_w**2))
    return pd.Series(selected_w, index=first.index[selection.astype(bool)]), {
        "MIDJA1_complete": int(len(first)),
        "included_dated_pair": int(selection.sum()),
        "excluded_or_no_complete_dated_pair": int(len(selection)-selection.sum()),
        "inclusion_rate": float(np.mean(selection)),
        "propensity_min": float(propensity.min()),
        "propensity_p05": float(np.quantile(propensity, .05)),
        "propensity_median": float(np.median(propensity)),
        "propensity_p95": float(np.quantile(propensity, .95)),
        "propensity_max": float(propensity.max()),
        "clip_count": int(np.sum((propensity < .05) | (propensity > .95))),
        "weight_max": float(selected_w.max()),
        "weight_effective_n": ess,
        "propensity_source": "cross-fitted baseline-only logistic regression",
        "identification": "exploratory observed-covariate attrition adjustment; MAR unverified",
    }


def smd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if min(len(a), len(b)) < 5:
        return np.nan
    s = float(np.sqrt((np.var(a, ddof=1)+np.var(b, ddof=1))/2))
    return float((np.mean(a)-np.mean(b))/s) if s > 1e-12 else np.nan


def attrition_balance(first, selection):
    vars = ["Age", "CESD_total", *EXTRAS, "Hb", "RBC", "MCV", "HbA1c"]
    mask = selection.astype(bool)
    out=[]
    for col in vars:
        x=first[col].to_numpy(float)
        out.append({"variable": col,
                    "selected_mean":float(np.nanmean(x[mask])),
                    "not_selected_mean":float(np.nanmean(x[~mask])),
                    "standardized_mean_difference":smd(x[mask], x[~mask]),
                    "selected_missing_fraction":float(np.mean(~np.isfinite(x[mask]))),
                    "not_selected_missing_fraction":float(np.mean(~np.isfinite(x[~mask])))})
    out.append({"variable":"Sex_indicator_coded_2",
                "selected_mean":float(np.mean(first.loc[mask,"Sex"]==2)),
                "not_selected_mean":float(np.mean(first.loc[~mask,"Sex"]==2)),
                "standardized_mean_difference":smd((first.loc[mask,"Sex"]==2).astype(float),
                                                      (first.loc[~mask,"Sex"]==2).astype(float)),
                "selected_missing_fraction":0.0, "not_selected_missing_fraction":0.0})
    return out


def one_panel(panel, w1, w2, extras, module, out):
    first, second = module.panel_rows(w1, panel), module.panel_rows(w2, panel)
    pool, before, after, selection, matched, dated = assemble(first, second, extras, module)
    y=after["CESD_total"].to_numpy(float)
    weights_s, selection_diag = attrition_weights(pool, selection, module)
    weights=weights_s.reindex(before.index).to_numpy(float)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("Selected population differs from paired population")
    balance=attrition_balance(pool, selection)
    (out/f"11_{panel}_attrition_balance.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(balance).to_csv(out/f"11_{panel}_attrition_balance.csv", index=False)
    results=[]
    first_seed=SEED+2
    saved={}
    for scope in SCOPES:
        for repeat in range(REPEATS):
            seed=first_seed+repeat*101
            p=cv_run(before,y,scope,seed,module)
            if repeat==0:
                saved[scope]=p
            for spec in SPECS:
                results.append({"scope":scope,"repeat":repeat,"seed":seed,
                     "model":spec,"n":module.disclose(len(y)),
                     "mse":wmse(y,p[spec]),
                     "gain_vs_prior_CESD":paired_gain(y,p["prior_CESD"],p[spec]),
                     "gain_vs_HbA1c":paired_gain(y,p["HbA1c"],p[spec]),
                     "ipw_gain_vs_prior_CESD":paired_gain(y,p["prior_CESD"],p[spec],weights)})
    results_df=pd.DataFrame(results)
    results_df.to_csv(out/f"11_{panel}_repeated_cv.csv",index=False)
    summary=[]
    for (scope,spec),g in results_df.groupby(["scope","model"],sort=False):
        summary.append({"scope":scope,"model":spec,"n":module.disclose(len(y)),
            "repeats":REPEATS,"mean_mse":float(g.mse.mean()),
            "mean_gain_vs_prior_CESD":float(g.gain_vs_prior_CESD.mean()),
            "min_gain_vs_prior_CESD":float(g.gain_vs_prior_CESD.min()),
            "max_gain_vs_prior_CESD":float(g.gain_vs_prior_CESD.max()),
            "fraction_repeats_positive":float(np.mean(g.gain_vs_prior_CESD>0)),
            "mean_gain_vs_HbA1c":float(g.gain_vs_HbA1c.mean()),
            "mean_ipw_gain_vs_prior_CESD":float(g.ipw_gain_vs_prior_CESD.mean())})
    pd.DataFrame(summary).to_csv(out/f"11_{panel}_model_summary.csv",index=False)
    comparisons=[]
    contrasts=(("prior_CESD","HbA1c"),
               ("prior_CESD","hematology_raw"),
               ("prior_CESD","raw_X4"),
               ("HbA1c","raw_X4"),
               ("prior_CESD","PC1_2"),
               ("prior_CESD","PC1_2_3_4"),
               ("prior_CESD","A_PC1_2_plus_G"))
    for scope in SCOPES:
        p=saved[scope]
        for j,(ref,cand) in enumerate(contrasts):
            for strat,wt in (("unweighted",None),("attrition_IPW",weights)):
                lo,hi=boot_ci(y,p[ref],p[cand],SEED+j*100+len(scope)*7+(wt is not None),wt)
                comparisons.append({"scope":scope,"weighting":strat,
                    "reference":ref,"candidate":cand,"n":module.disclose(len(y)),
                    "gain":paired_gain(y,p[ref],p[cand],wt),
                    "bootstrap_ci_95_low":lo,"bootstrap_ci_95_high":hi,
                    "direction":"favors_candidate" if lo>0 else "favors_reference" if hi<0 else "inconclusive"})
    pd.DataFrame(comparisons).to_csv(out/f"11_{panel}_paired_uncertainty.csv",index=False)
    diagnostics=[]
    sensitivity_rows=[]
    for col in EXTRAS:
        diagnostics.append({"variable":col,"n":module.disclose(len(before)),
                            "missing_fraction":float(before[col].isna().mean()),
                            "median":float(before[col].median()) if before[col].notna().any() else np.nan})
    pd.DataFrame(diagnostics).to_csv(out/f"11_{panel}_covariate_coverage.csv",index=False)
    # Prespecified sensitivity: remove extreme baseline glycemic/hematologic values.
    plausible=(before["HbA1c"].between(4,10)&before["Hb"].between(8,18)).to_numpy()
    sensitivity={"subset":"HbA1c_4to10_Hb_8to18","n":int(plausible.sum()),
                 "exclusions":int((~plausible).sum())}
    if plausible.sum()>=70:
        subd=before.iloc[np.flatnonzero(plausible)]
        suby=y[plausible]
        for scope in SCOPES:
            for rep in range(10):
                pp=cv_run(subd,suby,scope,first_seed+rep*101,module)
                for spec in ("HbA1c","raw_X4","PC1_2","PC1_2_3_4"):
                    sensitivity_rows.append({"scope":scope,"repeat":rep,"model":spec,
                       "n":module.disclose(len(suby)),
                       "gain_vs_prior_CESD":paired_gain(suby,pp["prior_CESD"],pp[spec])})
    pd.DataFrame(sensitivity_rows).to_csv(out/f"11_{panel}_extreme_value_sensitivity.csv",index=False)
    info={"panel":panel,"MIDJA1_complete":len(first),"MIDJA2_complete":len(second),
          "paired_all":matched,"paired_strict_later":dated,
          "followup_gap_months_median":float(before["Followup_months"].median()),
          "followup_gap_months_min":float(before["Followup_months"].min()),
          "followup_gap_months_max":float(before["Followup_months"].max()),
          "CESD_baseline_mean":float(before["CESD_total"].mean()),
          "CESD_followup_mean":float(np.mean(y)),
          "CESD_mean_change":float(np.mean(y-before["CESD_total"].to_numpy(float))),
          "attrition":selection_diag,"sensitivity":sensitivity}
    write_report(panel,info,pd.DataFrame(summary),pd.DataFrame(comparisons),balance,out)
    return info


def write_report(panel, info, summ, contrast, balance, out):
    lines=["# Script 11 — MIDJA longitudinal validation and failure audit", "",
        "## Study", "",
        f"- HbA1c panel: {panel}; MIDJA1 complete N={info['MIDJA1_complete']}; MIDJA2 complete N={info['MIDJA2_complete']}.",
        f"- Strict chronological complete pairs: {info['paired_strict_later']}; median follow-up interval: {info['followup_gap_months_median']:.1f} months.",
        f"- Baseline CES-D mean: {info['CESD_baseline_mean']:.3f}; later CES-D mean: {info['CESD_followup_mean']:.3f}.",
        "- All predictive variables measured at MIDJA1; elapsed follow-up time is descriptive and never used to predict.",
        "- Baseline PCA refitted inside each participant-held-out training fold; no NHANES/MIDUS loadings used.",
        f"- Repeated five-fold CV: {REPEATS} partitions; Ridge alpha={ALPHA:g}; identical splits for comparisons.", "",
        "## Repeated-CV predictive contribution", "",
        "ΔMSE = baseline MSE − candidate MSE; positive favors the candidate.", "",
        "| Adjustment | Model | Mean ΔMSE vs baseline | Minimum to maximum across CV splits | Fraction positive | Mean attrition-weighted ΔMSE |",
        "|---|---|---:|---|---:|---:|"]
    for _,r in summ.iterrows():
        lines.append(f"| {r.scope} | {r.model} | {r.mean_gain_vs_prior_CESD:+.4f} | [{r.min_gain_vs_prior_CESD:+.4f}, {r.max_gain_vs_prior_CESD:+.4f}] | {r.fraction_repeats_positive:.2f} | {r.mean_ipw_gain_vs_prior_CESD:+.4f} |")
    lines += ["", "## Single-partition paired descriptive intervals", "",
        "These bootstrap intervals resample the 211 or fewer held-out participant-level squared-error differences for one prespecified CV partition. They do not refit models or adjust for multiple testing/survey design.", "",
        "| Adjustment | Weighting | Comparison | ΔMSE | Descriptive 95% interval |",
        "|---|---|---|---:|---|"]
    for _,r in contrast.iterrows():
        lines.append(f"| {r.scope} | {r.weighting} | {r.reference} → {r.candidate} | {r.gain:+.4f} | [{r.bootstrap_ci_95_low:+.4f}, {r.bootstrap_ci_95_high:+.4f}] |")
    a=info['attrition']
    lines += ["", "## Attrition", "",
      f"- Dated-pair eligibility: {a['included_dated_pair']}/{a['MIDJA1_complete']} = {a['inclusion_rate']:.3f}.",
      f"- Cross-fitted selection propensity range: [{a['propensity_min']:.3f}, {a['propensity_max']:.3f}]; clipped at 0.05–0.95 for evaluation.",
      f"- Attrition-weight effective sample size: {a['weight_effective_n']:.1f}; maximum weight: {a['weight_max']:.3f}; propensity clips: {a['clip_count']}.",
      "- Selection propensity depends on recorded baseline measures only; inverse-probability weighting does not eliminate unmeasured attrition bias.",
      "", "## Prespecified robustness and interpretation", "",
      f"- Exclusion sensitivity: HbA1c [4,10] and Hb [8,18], retained N={info['sensitivity']['n']}; see extreme_value_sensitivity CSV for supplementary split results.",
      "- Expanded baseline adjusts for BMI, waist, baseline systolic blood pressure, medication count, chronic-condition count, ; missing values imputed using training folds only.",
      "- Compare full physiology against both baseline-only and HbA1c-only models; raw four-biomarker and four-PC representations contain the same linear information.",
      "- A positive held-out improvement is predictive association, not causality or demonstration of physiological mechanisms.",
      "- Case-bootstrap intervals are descriptive; repetitions with different random CV splits are dependent and are not independent replications.",
      "- BMI, blood pressure, RBC assay scale, CES-D score coding, laboratory harmonization, survey weights, and sampling/attrition mechanisms remain unverified.",
      "- No external prospective test set exists in this audit; validation takes place within the matched MIDJA cohort.",
      "- Strong support would require stable positive gains beyond both baseline and simple markers under expanded covariates, plausible exclusion, attrition sensitivity, and subsequent truly independent replication.", ""]
    (out/f"11_{panel}_readout.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    module=load_script10()
    w1,src1,all_first=module.read_wave("MIDJA1")
    w2,src2,_=module.read_wave("MIDJA2")
    extras=read_extra_covariates(module)
    OUT.mkdir(parents=True,exist_ok=True)
    results={}
    for panel in PANELS:
        info=one_panel(panel,w1,w2,extras,module,OUT)
        results[panel]=info
        print(f"  {panel:8s} | matched N={info['paired_strict_later']} | attrition IPW effective N={info['attrition']['weight_effective_n']:.1f}")
    manifest={"script":Path(__file__).name,"design":"Longitudinal MIDJA participant-level repeated-CV robustness and attrition audits",
        "seed":SEED,"cv_folds":FOLDS,"cv_repeats":REPEATS,"ridge_alpha":ALPHA,
        "bootstrap_replicates":BOOT,"source_script":SCRIPT10.name,
        "source_files":{"MIDJA1":src1,"MIDJA2":src2},
        "optional_variables":EXTRAS,"ranges":RANGES,"panels":results,
        "uses_future_biomarkers_as_predictors":False,
        "nhanes_or_midus_fit_used":False,"causal_identification":False,
        "survey_design_adjusted":False,"clinical_coding_and_assays_verified":False,
        "independent_external_prospective_cohort":False,"individual_records_exported":False}
    (OUT/"11_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("SCRIPT 11 | MIDJA longitudinal failure audit | complete")
    print(f"OUTPUT: {OUT/'11_raw_readout.md'}")
    print(f"        {OUT/'11_adjusted_readout.md'}")


if __name__=="__main__":
    main()
